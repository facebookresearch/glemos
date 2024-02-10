# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import numpy as np
from abc import ABC, abstractmethod

from sklearn.model_selection import KFold, StratifiedKFold

import settings


class Testbed(ABC):
    def __init__(self, task, perf_metric, perf_matrices, meta_feat, meta_features, n_splits,
                 graph_names, all_models, graph_domain, testbed_dir,
                 exclude_graphs_and_models_with_missing_performance=False):
        self.task = task
        self.testbed_dir = testbed_dir
        self.testbed_root = settings.TESTBED_ROOT / testbed_dir / task
        self.perf_metric = perf_metric
        self.exclude_graphs_and_models_with_missing_performance = exclude_graphs_and_models_with_missing_performance

        perf_mat_sub, meta_feat_mat_sub, graph_names_sub, all_models_sub, graph_domain_sub = None, None, None, None, None
        if exclude_graphs_and_models_with_missing_performance:
            perf_mat_sub, meta_feat_mat_sub, graph_names_sub, all_models_sub, graph_domain_sub = \
                Testbed.exclude_graphs_and_models_with_missing_performance(perf_matrices, task, perf_metric,
                                                                           meta_features, meta_feat, graph_names,
                                                                           all_models, graph_domain)

        if exclude_graphs_and_models_with_missing_performance:
            self.perf_mat = perf_mat_sub
        else:
            self.perf_mat = perf_matrices[task][perf_metric]

        self.meta_feat = meta_feat  # meta feature type, e.g., "regular"

        if exclude_graphs_and_models_with_missing_performance:
            self.meta_feat_mat = meta_feat_mat_sub
        else:
            self.meta_feat_mat = meta_features[task][meta_feat]

        self.n_splits = n_splits

        if exclude_graphs_and_models_with_missing_performance:
            self.graph_names = graph_names_sub
        else:
            self.graph_names = graph_names[task]

        if exclude_graphs_and_models_with_missing_performance:
            self.all_models = all_models_sub
        else:
            self.all_models = all_models[task]

        if exclude_graphs_and_models_with_missing_performance:
            self.graph_domain = graph_domain_sub
        else:
            self.graph_domain = graph_domain[task] if graph_domain is not None else None

        assert len(self.perf_mat) == len(self.meta_feat_mat) == len(self.graph_names)  # num-graphs
        assert self.perf_mat.shape[1] == len(self.all_models)  # num-models
        assert n_splits is None or len(self.perf_mat) >= n_splits, (len(self.perf_mat), n_splits)

    @classmethod
    def exclude_graphs_and_models_with_missing_performance(cls, perf_matrices, task, perf_metric,
                                                           meta_features, meta_feat, graph_names, all_models,
                                                           graph_domain):
        """
        Exclude graphs and models with missing performance records.
        Currently, this is used for the fully-observed and partially-observed testbeds to work with fully-observed graphs and models
        """
        perf_mat = perf_matrices[task][perf_metric]
        meta_feats_list = list(meta_features[task].values())
        assert len(perf_mat) == len(meta_feats_list[0]), (len(perf_mat), len(meta_feats_list[0]))  # num-graphs

        if graph_domain is not None:
            assert len(graph_domain[task]) == len(perf_mat), (len(graph_domain[task]), len(perf_mat))
        else:
            graph_domain = None

        fully_observed_model_indices, partially_observed_model_indices = [], []
        for model_i, perf_over_model in enumerate(perf_mat.T):
            if np.count_nonzero(np.isnan(perf_over_model)) == 0:
                fully_observed_model_indices.append(model_i)
            else:
                partially_observed_model_indices.append(model_i)
        assert len(fully_observed_model_indices) > 0

        perf_mat = perf_mat.copy()
        perf_mat_sub = perf_mat[:, fully_observed_model_indices]  # exclude partially observed models first

        fully_observed_graph_indices, partially_observed_graph_indices = [], []
        for graph_i, perf_over_graph in enumerate(perf_mat_sub):
            if np.count_nonzero(np.isnan(perf_over_graph)) == 0:
                fully_observed_graph_indices.append(graph_i)
            else:
                partially_observed_graph_indices.append(graph_i)
        assert len(fully_observed_graph_indices) > 0

        perf_mat_sub2 = perf_mat_sub[fully_observed_graph_indices]

        meta_feat_mat = meta_features[task][meta_feat].copy()
        meta_feat_mat_sub = meta_feat_mat[fully_observed_graph_indices, :]

        graph_names = np.array(graph_names[task].copy())
        graph_names_sub = graph_names[fully_observed_graph_indices].tolist()

        all_models = np.array(all_models[task].copy())
        all_models_sub = all_models[fully_observed_model_indices].tolist()

        if graph_domain is not None:
            graph_domain = np.array(graph_domain[task])
            graph_domain_sub = graph_domain[fully_observed_graph_indices].tolist()
        else:
            graph_domain_sub = graph_domain

        # if partially_observed_graph_indices:
        #     print(f"partially observed graphs: {', '.join([graph_names[graph_i] for graph_i in partially_observed_graph_indices])}")
        # if partially_observed_model_indices:
        #     print(f"partially observed models: {', '.join([all_models[model_i] for model_i in partially_observed_model_indices])}")

        return perf_mat_sub2, meta_feat_mat_sub, graph_names_sub, all_models_sub, graph_domain_sub

    @abstractmethod
    def testbed_settings(self):
        raise NotImplementedError

    def train_index_path(self, split_i, mkdir=True):
        if mkdir and not self.testbed_root.exists():
            self.testbed_root.mkdir(parents=True, exist_ok=True)
        return self.testbed_root / f"split{split_i}-train-index.csv"

    def test_index_path(self, split_i, mkdir=True):
        if mkdir and not self.testbed_root.exists():
            self.testbed_root.mkdir(parents=True, exist_ok=True)
        return self.testbed_root / f"split{split_i}-test-index.csv"

    @abstractmethod
    def get_train_perf_mat(self):
        raise NotImplementedError

    @abstractmethod
    def get_test_perf_mat(self):
        raise NotImplementedError

    def k_fold_splits(self):
        if self.graph_domain is None:
            kf = KFold(n_splits=self.n_splits, random_state=1, shuffle=True)
            splits = kf.split(self.perf_mat)
        else:
            kf = StratifiedKFold(n_splits=self.n_splits, random_state=1, shuffle=True)
            splits = kf.split(self.perf_mat, self.graph_domain)
        return splits

    @abstractmethod
    def generate(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    def load_settings(self):
        assert self.testbed_root.exists(), self.testbed_root
        settings = self.testbed_settings()
        with (self.testbed_root / 'settings.txt').open('r') as f:
            saved_settings = json.load(f)
            assert saved_settings == settings, f"Saved testbed settings ({saved_settings}) are different from the given settings ({settings})"
        return saved_settings
