# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json

import numpy as np
import sklearn

import settings
from testbeds.testbed_base import Testbed


class SmallToLargeTestbed(Testbed):
    testbed_dir = "small-to-large-testbed"

    def __init__(self,
                 task,
                 perf_metric,
                 perf_matrices,
                 meta_feat,
                 meta_features,
                 graph_names,
                 all_models,
                 num_nodes_threshold=10000,
                 n_splits=1,
                 graph_domain=None):
        super().__init__(task, perf_metric, perf_matrices, meta_feat, meta_features, n_splits, graph_names, all_models,
                         graph_domain, SmallToLargeTestbed.testbed_dir)
        self.num_nodes_threshold = num_nodes_threshold
        assert len(self.graph_names) == len(self.perf_mat) == len(self.meta_feat_mat), \
            (len(self.graph_names), len(self.perf_mat), len(self.meta_feat_mat))
        assert self.n_splits == 1, self.n_splits  # this testbed uses one pair of train split (small graphs) and test split (large graphs)

    def testbed_settings(self):
        return {
            'n_splits': self.n_splits,
            'num_nodes_threshold': self.num_nodes_threshold,
            'exclude_graphs_and_models_with_missing_performance': self.exclude_graphs_and_models_with_missing_performance,
        }

    def generate(self):
        from graphs.graphset import GraphSet
        graph_set = GraphSet()
        graphs = {
            'link-pred': graph_set.link_prediction_graphs(),
            'node-class': graph_set.node_classification_graphs(),
        }[self.task]

        graph_name_to_graph = {graph.name: graph for graph in graphs}
        graph_num_nodes = []
        for graph_name in self.graph_names:
            if graph_name in graph_name_to_graph:
                graph_num_nodes.append(graph_name_to_graph[graph_name].num_nodes)
        graph_num_nodes = np.array(graph_num_nodes)

        small_graph_indices = np.flatnonzero(graph_num_nodes < self.num_nodes_threshold)
        assert len(small_graph_indices) > 0, self.num_nodes_threshold
        large_graph_indices = np.flatnonzero(graph_num_nodes >= self.num_nodes_threshold)
        assert len(large_graph_indices) > 0, self.num_nodes_threshold

        np.savetxt(self.train_index_path(split_i=0), small_graph_indices, fmt='%i', delimiter=',')
        np.savetxt(self.test_index_path(split_i=0), large_graph_indices, fmt='%i', delimiter=',')

        print(f"num small {self.task} graphs (# nodes < {self.num_nodes_threshold}): {len(small_graph_indices)}")
        print(f"num large {self.task} graphs (# nodes >= {self.num_nodes_threshold}): {len(large_graph_indices)}")

        self.testbed_root.mkdir(parents=True, exist_ok=True)
        with (self.testbed_root / 'settings.txt').open('w') as f:
            json.dump(self.testbed_settings(), f)

        return self

    def get_train_perf_mat(self):
        return self.perf_mat

    def get_test_perf_mat(self):
        return self.perf_mat

    def get_result_dir_path(self):
        return settings.RESULTS_ROOT / self.meta_feat / self.testbed_dir / f"{self.task}_{self.perf_metric}"

    def load(self):
        settings = self.load_settings()
        assert settings['n_splits'] == 1, settings

        train_perf_mat = self.get_train_perf_mat()
        test_perf_mat = self.get_test_perf_mat()
        print("train_perf_mat:", train_perf_mat.shape)
        print("test_perf_mat:", test_perf_mat.shape)
        assert train_perf_mat.shape == test_perf_mat.shape, (train_perf_mat.shape, test_perf_mat.shape)

        P_splits, M_splits, D_splits = [], [], []

        train_index = np.loadtxt(self.train_index_path(split_i=0), dtype=int)
        test_index = np.loadtxt(self.test_index_path(split_i=0), dtype=int)

        P_train = train_perf_mat[train_index].reshape(-1, train_perf_mat.shape[1])
        P_test = test_perf_mat[test_index].reshape(-1, test_perf_mat.shape[1])
        P_splits.append({"train": P_train,
                         "train_imputed": P_train,
                         "train_full": P_train,
                         "test": P_test})

        M_norm = sklearn.preprocessing.minmax_scale(self.meta_feat_mat.copy(), axis=0)  # scale each col (meta-feat)
        M_train = M_norm[train_index].reshape(-1, M_norm.shape[1])
        M_test = M_norm[test_index].reshape(-1, M_norm.shape[1])
        M_splits.append({"train": M_train, "test": M_test})

        if self.graph_domain is not None:
            graph_domain = np.array(self.graph_domain)
            D_train, D_test = graph_domain[train_index], graph_domain[test_index]
        else:
            D_train, D_test = None, None
        D_splits.append({"train": D_train, "test": D_test})

        return P_splits, M_splits, D_splits


if __name__ == '__main__':
    from testbeds.workspace import perf_matrices, meta_features, graph_names, all_models

    for task in ['link-pred', 'node-class']:
        SmallToLargeTestbed(
            task=task,
            num_nodes_threshold=10000,
            perf_metric='map', perf_matrices=perf_matrices,
            meta_feat='regular', meta_features=meta_features,
            graph_names=graph_names, all_models=all_models,
        ).generate().load()

