# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

import settings
from testbeds.testbed_base import Testbed


class OutOfDomainTestbed(Testbed):
    testbed_dir = "out-of-domain-testbed"

    TEST_MODE_LOGO = 'leave-one-group-out'
    TEST_MODE_KFOLD = 'group-k-fold'

    def __init__(self,
                 task,
                 perf_metric,
                 perf_matrices,
                 meta_feat,
                 meta_features,
                 graph_domain,
                 graph_names,
                 all_models,
                 n_splits=None,
                 test_mode=TEST_MODE_KFOLD):
        super().__init__(task, perf_metric, perf_matrices, meta_feat, meta_features, n_splits, graph_names, all_models,
                         graph_domain, OutOfDomainTestbed.testbed_dir)
        self.testbed_root = settings.TESTBED_ROOT / OutOfDomainTestbed.testbed_dir / task / test_mode
        self.test_mode = test_mode
        assert test_mode in [self.TEST_MODE_LOGO, self.TEST_MODE_KFOLD]
        assert test_mode == self.TEST_MODE_LOGO or n_splits is not None

    def testbed_settings(self):
        if self.test_mode == self.TEST_MODE_LOGO:
            n_splits = len(self.domain_name_to_index())
        else:
            n_splits = self.n_splits

        return {
            'n_splits': n_splits,
            'test_mode': self.test_mode,
            'domain_name_to_index': self.domain_name_to_index(),
            'exclude_graphs_and_models_with_missing_performance': self.exclude_graphs_and_models_with_missing_performance,
        }

    def domain_name_to_index(self):
        return {d_name: d_idx for d_idx, d_name in enumerate(sorted(set(self.graph_domain)))}

    def k_fold_splits(self):
        domain_name_to_index = self.domain_name_to_index()
        domain_indices = [domain_name_to_index[domain_name] for domain_name in self.graph_domain]

        if self.test_mode == self.TEST_MODE_KFOLD:
            kf = GroupKFold(self.n_splits)
            splits = kf.split(self.perf_mat, groups=domain_indices)
        elif self.test_mode == self.TEST_MODE_LOGO:
            kf = LeaveOneGroupOut()
            splits = kf.split(self.perf_mat, groups=domain_indices)
        else:
            raise ValueError(f"Invalid test mode: {self.test_mode}")

        return splits

    def generate(self):
        for i, (train_index, test_index) in enumerate(self.k_fold_splits()):
            np.savetxt(self.train_index_path(i), train_index, fmt='%i', delimiter=',')
            np.savetxt(self.test_index_path(i), test_index, fmt='%i', delimiter=',')

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
        """Load k-fold splits for model performances, meta-graph features, and graph domains (if available)"""

        settings = self.load_settings()
        train_perf_mat = self.get_train_perf_mat()
        test_perf_mat = self.get_test_perf_mat()
        print("train_perf_mat:", train_perf_mat.shape)
        print("test_perf_mat:", test_perf_mat.shape)
        assert train_perf_mat.shape == test_perf_mat.shape, (train_perf_mat.shape, test_perf_mat.shape)
        P_splits, M_splits, D_splits = [], [], []
        for i in range(settings['n_splits']):
            train_index = np.loadtxt(self.train_index_path(i), dtype=int)
            test_index = np.loadtxt(self.test_index_path(i), dtype=int)

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
    from testbeds.workspace import y_graph_domain, perf_matrices, meta_features, graph_names, all_models

    for task in ['link-pred', 'node-class']:
        OutOfDomainTestbed(
            task=task,
            perf_metric='map', perf_matrices=perf_matrices,
            meta_feat='regular', meta_features=meta_features,
            graph_domain=y_graph_domain, graph_names=graph_names,
            all_models=all_models, n_splits=5,
            test_mode=OutOfDomainTestbed.TEST_MODE_KFOLD
        ).generate().load()

        OutOfDomainTestbed(
            task=task,
            perf_metric='map', perf_matrices=perf_matrices,
            meta_feat='regular', meta_features=meta_features,
            graph_domain=y_graph_domain, graph_names=graph_names,
            all_models=all_models, n_splits=None,
            test_mode=OutOfDomainTestbed.TEST_MODE_LOGO
        ).generate().load()
