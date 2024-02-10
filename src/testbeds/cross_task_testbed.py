# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import sklearn
import numpy as np

import settings


class CrossTaskTestbed:
    def __init__(self,
                 source_task: str,  # task for training
                 source_perf_metric: str,
                 target_task: str,  # task for testing
                 target_perf_metric: str,
                 perf_matrices: dict,
                 meta_feat: str,
                 meta_features: dict,
                 graph_names: dict,
                 models: dict,
                 testbed_dir="cross-task-testbed"):
        self.source_task = source_task
        self.source_perf_metric = source_perf_metric
        self.source_perf_mat = perf_matrices[source_task][source_perf_metric]

        self.target_task = target_task
        self.target_perf_metric = target_perf_metric
        self.target_perf_mat = perf_matrices[target_task][target_perf_metric]

        self.meta_feat = meta_feat
        self.source_meta_feat_mat = meta_features[source_task][meta_feat]
        self.target_meta_feat_mat = meta_features[target_task][meta_feat]

        self.graph_names = graph_names
        self.source_graph_names = graph_names[source_task]
        self.target_graph_names = graph_names[target_task]

        self.models = models
        self.source_models = models[source_task]
        self.target_models = models[target_task]

        self.testbed_dir = testbed_dir
        self.testbed_root = settings.TESTBED_ROOT / testbed_dir / f"{self.source_task}-to-{self.target_task}"

        assert source_task != target_task, (source_task, target_task)
        assert len(self.source_perf_mat) == len(self.source_meta_feat_mat), \
            (len(self.source_perf_mat), len(self.source_meta_feat_mat))
        assert self.source_perf_mat.shape == (len(self.source_graph_names), len(self.source_models))
        assert len(self.target_perf_mat) == len(self.target_meta_feat_mat), \
            (len(self.target_perf_mat), len(self.target_meta_feat_mat))
        assert self.target_perf_mat.shape == (len(self.target_graph_names), len(self.target_models))

    def testbed_settings(self):
        return {
            'source_task': self.source_task,
            'target_task': self.target_task,
            'models': self.models,
            'graph_names': self.graph_names,
        }

    def load_settings(self):
        assert self.testbed_root.exists()
        settings = self.testbed_settings()
        with (self.testbed_root / 'settings.txt').open('r') as f:
            saved_settings = json.load(f)
            assert saved_settings == settings, f"Saved testbed settings ({saved_settings}) are different from the given settings ({settings})"
        return saved_settings

    def get_train_graph_names(self):
        source_graph_names = set(self.source_graph_names)
        target_graph_names = set(self.target_graph_names)

        if source_graph_names.issubset(target_graph_names):
            return sorted(source_graph_names)
        elif target_graph_names.issubset(source_graph_names):
            return sorted(source_graph_names.difference(target_graph_names))
        else:
            return sorted(source_graph_names)

    def get_test_graph_names(self):
        source_graph_names = set(self.source_graph_names)
        target_graph_names = set(self.target_graph_names)

        if source_graph_names.issubset(target_graph_names):
            return sorted(target_graph_names.difference(source_graph_names))
        elif target_graph_names.issubset(source_graph_names):
            return sorted(target_graph_names)
        else:
            return sorted(target_graph_names.difference(source_graph_names))

    def get_common_models(self):
        return sorted(set(self.source_models).intersection(set(self.target_models)))

    def get_train_perf_mat(self):
        return self.source_perf_mat

    def get_test_perf_mat(self):
        return self.target_perf_mat

    def get_train_meta_feat_mat(self):
        return self.source_meta_feat_mat

    def get_test_meta_feat_mat(self):
        return self.target_meta_feat_mat

    def get_graph_index_path(self, train_or_test):
        return {
            'train': self.testbed_root / f"source-train-graph-index.csv",
            'test': self.testbed_root / f"target-test-graph-index.csv",
        }[train_or_test]

    def get_model_index_path(self, train_or_test):
        return {
            'train': self.testbed_root / f"source-train-model-index.csv",
            'test': self.testbed_root / f"target-test-model-index.csv",
        }[train_or_test]

    def get_result_dir_path(self):
        return settings.RESULTS_ROOT / self.meta_feat / self.testbed_dir / \
            f"{self.source_task}_{self.source_perf_metric}-to-{self.target_task}_{self.target_perf_metric}"

    def generate(self):
        source_graph_name_to_i = {graph_name: graph_i for graph_i, graph_name in enumerate(self.source_graph_names)}
        train_graph_names = self.get_train_graph_names()
        print("train_graph_names:", len(train_graph_names), train_graph_names)
        source_train_graph_index = np.array([source_graph_name_to_i[graph_name] for graph_name in train_graph_names])

        target_graph_name_to_i = {graph_name: graph_i for graph_i, graph_name in enumerate(self.target_graph_names)}
        test_graph_names = self.get_test_graph_names()
        print("test_graph_names:", len(test_graph_names), test_graph_names)
        target_test_graph_index = np.array([target_graph_name_to_i[graph_name] for graph_name in test_graph_names])

        self.testbed_root.mkdir(parents=True, exist_ok=True)
        np.savetxt(self.get_graph_index_path('train'), source_train_graph_index, fmt='%i', delimiter=',')
        np.savetxt(self.get_graph_index_path('test'), target_test_graph_index, fmt='%i', delimiter=',')

        common_models = self.get_common_models()
        print("common_models:", len(common_models))

        source_model_to_i = {model: model_i for model_i, model in enumerate(self.source_models)}
        source_train_model_index = [source_model_to_i[model] for model in common_models]

        target_model_to_i = {model: model_i for model_i, model in enumerate(self.target_models)}
        target_test_model_index = [target_model_to_i[model] for model in common_models]

        np.savetxt(self.get_model_index_path('train'), source_train_model_index, fmt='%i', delimiter=',')
        np.savetxt(self.get_model_index_path('test'), target_test_model_index, fmt='%i', delimiter=',')

        with (self.testbed_root / 'settings.txt').open('w') as f:
            json.dump(self.testbed_settings(), f)

        return self

    def load(self):
        self.load_settings()

        train_perf_mat = self.get_train_perf_mat()
        test_perf_mat = self.get_test_perf_mat()

        P_splits, M_splits, D_splits = [], [], []

        train_graph_index = np.loadtxt(self.get_graph_index_path('train'), dtype=int)
        train_model_index = np.loadtxt(self.get_model_index_path('train'), dtype=int)
        test_graph_index = np.loadtxt(self.get_graph_index_path('test'), dtype=int)
        test_model_index = np.loadtxt(self.get_model_index_path('test'), dtype=int)

        P_train = train_perf_mat[train_graph_index, :][:, train_model_index].reshape(len(train_graph_index), len(train_model_index))
        print("P_train:", P_train.shape)
        P_test = test_perf_mat[test_graph_index, :][:, test_model_index].reshape(len(test_graph_index), len(test_model_index))
        print("P_test:", P_test.shape)
        P_splits.append({"train": P_train,
                         "train_imputed": P_train,
                         "train_full": P_train,
                         "test": P_test})

        train_M_norm = sklearn.preprocessing.minmax_scale(self.get_train_meta_feat_mat().copy(), axis=0)  # scale each col (meta-feat)
        test_M_norm = sklearn.preprocessing.minmax_scale(self.get_test_meta_feat_mat().copy(), axis=0)  # scale each col (meta-feat)
        M_train = train_M_norm[train_graph_index, :].reshape(-1, train_M_norm.shape[1])
        M_test = test_M_norm[test_graph_index, :].reshape(-1, test_M_norm.shape[1])
        M_splits.append({"train": M_train, "test": M_test})

        D_splits.append({"train": None, "test": None})

        return P_splits, M_splits, D_splits


if __name__ == '__main__':
    from testbeds.workspace import perf_matrices, meta_features, graph_names, all_models

    for source_task, target_task in [('link-pred', 'node-class'), ('node-class', 'link-pred')]:
        print(f"\nsource_task={source_task}, target_task={target_task}")
        CrossTaskTestbed(
            source_task=source_task,
            source_perf_metric='map',
            target_task=target_task,
            target_perf_metric='map',
            perf_matrices=perf_matrices,
            meta_feat='regular',
            meta_features=meta_features,
            graph_names=graph_names,
            models=all_models,
        ).generate().load()
