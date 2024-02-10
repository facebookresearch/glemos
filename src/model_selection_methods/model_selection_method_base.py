# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from utils import create_eval_dict, eval_metrics_single_graph


class ModelSelectionMethodBase:
    def __init__(self, name=""):
        self.name = name
        self.eval_dict = None

    def eval_P(self, P_test, P_hat):
        if not hasattr(self, 'eval_dict') or self.eval_dict is None:
            self.eval_dict = create_eval_dict()

        P_test_non_nan_mask = ~np.isnan(P_test)
        for i in range(0, P_test.shape[0]):
            p_test = P_test[i, :]
            p_test_non_nan_mask = P_test_non_nan_mask[i, :]
            p_hat = P_hat[i, :]

            for metric, metric_score in eval_metrics_single_graph(p_test[p_test_non_nan_mask],
                                                                  p_hat[p_test_non_nan_mask]).items():
                self.eval_dict[metric].append(metric_score)

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        raise NotImplementedError
