# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from .model_selection_method_base import ModelSelectionMethodBase


# noinspection PyMethodMayBeStatic
class GlobalBestAveragePerf(ModelSelectionMethodBase):
    """
    GlobalBestAveragePerf baseline uses the model with the best average performance across all training graphs
    """

    def __init__(self, name='GB-Avg', use_imputed=False):
        super().__init__(name)
        self.use_imputed = use_imputed

        self.P_train = None
        self.P_train_imputed = None
        self.P_train_mean = None
        self.P_train_imputed_mean = None

    def __str__(self):
        return f"{self.name}(use_imputed={self.use_imputed})"

    def fit(self, P_train, P_train_imputed):
        self.P_train = P_train
        self.P_train_imputed = P_train_imputed

        self.P_train_mean = np.nanmean(P_train, axis=0)  # shape=(1, # models). Mean perf of each model over all train graphs
        self.P_train_imputed_mean = np.mean(P_train_imputed, axis=0)  # shape=(1, # models). Mean perf of each model over all train graphs

    def predict(self, P_test):
        if self.use_imputed:
            global_avg_best = np.tile(self.P_train_imputed_mean, (P_test.shape[0], 1))
        else:
            global_avg_best = np.tile(self.P_train_mean, (P_test.shape[0], 1))
        return global_avg_best

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(P_train, P_train_imputed)
        P_hat = self.predict(P_test)
        self.eval_P(P_test, P_hat)
        return P_hat
