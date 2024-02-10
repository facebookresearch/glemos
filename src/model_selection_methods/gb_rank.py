# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pandas as pd

from .model_selection_method_base import ModelSelectionMethodBase


# noinspection PyMethodMayBeStatic
class GlobalBestAverageRank(ModelSelectionMethodBase):
    """
    For each dataset, we compute the rank of all available algorithms.
    Then for each algorithm, we compute its average ranking over all datasets, and
    algorithms are ordered according to the average ranking.
    """

    def __init__(self, name='AvgRank', use_imputed=False):
        super().__init__(name)
        self.use_imputed = use_imputed

        self.P_train = None
        self.P_train_imputed = None
        self.model_avg_ranking = None

    def __str__(self):
        return f"{self.name}(use_imputed={self.use_imputed})"

    def fit(self, P_train, P_train_imputed):
        self.P_train = np.array(P_train.copy())
        self.P_train_imputed = np.array(P_train_imputed.copy())

        df_P_train = pd.DataFrame.from_records(self.P_train_imputed if self.use_imputed else self.P_train)  # graph-by-model
        P_train_ranks = df_P_train.rank(axis=1, ascending=True, pct=True).to_numpy()  # higher performance is assigned a larger ranking (in percentile)
        self.model_avg_ranking = np.nanmean(P_train_ranks, axis=0)  # shape=(# models,)

    def predict(self, P_test):
        assert self.P_train is not None and self.model_avg_ranking is not None

        P_hat_test = np.zeros_like(P_test)
        for i in range(0, P_test.shape[0]):
            P_hat_test[i, :] = self.model_avg_ranking

        return P_hat_test

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(P_train, P_train_imputed)
        P_hat = self.predict(P_test)
        self.eval_P(P_test, P_hat)
        return P_hat
