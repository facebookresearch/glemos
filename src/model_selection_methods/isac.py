# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from sklearn.cluster import KMeans

from .model_selection_method_base import ModelSelectionMethodBase


class ISAC(ModelSelectionMethodBase):
    """
    At training time, ISAC clusters meta-train datasets using meta features.
    At test time, ISAC finds the cluster closest to the test graph and
    selects the model with the largest average performance over all graphs in that cluster.
    """
    def __init__(self, name='ISAC', n_clusters=5, use_imputed=False):
        super().__init__(name)
        self.n_clusters = n_clusters
        self.use_imputed = use_imputed
        self.clustering = None
        self.M_train = None
        self.P_train = None
        self.P_train_imputed = None

    def __str__(self):
        return f"{self.name}(n_clusters={self.n_clusters}, use_imputed={self.use_imputed})"

    def fit(self, M_train, P_train, P_train_imputed):
        self.clustering = KMeans(n_clusters=self.n_clusters, n_init='auto')
        self.clustering.fit(M_train)
        self.M_train = M_train
        self.P_train = P_train
        self.P_train_imputed = P_train_imputed

    def predict(self, M_test):
        assert self.P_train is not None
        predicted_clusters = self.clustering.predict(M_test)

        ISAC_score = np.zeros((len(M_test), self.P_train.shape[1]))
        train_clusters = self.clustering.labels_  # shape=(# training graphs,)

        for i in range(ISAC_score.shape[0]):
            train_data_index = np.where(train_clusters == predicted_clusters[i])[0]
            if self.use_imputed:
                train_data_performance = self.P_train_imputed[train_data_index, :]  # submatrix of performances for points in same cluster
                ISAC_score[i, :] = np.mean(train_data_performance, axis=0)
            else:
                train_data_performance = self.P_train[train_data_index, :]  # submatrix of performances for points in same cluster
                graph_i_score = np.nanmean(train_data_performance, axis=0)  # shape: (# models,)
                ISAC_score[i, :] = np.nan_to_num(graph_i_score, nan=np.nanmean(graph_i_score).item())
                # NOTE: there may be nans in ISAC_score (when there are empty columns in train_data_performance).
                #       thus, replace nans in ISAC_score (if exist) with the average of graph_i_score.

        return ISAC_score

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(M_train, P_train, P_train_imputed)
        P_hat = self.predict(M_test)
        self.eval_P(P_test, P_hat)
        return P_hat
