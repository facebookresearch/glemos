# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .model_selection_method_base import ModelSelectionMethodBase


class ARGOSMART(ModelSelectionMethodBase):
    """
    ARGOSMART (AS), aka 1NN baseline, finds the meta-train graph
    that is closest to the test graph in terms of the meta-graph features,
    then selects the best performing model for the 1NN meta-train graph
    """

    def __init__(self, num_models, name='AS', use_imputed=False):
        super().__init__(name)
        self.P_train = None
        self.P_train_imputed = None
        self.M_train = None
        self.num_models = num_models
        self.use_imputed = use_imputed

    def __str__(self):
        return f"{self.name}(use_imputed={self.use_imputed})"

    def fit(self, M_train, P_train, P_train_imputed):
        self.M_train = M_train
        self.P_train = P_train
        self.P_train_imputed = P_train_imputed

    def predict(self, M_test):
        assert self.M_train is not None and self.P_train is not None

        P_hat_NN = np.zeros((M_test.shape[0], self.num_models))
        for i in range(0, M_test.shape[0]):
            cos_sim = cosine_similarity(self.M_train, M_test[i, :].reshape(1, -1)).flatten()
            one_nn_graph_idx = np.argmax(cos_sim)

            if self.use_imputed:
                P_hat_NN[i, :] = self.P_train_imputed[one_nn_graph_idx]
            else:
                onn_nn_graph_scores = self.P_train[one_nn_graph_idx]
                P_hat_NN[i, :] = np.nan_to_num(onn_nn_graph_scores, nan=np.nanmean(onn_nn_graph_scores).item())

        return P_hat_NN

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(M_train, P_train, P_train_imputed)
        P_hat = self.predict(M_test)
        self.eval_P(P_test, P_hat)
        return P_hat
