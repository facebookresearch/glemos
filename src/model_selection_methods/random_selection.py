# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from contextlib import contextmanager

import numpy as np

from .model_selection_method_base import ModelSelectionMethodBase


class RandomSelection(ModelSelectionMethodBase):
    def __init__(self, name='RandomSelection'):
        super().__init__(name)

    def __str__(self):
        return f"{self.name}()"

    def fit(self, P_train):
        pass

    # noinspection PyMethodMayBeStatic
    def predict(self, P_test):
        num_graphs, num_models = P_test.shape[0], P_test.shape[1]

        with temp_np_seed(127):
            P_pred = np.zeros_like(P_test)
            for i in range(0, num_graphs):
                P_pred[i, :] = np.random.permutation(num_models)

            return P_pred

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(P_train)
        P_hat = self.predict(P_test)
        self.eval_P(P_test, P_hat)
        return P_hat


@contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
