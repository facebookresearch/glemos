# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy import dot
from scipy import linalg
from sklearn.decomposition import non_negative_factorization
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from .model_selection_method_base import ModelSelectionMethodBase


class ALORS(ModelSelectionMethodBase):
    def __init__(self, k_dim, name='ALORS', use_imputed=False):
        super().__init__(name)
        self.k_dim = k_dim
        self.use_imputed = use_imputed
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        self.V = None  # latent factors for models
        self.model_est_M_to_U = None

    def __str__(self):
        return f"{self.name}(k_dim={self.k_dim}, use_imputed={self.use_imputed})"

    def fit(self, P_train, P_train_imputed, M_train):
        _P_train_ = P_train_imputed if self.use_imputed else P_train

        U_nmf, V_nmf = nmf(_P_train_.copy(), latent_features=min(self.k_dim, P_train_imputed.shape[1]))
        # U_nmf, V_nmf, n_iter = non_negative_factorization(_P_train_.copy(),
        #                                                   n_components=min(self.k_dim, _P_train_.shape[1]),
        #                                                   init='nndsvda', W=None, H=None,
        #                                                   beta_loss='kullback-leibler', solver='mu',
        #                                                   random_state=1, max_iter=200)

        U = U_nmf.copy()
        self.V = V_nmf.copy()

        self.model_est_M_to_U = MLPRegressor(random_state=1, hidden_layer_sizes=(self.k_dim * 4, self.k_dim * 4))
        self.model_est_M_to_U.fit(M_train.copy(), U)

    def predict(self, M_test):
        U_test = self.model_est_M_to_U.predict(M_test.copy())
        return np.matmul(U_test, self.V)

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(P_train, P_train_imputed, M_train)
        P_hat = self.predict(M_test)
        self.eval_P(P_test, P_hat)
        return P_hat


def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    Ref: https://stackoverflow.com/questions/22767695/python-non-negative-matrix-factorization-that-handles-both-zeros-and-missing-dat
    """
    eps = 1e-5
    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    X = np.array(X)  # nans in X denote missing values

    # mask
    # mask = np.sign(X)
    mask = ~np.isnan(X)
    X[np.isnan(X)] = 0

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    # Note: While this code is suggested as a way to fix a bug in the initialization of Y, it didn't perform well.
    # Y = np.zeros((latent_features, columns))
    # bool_mask = mask.astype(bool)
    # for i in range(columns):
    #     Y[:, i] = linalg.lstsq(A[bool_mask[:, i], :], X[bool_mask[:, i], i])[0]

    # Y = np.random.rand(latent_features, columns)
    Y = linalg.lstsq(A, X)[0]  # yields a bias toward estimating missing values as zeros in the initial A and Y
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)

        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            # print('Iteration {}:'.format(i))
            # print('fit residual', np.round(fit_residual, 4))
            # print('total residual', np.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y
