# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from utils.log_utils import logger
from ..model_selection_method_base import ModelSelectionMethodBase


def sigmoid(x, a=1):
    # return 1 / (1 + np.exp(-1 * a * x))
    return 1 / (1 + np.exp(np.clip(-1 * a * x, a_min=None, a_max=709)))  # to handle "RuntimeWarning: overflow encountered in exp" that sometimes occurs


def sigmoid_derivate(x, a=1):
    return sigmoid(x, a) * (1 - sigmoid(x, a))


class MetaOD(ModelSelectionMethodBase):
    def __init__(self,
                 n_factors=40,
                 learning='sgd',
                 verbose=False,
                 name="MetaOD"):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        train_performance matrix which is ~ user x item (user = dataset, item = method)
        
        Params
        ======
        train_performance : (ndarray)
            User x Item matrix with corresponding train_performance
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        learning : (str)
            Method of optimization. Options include 
            'sgd' or 'als'.
        
        item_fact_reg : (float)
            Regularization term for item latent factors
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            
        item_bias_reg : (float)
            Regularization term for item biases
        
        user_bias_reg : (float)
            Regularization term for user biases
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        super().__init__(name)
        self.n_factors = n_factors
        self.learning = learning
        self._v = verbose

        self.ratings = None
        self.valid_ratings = None
        self.valid_ratings_non_nan_mask = None
        self.n_users = None
        self.n_items = None
        self.n_samples, self.n_models = None, None  # used if self.learning == 'sgd'

        self.train_loss_ = [0]
        self.valid_loss_ = [0]
        self.learning_rates_ = []
        self.scalar_ = None
        self.pca_ = None

    def __str__(self):
        return f"{self.name}(n_factors={self.n_factors}, learning={self.learning})"

    def reset_params(self):
        self.ratings = None
        self.valid_ratings = None
        self.valid_ratings_non_nan_mask = None
        self.n_users = None
        self.n_items = None
        self.n_samples, self.n_models = None, None  # used if self.learning == 'sgd'

        self.train_loss_ = [0]
        self.valid_loss_ = [0]
        self.learning_rates_ = []
        self.scalar_ = None
        self.pca_ = None

    def fit(self, P_train, M_train, val_ratio=0.2):
        assert 0 < val_ratio < 1, val_ratio
        P_train, M_train = np.array(P_train), np.array(M_train)
        P_train, P_val, M_train, M_val = \
            train_test_split(P_train, M_train, test_size=val_ratio, shuffle=True, random_state=1)
        self.train(P_train, P_val, M_train, M_val)

    def train(self, train_performance, valid_performance, meta_features, valid_meta,
              n_iter=10, learning_rate=0.1, n_estimators=100, max_depth=10, max_rate=1.05,
              min_rate=0.1, discount=0.95, n_steps=10):
        """ Train model for n_iter iterations from scratch."""

        self.reset_params()
        self.ratings = train_performance
        self.valid_ratings = valid_performance
        self.valid_ratings_non_nan_mask = ~np.isnan(valid_performance)
        self.n_users, self.n_items = train_performance.shape
        if self.learning == 'sgd':
            self.n_samples, self.n_models = self.ratings.shape[0], self.ratings.shape[1]

        self.pca_ = PCA(n_components=self.n_factors)
        self.pca_.fit(meta_features)

        meta_features_pca = self.pca_.transform(meta_features)
        meta_valid_pca = self.pca_.transform(valid_meta)

        self.scalar_ = StandardScaler()
        self.scalar_.fit(meta_features_pca)

        meta_features_scaled = self.scalar_.transform(meta_features_pca)
        meta_valid_scaled = self.scalar_.transform(meta_valid_pca)

        self.user_vecs = meta_features_scaled

        self.item_vecs = np.random.normal(scale=1. / self.n_factors,
                                          size=(self.n_items, self.n_factors))

        step_size = (max_rate - min_rate) / (n_steps - 1)
        lr_list = list(np.arange(min_rate, max_rate, step_size))
        lr_list.append(max_rate)
        lr_list_reverse = deepcopy(lr_list)
        lr_list_reverse.reverse()

        learning_rate_full = []
        for w in range(n_iter):
            learning_rate_full.extend(lr_list)
            learning_rate_full.extend(lr_list_reverse)

        self.learning_rate_ = min_rate
        self.learning_rates_.append(self.learning_rate_)

        ctr = 1
        np_ctr = 1
        while ctr <= n_iter:

            self.learning_rate_ = learning_rate_full[ctr - 1]
            self.learning_rates_.append(self.learning_rate_)

            self.regr_multirf = MultiOutputRegressor(RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, n_jobs=4))

            # make sure it is non zero
            self.user_vecs[np.isnan(self.user_vecs)] = 0

            self.regr_multirf.fit(meta_features_scaled, self.user_vecs)

            meta_valid_scaled_new = self.regr_multirf.predict(meta_valid_scaled)

            ndcg_s = []
            for w in range(self.valid_ratings.shape[0]):
                y_true = self.valid_ratings[w, :]
                y_pred = np.dot(meta_valid_scaled_new[w, :], self.item_vecs.T)
                # CHANGE FOR SPARSE OPTIMIZATION: include observed entries only for validation
                non_nan_mask = self.valid_ratings_non_nan_mask[w, :]
                if sum(non_nan_mask) <= 1:  # not enough observed entries to compute ndcg
                    continue
                ndcg_s.append(ndcg_score([y_true[non_nan_mask]], [y_pred[non_nan_mask]], k=self.n_items))

            # print('ALORS Fixed iteration', ctr, ndcg_score(self.train_performance, np.dot(self.user_vecs, self.item_vecs.T)))
            # print('ALORS Rank Fixed iteration', ctr, 'valid', np.mean(ndcg_s))
            self.valid_loss_.append(np.mean(ndcg_s))

            print('MetaOD', ctr, 'valid loss', self.valid_loss_[-1],
                  'learning rate', self.learning_rates_[-1])

            # improvement is smaller than 1 perc
            if ((self.valid_loss_[-1] - self.valid_loss_[-2]) /
                self.valid_loss_[-2]) <= 0.001:
                # print(((self.valid_loss_[-1] - self.valid_loss_[-2])/self.valid_loss_[-2]))
                np_ctr += 1
            else:
                np_ctr = 1
            if np_ctr > 5:
                break

            # update learning rates
            # self.learning_rate_ = self.learning_rate_ + 0.05
            # self.learning_rates_.append(self.learning_rate_)
            # if ctr % 2:
            #     if ctr <=50:
            #         self.learning_rate_ = min_rate * np.power(discount,ctr)
            #     else:
            #         self.learning_rate_ = min_rate * np.power(discount,50)

            # else:
            #     if ctr <=50:
            #         self.learning_rate_ = max_rate * np.power(discount,ctr)
            #     else:
            #         self.learning_rate_ = max_rate * np.power(discount,50)

            # self.learning_rates_.append(self.learning_rate_)

            train_indices = list(range(self.n_samples))
            np.random.shuffle(train_indices)
            # print(train_indices)

            for idx, h in enumerate(train_indices):  # h refers to dataset
                if idx % 50 == 0:
                    logger.info(f"[epoch {ctr} / {n_iter}] processing train index-{idx} / {len(train_indices)}")

                uh = self.user_vecs[h, :].reshape(1, -1)
                # print(uh.shape)
                grads = []

                for i in range(self.n_models):  # i refers to method
                    # CHANGE FOR SPARSE OPTIMIZATION: skip if self.ratings[h, i] is not observed
                    if np.isnan(self.ratings[h, i]):
                        continue

                    # outler loop
                    vi = self.item_vecs[i, :].reshape(-1, 1)
                    phis = []
                    rights = []
                    rights_v = []
                    # remove i from js
                    js = list(range(self.n_models))
                    js.remove(i)

                    # CHANGE FOR SPARSE OPTIMIZATION: remove models without observed performance
                    nan_indices = np.isnan(self.ratings[h, :]).nonzero()[0].tolist()
                    for nan_index in nan_indices:
                        js.remove(nan_index)

                    for j in js:
                        vj = self.item_vecs[j, :].reshape(-1, 1)
                        # temp_vt = np.exp(np.matmul(uh, (vj-vi)))
                        # temp_vt = np.ndarray.item(temp_vt)
                        temp_vt = sigmoid(np.ndarray.item(np.matmul(uh, (vj - vi))), a=1)
                        temp_vt_derivative = sigmoid_derivate(np.ndarray.item(np.matmul(uh, (vj - vi))), a=1)
                        # print(uh.re, (self.item_vecs[j,:]-self.item_vecs[i,:]).T.shape)
                        # print((self.item_vecs[j,:]-self.item_vecs[i,:]).reshape(-1, 1).shape)
                        # print(temp_vt.shape)
                        # assert (len(temp_vt)==1)
                        phis.append(temp_vt)
                        rights.append(temp_vt_derivative * (vj - vi))
                        rights_v.append(temp_vt_derivative * uh)
                    phi = np.sum(phis) + 1.5
                    # CHANGE FOR SPARSE OPTIMIZATION: minus the number of nans
                    rights = np.asarray(rights).reshape(self.n_models - 1 - len(nan_indices), self.n_factors)
                    rights_v = np.asarray(rights_v).reshape(self.n_models - 1 - len(nan_indices), self.n_factors)

                    # print(rights.shape, rights_v.shape)

                    right = np.sum(np.asarray(rights), axis=0)
                    right_v = np.sum(np.asarray(rights_v), axis=0)
                    # print(right, right_v)

                    # print(np.asarray(rights).shape, np.asarray(right).shape)
                    grad = (10 ** (self.ratings[h, i]) - 1) / (phi * (np.log(phi)) ** 2) * right
                    grad_v = (10 ** (self.ratings[h, i]) - 1) / (phi * (np.log(phi)) ** 2) * right_v

                    self.item_vecs[i, :] += self.learning_rate_ * grad_v

                    # print(h, i, grad.shape)
                    grads.append(grad)

                grads_uh = np.asarray(grads)
                grad_uh = np.sum(grads_uh, axis=0)

                self.user_vecs[h, :] -= self.learning_rate_ * grad_uh
                # print(self.learning_rate_)

            ctr += 1

        # self.regr_multirf = MultiOutputRegressor(RandomForestRegressor(
        #     n_estimators=n_estimators, max_depth=max_depth, n_jobs=4))

        # self.regr_multirf = MultiOutputRegressor(Lasso()))
        # self.regr_multirf = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=n_estimators))

        # self.regr_multirf.fit(meta_features, self.user_vecs)

        return self

    # def predict(self, u, i):
    #     """ Single user and item prediction."""
    #     # prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
    #     prediction = self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    #     # prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    #     return prediction

    # def predict_all(self):
    #     """ Predict train_performance for every user and item."""
    #     predictions = np.zeros((self.user_vecs.shape[0],
    #                             self.item_vecs.shape[0]))
    #     for u in range(self.user_vecs.shape[0]):
    #         for i in range(self.item_vecs.shape[0]):
    #             predictions[u, i] = self.predict(u, i)

    #     return predictions

    def predict(self, test_meta):
        test_meta = check_array(test_meta)
        # assert (test_meta.shape[1]==200)

        test_meta_scaled = self.pca_.transform(test_meta)
        # print('B', test_meta_scaled.shape)

        test_meta_scaled = self.scalar_.transform(test_meta_scaled)
        test_meta_scaled = self.regr_multirf.predict(test_meta_scaled)

        # predicted_scores = np.dot(test_k, self.item_vecs.T) + self.item_bias
        predicted_scores = np.dot(test_meta_scaled, self.item_vecs.T)
        # print(predicted_scores.shape)
        assert (predicted_scores.shape[0] == test_meta.shape[0])
        assert (predicted_scores.shape[1] == self.n_models)

        return predicted_scores

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(P_train, M_train)
        P_hat = self.predict(M_test)
        print("P_hat in metaod:", P_hat.shape)
        self.eval_P(P_test, P_hat)
        return P_hat
