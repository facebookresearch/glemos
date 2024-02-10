# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import EarlyStopping, logger, hit_rate_at_k, as_torch_tensor
from .model_selection_method_base import ModelSelectionMethodBase


class SupervisedSurrogate(nn.Module, ModelSelectionMethodBase):
    def __init__(self, hid_dim, num_meta_feats, num_models, device,
                 batch_size=1000, epochs=300, patience=20, val_ratio=0.2,
                 use_imputed=False, name="S2"):
        super().__init__()
        self.name = name
        self.hid_dim = hid_dim
        self.device = device
        self.num_meta_feats = num_meta_feats
        self.num_models = num_models
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_ratio = val_ratio
        self.use_imputed = use_imputed
        self.mlp_M_to_P = nn.Sequential(
            nn.Linear(num_meta_feats, hid_dim * 4),
            nn.ReLU(),
            nn.Linear(hid_dim * 4, num_models),
        ).to(device)

    def forward(self, M):
        return self.mlp_M_to_P(M)

    @staticmethod
    def compute_loss(P_hat, P, P_non_nan_mask):
        assert P_hat.shape == P.shape == P_non_nan_mask.shape, (P_hat.shape, P.shape, P_non_nan_mask.shape)
        mse_loss = nn.MSELoss()
        return mse_loss(P_hat[P_non_nan_mask], P[P_non_nan_mask])

    def fit(self, P_train, P_train_imputed, M_train):
        if self.use_imputed:
            P_train = P_train_imputed

        P_train_non_nan_mask = ~np.isnan(P_train)
        P_train_non_nan_mask = torch.from_numpy(P_train_non_nan_mask).to(self.device)
        M_train = torch.from_numpy(M_train).float().to(self.device)
        P_train = torch.from_numpy(P_train).float().to(self.device)

        P_train, P_val, P_train_non_nan_mask, P_val_non_nan_mask, M_train, M_val = \
            train_test_split(P_train, P_train_non_nan_mask, M_train, test_size=self.val_ratio, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters())
        stopper = EarlyStopping(patience=self.patience, minimizing_objective=False, logging=False)

        self.train()
        epoch_tqdm = tqdm(range(self.epochs))
        num_graphs = M_train.shape[0]
        for epoch in epoch_tqdm:
            graph_indices = torch.randperm(num_graphs)

            total_loss = 0
            for batch_i in range(0, num_graphs, self.batch_size):
                optimizer.zero_grad()

                batch_indices = graph_indices[batch_i:batch_i + self.batch_size]
                batch_P = P_train[batch_indices]
                batch_P_non_nan_mask = P_train_non_nan_mask[batch_indices]
                batch_M = M_train[batch_indices]

                batch_P_hat = self.forward(batch_M)
                batch_loss = self.compute_loss(batch_P_hat, batch_P, batch_P_non_nan_mask)
                total_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()

            epoch_tqdm.set_description(f"[Epoch-{epoch}] train loss={total_loss:.6f}")
            if stopper.step(self.evaluate(P_val, P_val_non_nan_mask, M_val), self):
                logger.info(f"[Epoch-{epoch}] Early stop!")
                break

        if stopper.early_stop:
            stopper.load_checkpoint(self)

    def evaluate(self, P_val, P_val_non_nan_mask, M_val):
        assert P_val.shape[0] == P_val_non_nan_mask.shape[0] == M_val.shape[0], \
            (P_val.shape[0], P_val_non_nan_mask.shape[0], M_val.shape[0])
        self.eval()

        P_hat_val = self.predict(M_val)
        P_val = P_val.cpu().detach().numpy()
        P_val_non_nan_mask = P_val_non_nan_mask.cpu().detach().numpy()
        val_scores = []
        for y_true, y_true_non_nan_mask, y_pred in zip(P_val, P_val_non_nan_mask, P_hat_val):
            val_scores.append(hit_rate_at_k(y_true[y_true_non_nan_mask], np.array(y_pred).flatten()[y_true_non_nan_mask],
                                            k=min(30, sum(y_true_non_nan_mask))))

        return np.mean(val_scores)

    def predict(self, M_test):
        self.eval()
        with torch.no_grad():
            M_test = as_torch_tensor(M_test).to(self.device)
            P_test = self.forward(M_test)
            return P_test.cpu().detach().numpy()

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(P_train, P_train_imputed, M_train)
        P_hat = self.predict(M_test)
        self.eval_P(P_test, P_hat)
        return P_hat
