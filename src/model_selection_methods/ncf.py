# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import EarlyStopping, logger, hit_rate_at_k, as_torch_tensor
from .model_selection_method_base import ModelSelectionMethodBase


class NCF(nn.Module, ModelSelectionMethodBase):
    def __init__(self, hid_dim, model_feats_dim, num_meta_feats, num_models, device,
                 batch_size=1000, epochs=200, patience=30, scoring='ncf', use_imputed=False,
                 name="NCF"):
        super().__init__()
        self.name = name
        self.hid_dim = hid_dim
        self.model_feats_dim = model_feats_dim
        self.num_meta_feats = num_meta_feats
        self.num_models = num_models
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.scaler = MinMaxScaler()

        self.model_feats = nn.Embedding(num_models, model_feats_dim).to(device)
        self.model_transform = nn.Sequential(nn.Linear(model_feats_dim, hid_dim * 4), nn.ReLU(), nn.Dropout(p=0.2),
                                             nn.Linear(hid_dim * 4, hid_dim)).to(device)
        self.meat_feat_transform = nn.Sequential(nn.Linear(num_meta_feats, hid_dim * 4), nn.ReLU(), nn.Dropout(p=0.2),
                                                 nn.Linear(hid_dim * 4, hid_dim)).to(device)

        self.scoring = scoring
        if self.scoring == 'ncf':
            self.ncf_transform = NCFPredictor(hid_dim, hid_dim, device)
        else:
            raise ValueError(f"Invalid scoring: {self.scoring}")
        self.use_imputed = use_imputed

    def forward(self, M):
        graph_emb = self.meat_feat_transform(M)  # shape=(# graphs, hid_dim)
        model_emb = self.model_transform(self.model_feats.weight)  # shape=(# models, hid_dim)

        if self.scoring == 'ncf':
            emb_dict = {'graph': graph_emb, 'model': model_emb}
            scores = self.ncf_transform(emb_dict)
        else:
            raise ValueError(f"scoring: {self.scoring}")
        assert scores.shape[0] == M.shape[0] and scores.shape[1] == self.num_models  # scores shape=(# graphs, # models)

        return scores

    @staticmethod
    def compute_loss(P_hat, P, P_non_nan_mask):
        assert P_hat.shape == P.shape == P_non_nan_mask.shape, (P_hat.shape, P.shape, P_non_nan_mask.shape)
        mse_loss = nn.MSELoss()
        return mse_loss(P_hat[P_non_nan_mask], P[P_non_nan_mask])

    def fit(self, P_train, P_train_imputed, M_train):
        assert P_train.shape[0] == M_train.shape[0], (P_train.shape[0], M_train.shape[0])
        if self.use_imputed:
            P_train = P_train_imputed

        P_train_non_nan_mask = ~np.isnan(P_train)
        P_train_non_nan_mask = torch.from_numpy(P_train_non_nan_mask).to(self.device)
        M_train = torch.from_numpy(M_train).float().to(self.device)
        P_train = torch.from_numpy(P_train).float().to(self.device)

        P_train, P_val, P_train_non_nan_mask, P_val_non_nan_mask, M_train, M_val = \
            train_test_split(P_train, P_train_non_nan_mask, M_train, test_size=0.2, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
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
            if epoch > 30 and stopper.step(self.evaluate(P_val, P_val_non_nan_mask, M_val), self):
                logger.info(f"[Epoch-{epoch}] Early stop!")
                break

        if stopper.early_stop:
            stopper.load_checkpoint(self)

    def evaluate(self, P_val, P_val_non_nan_mask, M_val):
        assert P_val.shape[0] == P_val_non_nan_mask.shape[0] == M_val.shape[0], (P_val.shape[0], P_val_non_nan_mask.shape[0], M_val.shape[0])
        self.eval()

        P_hat_val = self.predict(M_val)
        P_val = P_val.cpu().detach().numpy()
        P_val_non_nan_mask = P_val_non_nan_mask.cpu().detach().numpy()
        val_scores = []
        for y_true, y_true_non_nan_mask, y_pred in zip(P_val, P_val_non_nan_mask, P_hat_val):
            # val_scores.append(roc_auc_score(y_true[y_true_non_nan_mask], np.array(y_pred).flatten()[y_true_non_nan_mask]))
            val_scores.append(hit_rate_at_k(y_true[y_true_non_nan_mask], np.array(y_pred).flatten()[y_true_non_nan_mask],
                                            k=min(30, sum(y_true_non_nan_mask))))

        return np.mean(val_scores)

    def predict(self, M_test):
        self.eval()
        with torch.no_grad():
            # M_test = self.scaler.transform(M_test)
            M_test = as_torch_tensor(M_test).to(self.device)
            P_test = self.forward(M_test)
            return P_test.cpu().detach().numpy()

    def fit_predict(self, M_train, M_test, P_train, P_train_imputed, P_train_full, P_test, D_train=None):
        self.fit(P_train, P_train_imputed, M_train)
        P_hat = self.predict(M_test)
        self.eval_P(P_test, P_hat)
        return P_hat


class NCFPredictor(nn.Module):
    def __init__(self, in_feats, hid_feats, device):
        super().__init__()

        self.concat_W = nn.Sequential(
            nn.Linear(in_feats * 2, hid_feats),
            nn.ReLU(),
            nn.Linear(hid_feats, hid_feats)
        ).to(device)
        self.last_W = nn.Linear(hid_feats * 2, 1)
        self.device = device

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']

        concat_h = self.concat_W(torch.cat([h_u, h_v], dim=1))
        elem_h = h_u * h_v  # elementwise product

        concat_h2 = torch.cat([concat_h, elem_h], dim=1)
        score = self.last_W(concat_h2)

        return {'score': score}

    # noinspection PyMethodMayBeStatic
    def forward(self, emb_dict):
        graph_emb, model_emb = emb_dict['graph'], emb_dict['model']
        num_graphs, num_models = len(graph_emb), len(model_emb)
        graph_idx, model_idx = torch.arange(num_graphs), torch.arange(num_models)

        import dgl
        g2m_graph = dgl.heterograph(data_dict={
            ('graph', 'g2m', 'model'): (graph_idx.repeat_interleave(num_models), model_idx.repeat(num_graphs)),
        }, num_nodes_dict={'graph': len(graph_emb), 'model': len(model_emb)}).to(self.device)
        g2m_graph.ndata['h'] = emb_dict

        g2m_graph.apply_edges(self.apply_edges)
        return g2m_graph.edata['score'].reshape(num_graphs, num_models)
