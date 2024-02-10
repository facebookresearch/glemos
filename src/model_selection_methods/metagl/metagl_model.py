# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import time
import traceback
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl import DGLHeteroGraph
from numpy import dot
from scipy import linalg
from sklearn.decomposition import KernelPCA, PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils import EarlyStopping, eval_metrics, logger, as_torch_tensor, create_eval_dict
from utils import eval_metrics_single_graph
from .hgt import HGT
from .loss import top_one_loss


class MetaGL(nn.Module):
    def __init__(self, num_models, metafeats_dim, device, hid_dim=32, knn_k=30,
                 use_g_m_network=True, use_single_type_hetero_graph=False,
                 batch_size=80, epochs=500, patience=50, loss='top_one',
                 val_ratio=0.3, eval_every=1, eval_after_epoch=0,
                 optimize_model_embeddings=False):
        super().__init__()
        self.name = "MetaGL"
        self.eval_dict = None
        self.hid_dim = hid_dim
        self.input_dim = min(hid_dim * 2, metafeats_dim, num_models)
        self.num_models = num_models
        self.metafeats_dim = metafeats_dim
        self.device = device
        self.knn_k = knn_k
        self.optimize_model_embeddings = optimize_model_embeddings
        self.factorization = None
        self.val_ratio = val_ratio
        self.eval_every = eval_every
        self.eval_after_epoch = eval_after_epoch
        self.graph_conv = None
        self.use_g_m_network = use_g_m_network
        self.use_single_type_hetero_graph = use_single_type_hetero_graph
        self.M_to_P_graph_factors_regressor = self.latent_factor_regressor()
        self.P_train_model_factors_init = None
        self.P_train_model_factors = None  # used to obtain node embeddings. optimized (if optimize_model_embeddings=True) after initialized with P_train_model_factors_init
        self.graph_emb_net = nn.Linear(metafeats_dim * 2 + self.input_dim, self.input_dim).to(device)
        self.score_net = DotProductPredictor()
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.predict_times = []
        self.loss = loss
        if loss == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss == 'top_one':
            self.loss_fn = top_one_loss
        else:
            raise ValueError(f"Invalid loss: {loss}")

    @staticmethod
    def create_knn_graph(M, P_graph_factors, P_model_factors, knn_k):
        assert len(M) == len(P_graph_factors), (len(M), len(P_graph_factors))
        M, P_graph_factors, P_model_factors = [x.cpu().clone().detach() for x in [M, P_graph_factors, P_model_factors]]

        M_g2g_edges = knn_edges_x_to_y(M, M, knn_k)  # (graph, graph) edges induced by meta-features
        P_g2g_edges = knn_edges_x_to_y(P_graph_factors, P_graph_factors, knn_k)
        P_g2m_edges = knn_edges_x_to_y(P_graph_factors, P_model_factors, knn_k)
        P_m2g_edges = knn_edges_x_to_y(P_model_factors, P_graph_factors, knn_k)
        P_m2m_edges = knn_edges_x_to_y(P_model_factors, P_model_factors, knn_k)

        G = dgl.heterograph(data_dict={
            ('graph', 'M_g2g', 'graph'): M_g2g_edges,
            ('graph', 'P_g2g', 'graph'): P_g2g_edges,
            ('model', 'P_m2m', 'model'): P_m2m_edges,
            ('graph', 'P_g2m', 'model'): P_g2m_edges,
            ('model', 'P_m2g', 'graph'): P_m2g_edges,
        }, num_nodes_dict={'graph': len(P_graph_factors), 'model': len(P_model_factors)})
        return G

    def set_node_and_edge_dict(self, G):
        """Used in HGT"""
        G.node_dict = {}
        G.edge_dict = {}
        for ntype in G.ntypes:
            G.node_dict[ntype] = len(G.node_dict)
        for etype in G.etypes:
            G.edge_dict[etype] = len(G.edge_dict)
            G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device=self.device) * \
                                        G.edge_dict[etype]

    def create_train_test_graphs(self, M_train, P_hat_train_graph_factors, P_train_model_factors,
                                 M_test, P_hat_test_graph_factors):
        G_train = self.create_knn_graph(M_train, P_hat_train_graph_factors, P_train_model_factors, self.knn_k).to(device=self.device)
        self.set_node_and_edge_dict(G_train)  # for HGT
        test_graph_offset = len(M_train)

        # M_g2g_edges from train graph to test graph
        M_g2g_test_graph_idx0, M_g2g_train_graph_idx = knn_edges_x_to_y(M_test, M_train, self.knn_k)
        M_g2g_edges = (M_g2g_train_graph_idx.to(self.device), M_g2g_test_graph_idx0.to(self.device) + test_graph_offset)

        # P_g2g_edges from train graph to test graph
        P_g2g_test_graph_idx0, P_g2g_train_graph_idx = knn_edges_x_to_y(P_hat_test_graph_factors, P_hat_train_graph_factors, self.knn_k)
        P_g2g_edges = (P_g2g_train_graph_idx.to(self.device), P_g2g_test_graph_idx0.to(self.device) + test_graph_offset)

        # P_g2m_edges from test graph to model
        P_g2m_test_graph_idx0, P_g2m_model_idx = knn_edges_x_to_y(P_hat_test_graph_factors, P_train_model_factors, self.knn_k)
        P_g2m_edges = (P_g2m_test_graph_idx0.to(self.device) + test_graph_offset, P_g2m_model_idx.to(self.device))

        # P_m2g_edges from model to test graph
        P_m2g_model_idx, P_m2g_test_graph_idx0 = knn_edges_x_to_y(P_train_model_factors, P_hat_test_graph_factors, self.knn_k)
        P_m2g_edges = (P_m2g_model_idx.to(self.device), P_m2g_test_graph_idx0.to(self.device) + test_graph_offset)

        G_traintest = copy.deepcopy(G_train)
        G_traintest.add_nodes(len(M_test), ntype='graph')  # test graph-nodes
        G_traintest.add_edges(*M_g2g_edges, etype='M_g2g')
        G_traintest.add_edges(*P_g2g_edges, etype='P_g2g')
        G_traintest.add_edges(*P_g2m_edges, etype='P_g2m')
        G_traintest.add_edges(*P_m2g_edges, etype='P_m2g')
        self.set_node_and_edge_dict(G_traintest)  # for HGT
        G_traintest.nodes['graph'].data['eval'] = torch.Tensor([0] * len(M_train) + [1] * len(M_test)).bool().to(device=self.device)
        assert G_train.node_dict == G_traintest.node_dict and G_train.edge_dict == G_traintest.edge_dict

        return G_train, G_traintest

    def get_input_node_embeddings(self, G, M, P_graph_factors, P_model_factors,
                                  hetero_graph_ntypes):  # e.g., hetero_graph_ntypes=["graph", "model"]
        # 'model' node embeddings
        model_node_emb = P_model_factors.to(self.device)

        # 'graph' node embeddings
        assert not M.requires_grad and not P_graph_factors.requires_grad
        graph_node_emb = self.graph_emb_net(torch.cat([M, P_graph_factors], dim=1)).to(self.device)

        ntype2nid = {ntype: nid for nid, ntype in enumerate(hetero_graph_ntypes)}
        assert isinstance(G, DGLHeteroGraph), type(G)

        if len(G.ntypes) == 1:  # single type, heterogeneous graph
            if G.ndata[f"ORG{dgl.NTYPE}"][0].item() == ntype2nid['graph']:
                return {'node_type': torch.cat([graph_node_emb, model_node_emb], dim=0)}
            else:
                return {'node_type': torch.cat([model_node_emb, graph_node_emb], dim=0)}
        else:
            assert G.number_of_nodes('graph') == len(M) == len(P_graph_factors)
            assert G.number_of_nodes('model') == len(P_model_factors)
            assert G.ntypes == hetero_graph_ntypes, (G.ntypes, hetero_graph_ntypes)

            node_embedding_dict = {'model': model_node_emb, 'graph': graph_node_emb}
            assert node_embedding_dict['model'].shape[1] == node_embedding_dict['graph'].shape[1], \
                (node_embedding_dict['model'].shape, node_embedding_dict['graph'].shape)

            return node_embedding_dict

    def set_factorization(self, P):
        if np.isnan(P).any():
            self.factorization = 'sparse_nmf'
        else:
            self.factorization = 'pca'

    @classmethod
    def latent_factor_regressor(cls):
        return RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                     max_features="auto", max_depth=None, n_jobs=-1,
                                     warm_start=False, ccp_alpha=0, random_state=1)

    def fit_graph_and_model_factors(self, P_train, M_train, M_test):
        """Model factors"""
        P_train_np = P_train.cpu().detach().numpy()
        if self.factorization == 'pca':
            pca_graph = PCA(n_components=self.input_dim, random_state=100)
            P_train_graph_factors = pca_graph.fit_transform(P_train_np)
            pca_model = PCA(n_components=self.input_dim, random_state=100)
            P_train_model_factors_init = pca_model.fit_transform(P_train_np.T)
            P_train_model_factors_init = as_torch_tensor(P_train_model_factors_init).to(self.device)
            self.P_train_model_factors_init = P_train_model_factors_init
        elif self.factorization == 'sparse_nmf':
            P_train_graph_factors, P_train_model_factors_T = sparse_nmf(P_train_np, latent_features=self.input_dim)
            P_train_model_factors_init = as_torch_tensor(P_train_model_factors_T.T).to(self.device)
            self.P_train_model_factors_init = P_train_model_factors_init
        elif 'kernel_pca' in self.factorization:
            kernel = self.factorization.split('-')[1].strip()
            pca_graph = KernelPCA(n_components=self.input_dim, kernel=kernel, n_jobs=-1, random_state=1)
            P_train_graph_factors = pca_graph.fit_transform(P_train_np)
            pca_model = KernelPCA(n_components=self.input_dim, kernel=kernel, n_jobs=-1, random_state=1)
            P_train_model_factors_init = pca_model.fit_transform(P_train_np.T)
            P_train_model_factors_init = as_torch_tensor(P_train_model_factors_init).to(self.device)
            self.P_train_model_factors_init = P_train_model_factors_init
        else:
            raise ValueError(f"Invalid: {self.factorization}")

        self.P_train_model_factors = self.P_train_model_factors_init.clone().detach()
        if self.optimize_model_embeddings:
            self.P_train_model_factors.requires_grad_()

        """Graph factors"""
        self.M_to_P_graph_factors_regressor.fit(M_train.cpu().detach().numpy(), P_train_graph_factors)
        P_hat_train_graph_factors = self.predict_P_graph_factors(M_train)
        P_hat_test_graph_factors = self.predict_P_graph_factors(M_test)

        return P_hat_train_graph_factors, P_hat_test_graph_factors, P_train_model_factors_init

    def predict_P_graph_factors(self, M):
        if isinstance(M, torch.Tensor):
            M = M.cpu().detach().numpy()
        P_hat_graph_factors = self.M_to_P_graph_factors_regressor.predict(M)
        return as_torch_tensor(P_hat_graph_factors).to(self.device)

    def fit(self, P_train_input, M_train_input, D_train):
        self.set_factorization(P_train_input)
        P_train_input = (MinMaxScaler().fit_transform(P_train_input.T).T * 10 + 1)
        P_train_input, M_train_input = [as_torch_tensor(X).to(self.device) for X in [P_train_input, M_train_input]]
        M_train_input = torch.cat([M_train_input, torch.log(M_train_input + 1.0)], dim=1)
        P_train, P_val, M_train, M_val = \
            train_test_split(P_train_input, M_train_input, test_size=self.val_ratio, shuffle=True, random_state=1)
        P_trainval = torch.cat([P_train, P_val], dim=0)
        M_trainval = torch.cat([M_train, M_val], dim=0)
        num_train_graphs = P_train.shape[0]

        P_hat_train_graph_factors, P_hat_val_graph_factors, P_train_model_factors_init = \
            self.fit_graph_and_model_factors(P_train, M_train, M_val)
        G_train_val = self.create_train_test_graphs(M_train, P_hat_train_graph_factors, P_train_model_factors_init,
                                                    M_val, P_hat_val_graph_factors)
        hetero_graph_ntypes = G_train_val[0].ntypes

        if self.use_single_type_hetero_graph:
            G_train_val = self.to_single_type_heterogeneous_graph(G_train_val[0]), self.to_single_type_heterogeneous_graph(G_train_val[1])
            for G in G_train_val:
                self.set_node_and_edge_dict(G)  # for HGT

        G_train = G_train_val[0]
        self.graph_conv = HGT(G_train, G_train.node_dict, G_train.edge_dict, n_inp=self.input_dim,
                              n_hid=self.hid_dim, n_out=self.hid_dim, n_layers=2, n_heads=4, use_norm=True)
        self.graph_conv = self.graph_conv.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=7.5e-4, weight_decay=1e-4)
        if self.patience is not None:
            stopper = EarlyStopping(patience=self.patience, minimizing_objective=False, logging=True)
        else:
            stopper = None

        self.train()
        try:
            epoch_tqdm = tqdm(range(self.epochs))
            for epoch in epoch_tqdm:
                graph_indices = torch.randperm(num_train_graphs)

                total_loss = 0
                for batch_i in range(0, num_train_graphs, self.batch_size):
                    optimizer.zero_grad()
                    train_node_emb = self.get_input_node_embeddings(G_train_val[0], M_train, P_hat_train_graph_factors,
                                                                    self.P_train_model_factors, hetero_graph_ntypes)
                    P_hat_train = self.forward(G_train_val[0], train_node_emb, hetero_graph_ntypes)

                    batch_indices = graph_indices[batch_i:batch_i + self.batch_size]
                    batch_P_train = P_train[batch_indices]
                    batch_P_hat_train = P_hat_train[batch_indices]

                    batch_loss = self.loss_fn(batch_P_hat_train, batch_P_train)
                    total_loss += batch_loss.item()
                    batch_loss.backward()
                    optimizer.step()

                if epoch > self.eval_after_epoch and epoch % self.eval_every == 0:
                    desc = f"Epoch-{epoch:04d}: [Train] loss={total_loss:.4f}"

                    """Validation"""
                    P_hat_trainval_graph_factors = torch.cat([P_hat_train_graph_factors, P_hat_val_graph_factors], dim=0)
                    val_ev_dict = self.evaluate(G_train_val[1], M_trainval, P_trainval,
                                                P_hat_trainval_graph_factors, self.P_train_model_factors,
                                                hetero_graph_ntypes)
                    desc += f" | [Val] AUC={val_ev_dict['AUC']:.4f}, MRR={val_ev_dict['MRR']:.4f}, " \
                            f"NDCG@1={val_ev_dict['nDCG@1']:.4f}"
                    logger.info(desc)

                    val_score = np.mean(np.array([val_ev_dict['AUC'], val_ev_dict['MRR']]))
                    if val_ev_dict and stopper and stopper.step(val_score, self):
                        logger.info(f"[Epoch-{epoch}] Early stop!")
                        break
                else:
                    logger.info(f"Epoch-{epoch:04d}: [Train] loss={total_loss:.6f}")

        except KeyboardInterrupt:
            print("\n=== TRAINING INTERRUPTED ===\n")
        except Exception:
            traceback.print_exc()
            raise

        if stopper and stopper.early_stop:
            stopper.load_checkpoint(self)

    @classmethod
    def to_single_type_heterogeneous_graph(cls, G):
        assert not G.is_homogeneous
        hm_G = dgl.to_homogeneous(G)

        single_type_ndata_nid = torch.arange(len(hm_G.ndata[dgl.NID]))
        single_type_ndata_ntype = torch.zeros_like(hm_G.ndata[dgl.NTYPE])
        single_type_edata_eid = torch.arange(len(hm_G.edata[dgl.EID]))
        single_type_edata_etype = torch.zeros_like(hm_G.edata[dgl.ETYPE])

        hm_G.ndata[f"ORG{dgl.NID}"], hm_G.ndata[f"ORG{dgl.NTYPE}"] = hm_G.ndata[dgl.NID], hm_G.ndata[dgl.NTYPE]
        hm_G.edata[f"ORG{dgl.EID}"], hm_G.edata[f"ORG{dgl.ETYPE}"] = hm_G.edata[dgl.EID], hm_G.edata[dgl.ETYPE]

        hm_G.ndata[dgl.NID], hm_G.ndata[dgl.NTYPE] = single_type_ndata_nid, single_type_ndata_ntype
        hm_G.edata[dgl.EID], hm_G.edata[dgl.ETYPE] = single_type_edata_eid, single_type_edata_etype

        hm_ht_G = dgl.to_heterogeneous(hm_G, ntypes=['node_type'], etypes=['edge_type'])
        hm_ht_G.ndata[f"ORG{dgl.NID}"], hm_ht_G.ndata[f"ORG{dgl.NTYPE}"] = hm_G.ndata[f"ORG{dgl.NID}"], hm_G.ndata[f"ORG{dgl.NTYPE}"]

        try:
            hm_ht_G.graph_node_eval_mask = G.nodes['graph'].data['eval']
        except Exception:
            pass

        return hm_ht_G

    def forward(self, G, node_emb, hetero_graph_ntypes):  # e.g., hetero_graph_ntypes=["graph", "model"]
        if self.use_g_m_network:
            ntype2nid = {ntype: nid for nid, ntype in enumerate(hetero_graph_ntypes)}

            assert isinstance(G, DGLHeteroGraph), type(G)
            emb_dict = self.graph_conv(G, node_emb)

            if len(G.ntypes) == 1:  # single type, heterogeneous graph
                emb = emb_dict['node_type']
                graph_node_mask = G.ndata[f"ORG{dgl.NTYPE}"] == ntype2nid['graph']
                model_node_mask = G.ndata[f"ORG{dgl.NTYPE}"] == ntype2nid['model']
                emb_dict = {'graph': emb[graph_node_mask], 'model': emb[model_node_mask]}
        else:
            assert not G.is_homogeneous
            emb_dict = node_emb

        return self.score_net(emb_dict)

    def evaluate(self, G, M, P, P_graph_factors, P_model_factors, hetero_graph_ntypes):
        self.eval()
        with torch.no_grad():
            node_emb = self.get_input_node_embeddings(G, M, P_graph_factors, P_model_factors, hetero_graph_ntypes)
            P_hat = self.forward(G, node_emb, hetero_graph_ntypes)

        assert P.shape == P_hat.shape, (P.shape, P_hat.shape)
        if not G.is_homogeneous:
            eval_graph_mask = G.nodes['graph'].data['eval']
        else:
            eval_graph_mask = G.graph_node_eval_mask
        P, P_hat = P[eval_graph_mask], P_hat[eval_graph_mask]

        if torch.isnan(P).any():
            return self.eval_metric_sparse_p(P, P_hat)
        else:
            return eval_metrics(Y_true=P, Y_pred=P_hat)

    @classmethod
    def eval_metric_sparse_p(cls, P_test, P_hat):
        if isinstance(P_test, torch.Tensor):
            P_test = np.matrix(P_test.cpu().detach().numpy())
        if isinstance(P_hat, torch.Tensor):
            P_hat = np.matrix(P_hat.cpu().detach().numpy())
        assert P_test.shape == P_hat.shape
        P_hat_non_nan_mask = ~np.isnan(P_test)

        eval_dict = create_eval_dict()
        for i in range(0, P_test.shape[0]):
            p_test = P_test[i, :]
            p_hat = P_hat[i, :]
            p_hat_non_nan_mask = P_hat_non_nan_mask[i, :]

            for metric, metric_score in cls.eval_metrics_single_graph(p_test[p_hat_non_nan_mask],
                                                                      p_hat[p_hat_non_nan_mask]).items():
                eval_dict[metric].append(metric_score)

        return {metric: np.mean(eval_dict[metric]) for metric in ['AUC', 'MRR', 'nDCG@1']}

    @classmethod
    def eval_metrics_single_graph(cls, y_true, y_pred):
        assert len(y_true.shape) == 2 and y_true.shape[0] == 1, y_true.shape
        assert y_pred.shape == y_true.shape, y_pred.shape

        idx_best_model = np.argmax(y_true)
        num_models = y_true.shape[1]
        y_true_bin = np.matrix(np.zeros((1, num_models), dtype=int))
        y_true_bin[0, idx_best_model] = 1

        eval_dict = {}
        for k in [1]:
            eval_dict[f'nDCG@{k}'] = ndcg_score(np.asarray(y_true), np.asarray(y_pred), k=k)

        y_true_flatten = np.array(y_true).flatten()
        for k in list(filter(lambda x: x <= len(y_true_flatten), [1])):
            top_k_ind = np.argpartition(y_true_flatten, -k)[-k:]
            num_models = y_true.shape[1]
            y_true_bin = np.matrix(np.zeros((1, num_models), dtype=int))
            y_true_bin[0, top_k_ind] = 1

            eval_dict[f'AUC@{k}'] = roc_auc_score(np.array(y_true_bin).flatten(), np.array(y_pred).flatten())
            eval_dict[f'MAP@{k}'] = eval_dict[f'MRR@{k}'] = average_precision_score(np.array(y_true_bin).flatten(), np.array(y_pred).flatten())

        eval_dict['AUC'] = eval_dict['AUC@1']
        eval_dict['MAP'] = eval_dict['MRR'] = eval_dict['MRR@1']

        return eval_dict

    def do_predict(self, M_train, M_test, P_test=None):
        M_train, M_test = [as_torch_tensor(X).to(self.device) for X in [M_train, M_test]]
        M_train = torch.cat([M_train, torch.log(M_train + 1.0)], dim=1)
        M_test = torch.cat([M_test, torch.log(M_test + 1.0)], dim=1)

        self.eval()
        with torch.no_grad():
            P_hat_train_graph_factors = self.predict_P_graph_factors(M_train)
            P_hat_test_graph_factors = self.predict_P_graph_factors(M_test)
            P_hat_traintest_graph_factors = torch.cat([P_hat_train_graph_factors, P_hat_test_graph_factors], dim=0)
            M_traintest = torch.cat([M_train, M_test], dim=0)

            G_train_test = self.create_train_test_graphs(M_train, P_hat_train_graph_factors, self.P_train_model_factors_init,
                                                         M_test, P_hat_test_graph_factors)
            hetero_graph_ntypes = G_train_test[0].ntypes
            G_test = G_train_test[1]

            if self.use_single_type_hetero_graph:
                G_test = self.to_single_type_heterogeneous_graph(G_test)
                self.set_node_and_edge_dict(G_test)  # for HGT

            node_emb = self.get_input_node_embeddings(G_test, M_traintest, P_hat_traintest_graph_factors,
                                                      self.P_train_model_factors, hetero_graph_ntypes)
            P_hat_all = self.forward(G_test, node_emb, hetero_graph_ntypes)

            if not G_test.is_homogeneous:
                eval_graph_mask = G_test.nodes['graph'].data['eval']
            else:
                eval_graph_mask = G_test.graph_node_eval_mask
            P_hat = P_hat_all[eval_graph_mask].cpu().detach().numpy()

            if P_test is not None:
                print("\n" + "=" * 120)
                test_ev_dict = eval_metrics(Y_true=P_test, Y_pred=P_hat)
                print("Test:", [f"{m}={test_ev_dict[m]:.4f}" for m in sorted(test_ev_dict.keys())])
                print("=" * 120 + "\n")

            return P_hat

    def do_eval(self, P_test, P_hat):
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
        self.fit(P_train, M_train, D_train)
        start = time.time()
        P_hat = self.do_predict(M_train, M_test)
        self.predict_times.append(time.time() - start)
        self.do_eval(P_test, P_hat)
        return P_hat



class DotProductPredictor(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, emb_dict):
        graph_emb, model_emb = emb_dict['graph'], emb_dict['model']
        return torch.mm(graph_emb, model_emb.T)


def knn_edges_x_to_y(X, Y, knn_k):
    cos_sim = cosine_sim_matrix(X, Y)
    knn_k = min(knn_k, len(Y))
    values, indices = torch.topk(cos_sim, knn_k, dim=1)

    u = torch.arange(len(X)).repeat_interleave(knn_k)
    v = indices.reshape(-1)
    return u, v


def cosine_sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sparse_nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X into A*Y
    Ref: https://stackoverflow.com/questions/22767695/python-non-negative-matrix-factorization-that-handles-both-zeros-and-missing-dat
    """
    eps = 1e-5
    # print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    X = np.array(X)  # nans in X denote missing values

    # mask
    mask = ~np.isnan(X)
    X[np.isnan(X)] = 0

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

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
