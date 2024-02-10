# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sps
import torch
import torch.nn as nn
from scipy.sparse import linalg
from torch_geometric.data import Data
from torch_geometric.nn import DeepGraphInfomax, GCNConv, Node2Vec
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, to_networkx


class SpectralEmbedding(nn.Module):
    def __init__(self, num_components, out_channels, tolerance, in_channels=None):
        super().__init__()
        self.num_components = num_components  # latent node embeddings obtained by spectral embedding
        self.out_channels = out_channels  # final node embedding
        self.in_channels = in_channels  # not used. left for consistency with other methods.

        self.tolerance = tolerance

        self.node_emb_transform: Optional[nn.Module] = None
        self.latent_node_emb: Optional[torch.Tensor] = None  # unsupervised node embeddings

    def train_unsupervised_node_emb(self, pyg_graph: Data, device, logger, run_root=None):
        if self.latent_node_emb is not None:
            return

        if 'train_edge_index' in pyg_graph:  # use training graph for link prediction task
            edge_index = pyg_graph['train_edge_index']  # train_edge_index has been preprocessed to be undirected
        else:
            if pyg_graph.is_directed():
                edge_index = to_undirected(edge_index=pyg_graph.edge_index)
            else:
                edge_index = pyg_graph.edge_index

        # add remaining self-loops
        self_loops = torch.stack([torch.arange(pyg_graph.num_nodes),
                                  torch.arange(pyg_graph.num_nodes)], dim=0).to(edge_index.device)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        from torch_geometric.transforms import RemoveDuplicatedEdges
        transform = RemoveDuplicatedEdges()
        out = transform(Data(edge_index=edge_index))
        edge_index = out.edge_index

        coo_A = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=pyg_graph.num_nodes)
        A = sps.csc_matrix(coo_A)
        logger.info(f"A.shape = {A.shape} {np.max(A.shape)}")
        # A = A + A.T  # symmetrize - already done above.

        e = np.ones(A.shape[1])
        degs = A.dot(e)
        Dinv = np.ravel(1. / np.power(degs, 0.5))
        Dinv = sps.diags(Dinv, 0)
        L = Dinv.dot(A.dot(Dinv)).tocsc()
        # L, degs = sps.csgraph.laplacian(A, normed=True, return_diag=True)   # returns COO format
        # L = L.tocsc()
        L = (L + L.transpose()) / 2.

        if pyg_graph.num_nodes <= self.num_components:
            k = pyg_graph.num_nodes - 1
        else:
            k = self.num_components
        self.node_emb_transform = nn.Linear(k, self.out_channels)

        eigenvals, X = sps.linalg.eigsh(L, k=k, tol=self.tolerance, which='LM')
        logger.info(f"latent_node_emb: {X.shape}")
        assert not np.isnan(X).any()
        self.latent_node_emb = torch.from_numpy(X.copy()).float().to(device)

        return self

    def forward(self, x, edge_index):
        assert self.latent_node_emb is not None and self.node_emb_transform is not None, (self.latent_node_emb, self.node_emb_transform)
        return self.node_emb_transform(self.latent_node_emb)


class GraRep(nn.Module):
    def __init__(self, num_components, out_channels, power, in_channels=None):
        super().__init__()
        self.num_components = num_components  # latent node embeddings obtained by spectral embedding
        self.out_channels = out_channels  # final node embedding
        self.in_channels = in_channels  # not used. left for consistency with other methods.

        self.power = power

        self.node_emb_transform: Optional[nn.Module] = None
        self.latent_node_emb: Optional[torch.Tensor] = None  # unsupervised node embeddings

    def train_unsupervised_node_emb(self, pyg_graph: Data, device, logger, run_root=None):
        if self.latent_node_emb is not None:
            return

        if 'train_edge_index' in pyg_graph:  # use training graph for link prediction task
            edge_index = pyg_graph['train_edge_index']
        else:
            edge_index = pyg_graph.edge_index

        coo_A = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=pyg_graph.num_nodes)
        A = sps.csc_matrix(coo_A)
        logger.info(f"A.shape = {A.shape} {np.max(A.shape)}")

        for k_pow in range(1, self.power):
            A = A * A  # raise A to a power
            print(f"[k_pow={k_pow}] nnz(A) = {str(A.nnz)}")
        logger.info(f"nnz(A) = {str(A.nnz)}")

        n = pyg_graph.num_nodes
        if n < self.num_components:
            k = n - 1
        else:
            k = self.num_components
        self.node_emb_transform = nn.Linear(k, self.out_channels)

        X, singular_vals, V = sps.linalg.svds(A, k=k)  # , which='SM')
        logger.info(f"latent_node_emb: {X.shape}")
        assert not np.isnan(X).any()

        self.latent_node_emb = torch.from_numpy(X.copy()).float().to(device)
        return self

    def forward(self, x, edge_index):
        assert self.latent_node_emb is not None
        return self.node_emb_transform(self.latent_node_emb)


class DGI(nn.Module):
    class DGIEncoder(nn.Module):
        def __init__(self, in_channels, hidden_channels, act):
            super().__init__()
            self.conv = GCNConv(in_channels, hidden_channels, cached=True)
            self.act = {
                'prelu': nn.PReLU(hidden_channels),
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
            }[act]

        def forward(self, x, edge_index):
            x = self.conv(x, edge_index)
            x = self.act(x)
            return x

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def __init__(self, in_channels, out_channels, hidden_channels, summary, encoder_act):
        super().__init__()
        self.in_channels = in_channels  # input feature size
        self.out_channels = out_channels  # final node embedding
        self.hidden_channels = hidden_channels  # latent node embeddings
        self.summary = summary
        assert summary in ['mean', 'max', 'min', 'var'], summary
        self.encoder_act = encoder_act
        self.epochs = 300

        self.latent_node_emb: Optional[torch.Tensor] = None  # unsupervised node embeddings
        self.node_emb_transform: nn.Module = nn.Linear(self.hidden_channels, self.out_channels)

    def train_unsupervised_node_emb(self, data: Data, device, logger, run_root=None):
        if self.latent_node_emb is not None:
            return

        if self.summary == 'mean':
            summary_fn = lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0))
        elif self.summary == 'max':
            summary_fn = lambda z, *args, **kwargs: torch.sigmoid(z.max(dim=0).values)
        elif self.summary == 'min':
            summary_fn = lambda z, *args, **kwargs: torch.sigmoid(z.min(dim=0).values)
        elif self.summary == 'var':
            summary_fn = lambda z, *args, **kwargs: torch.sigmoid(z.var(dim=0))
        else:
            raise ValueError(f"Invalid: {self.summary}")

        model = DeepGraphInfomax(
            hidden_channels=self.hidden_channels,
            encoder=DGI.DGIEncoder(self.in_channels, self.hidden_channels, self.encoder_act),
            summary=summary_fn,
            corruption=DGI.corruption
        ).to(device)
        data = data.to(device)

        if 'train_edge_index' in data:  # use training graph for link prediction task
            edge_index = data['train_edge_index']
        else:
            edge_index = data.edge_index

        if data.x is not None:
            data_x = data.x.float()
            if isinstance(data_x, SparseTensor):  # gcn edge weight normalization does not work when given a sparse feature tensor
                data_x = data_x.to_dense()
            parameters = model.parameters()
        else:
            data_x = data.rand_x.float()
            data_x = torch.nn.Parameter(data_x, requires_grad=True)  # learnable node features
            parameters = list(model.parameters()) + [data_x]

        logger.info(f'Starting training')
        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")

        optimizer = torch.optim.Adam(parameters, lr=0.001)
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            pos_z, neg_z, summary = model(data_x, edge_index)

            loss = model.loss(pos_z, neg_z, summary)
            logger.info(f'[Epoch-{epoch:03d}] Train Loss={loss.item():.6f}')

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            z, _, _ = model(data_x, edge_index)
            logger.info(f"z: {z.shape}")

        self.latent_node_emb = z.detach().clone()
        return self

    def forward(self, x, edge_index):
        assert self.latent_node_emb is not None
        return self.node_emb_transform(self.latent_node_emb)


class Node2VecModel(nn.Module):
    def __init__(self, out_channels, hidden_channels, p, q, walk_length, context_size,
                 # walks_per_node,
                 in_channels=None, epochs=10):
        super().__init__()
        self.in_channels = in_channels  # input feature size
        self.out_channels = out_channels  # final node embedding
        self.hidden_channels = hidden_channels  # latent node embeddings
        self.p = float(p)
        self.q = float(q)
        self.walk_length = walk_length
        self.context_size = context_size
        # self.walks_per_node = walks_per_node
        self.epochs = epochs

        self.latent_node_emb: Optional[torch.Tensor] = None  # unsupervised node embeddings
        self.node_emb_transform: nn.Module = nn.Linear(self.hidden_channels, self.out_channels)

    def train_unsupervised_node_emb(self, data: Data, device, logger, run_root):
        if self.latent_node_emb is not None:
            return

        if 'train_edge_index' in data:  # use training graph for link prediction task
            edge_index = data['train_edge_index']
        else:
            edge_index = data.edge_index
        # edge_list = edge_index.t().cpu().detach().numpy()
        # np.savetxt(edge_list_path, edge_list, delimiter=",", fmt='%i')

        edge_list_path = run_root / "grape_edge_list.csv"
        logger.info(f"edge_list_path: {edge_list_path}")
        nx_G = to_networkx(Data(edge_index=edge_index, num_nodes=data.num_nodes))
        nx_G = nx_G.to_undirected()
        nx.write_edgelist(nx_G, edge_list_path, delimiter=",", data=False)

        from grape import Graph
        from grape.embedders import Node2VecGloVeEnsmallen, Node2VecSkipGramEnsmallen, Node2VecCBOWEnsmallen
        model = Node2VecSkipGramEnsmallen(
            embedding_size=self.hidden_channels,
            walk_length=self.walk_length,
            window_size=self.context_size,
            return_weight=1 / self.p,
            explore_weight=1 / self.q,
            number_of_negative_samples=5,
            epochs=self.epochs,
        )

        logger.info(f'Starting training')
        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")

        graph = Graph.from_csv(
            edge_path=str(edge_list_path),
            edge_list_separator=',',
            edge_list_header=False,
            edge_list_numeric_node_ids=True,
            sources_column_number=0,
            destinations_column_number=1,
            directed=False,
            verbose=True,
            number_of_nodes=data.num_nodes,
        )

        emb = model.fit_transform(graph, return_dataframe=True)
        node_emb = emb.get_node_embedding_from_index(0)
        node_emb.index = node_emb.index.astype(int)
        assert node_emb.equals(node_emb.sort_index())

        z = node_emb.to_numpy()
        z = torch.from_numpy(z)
        assert z.shape[0] == data.num_nodes, (z.shape[0], data.num_nodes)
        logger.info(f"z: {z.shape}")

        edge_list_path.unlink()

        self.latent_node_emb = z.to(device)
        return self

    def train_unsupervised_node_emb_prev(self, data: Data, device, logger, run_root=None):
        if self.latent_node_emb is not None:
            return

        if 'train_edge_index' in data:  # use training graph for link prediction task
            edge_index = data['train_edge_index']
        else:
            edge_index = data.edge_index

        model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=self.hidden_channels,
            walk_length=self.walk_length,
            context_size=10,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=1,
            num_nodes=data.num_nodes,
            # sparse=True,
        ).to(device)
        loader = model.loader(batch_size=1024, shuffle=True, num_workers=8)

        logger.info(f'Starting training')
        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")

        # optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()

                loss = model.loss(pos_rw.to(device), neg_rw.to(device))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            loss = total_loss / len(loader)
            logger.info(f'[Epoch-{epoch:03d}] Train Loss={loss:.6f}')

        with torch.no_grad():
            model.eval()

            z = model(torch.arange(data.num_nodes, device=device))
            logger.info(f"z: {z.shape}")

        self.latent_node_emb = z.detach().clone()
        return self

    def forward(self, x, edge_index):
        assert self.latent_node_emb is not None
        return self.node_emb_transform(self.latent_node_emb)


if __name__ == '__main__':
    from torch_geometric.datasets import KarateClub
    graph = KarateClub()[0]

    from graphs.graphset import GraphSet
    # graphset = GraphSet(data_sources=['netrepo', 'pyg'], sort_graphs='num_edges')
    graphset = GraphSet(data_sources=['netrepo'], sort_graphs='num_edges')
    import argparse
    import settings
    from utils import logger
    from performances.taskrunner import Runner, Task
    args = {'lr': 0.001, 'epochs': 2, 'patience': 10, 'device': torch.device('cpu'),
            'node_batch_size': 100, 'edge_batch_size': 300}
    runner = Runner(task=Task.LINK_PREDICTION, root=settings.NODE_CLASS_PERF_ROOT, args=argparse.Namespace(**args),
                    graphset=graphset)
    graph = runner.load_graph(graph_i=0, split_i=0)

    print("graph:", graph)

    logger.info("SpectralEmbedding start")
    spectral_embedding = SpectralEmbedding(num_components=32, out_channels=16, tolerance=0.0001)
    spectral_embedding.train_unsupervised_node_emb(graph, torch.device('cpu'), logger)
    logger.info("SpectralEmbedding finish")

    logger.info("GraRep start")
    grarep = GraRep(num_components=32, out_channels=16, power=-1)
    grarep.train_unsupervised_node_emb(graph, torch.device('cpu'), logger)
    logger.info("GraRep finish")

    logger.info("DGI start")
    in_channels = graph.x.shape[1] if hasattr(graph, 'x') and graph.x is not None else graph.rand_x.shape[1]
    grarep = DGI(in_channels=in_channels, hidden_channels=32, out_channels=16, summary='mean', encoder_act='prelu')
    grarep.train_unsupervised_node_emb(graph, torch.device('cpu'), logger)
    logger.info("DGI finish")

    logger.info("Node2Vec start")
    node2vec = Node2VecModel(hidden_channels=32, out_channels=16, p=1, q=1, walk_length=10, context_size=5, epochs=2)
    from pathlib import Path
    node2vec.train_unsupervised_node_emb(graph, torch.device('cpu'), logger, run_root=Path("~/Downloads/").expanduser())
    logger.info("Node2Vec finish")
