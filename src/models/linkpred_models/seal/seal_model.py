# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from torch.nn import Conv1d, MaxPool1d, ModuleList
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import MLP, GCNConv, SAGEConv, GATConv, SortAggregation
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix

import settings
from utils.log_utils import logger


class SEALDataset(InMemoryDataset):
    def __init__(
            self,
            graph: Data,
            num_hops,
            mode='train',
            split_i=0,
            num_splits=5
    ):
        self.graph = graph
        self.edge_split_root = settings.GRAPH_SPLIT_ROOT / graph.name / f"edge-{num_splits}splits"
        self.num_hops = num_hops
        self.mode = mode
        self.split_i = split_i
        self.num_splits = num_splits
        super().__init__(root=self.edge_split_root)

        index = ['train', 'val', 'test'].index(mode)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def processed_file_names(self):
        return [f"SEAL-{self.graph.name}-{mode}_data-num_hops_{self.num_hops}.pt"
                for mode in ['train', 'val', 'test']]

    def process(self):
        self._max_z = 0

        # Collect a list of subgraphs for training, validation and testing:
        train_pos_data_list = self.extract_enclosing_subgraphs(
            edge_index=self.graph['train_edge_index'],
            edge_label_index=self.graph['train_edge_label_index'][:, self.graph['train_edge_label'].long() == 1],
            y=1
        )
        train_neg_data_list = self.extract_enclosing_subgraphs(
            edge_index=self.graph['train_edge_index'],
            edge_label_index=self.graph['train_edge_label_index'][:, self.graph['train_edge_label'].long() == 0],
            y=0
        )

        val_pos_data_list = self.extract_enclosing_subgraphs(
            edge_index=self.graph['val_edge_index'],
            edge_label_index=self.graph['val_edge_label_index'][:, self.graph['val_edge_label'].long() == 1],
            y=1
        )
        val_neg_data_list = self.extract_enclosing_subgraphs(
            edge_index=self.graph['val_edge_index'],
            edge_label_index=self.graph['val_edge_label_index'][:, self.graph['val_edge_label'].long() == 0],
            y=0
        )

        test_pos_data_list = self.extract_enclosing_subgraphs(
            edge_index=self.graph['test_edge_index'],
            edge_label_index=self.graph['test_edge_label_index'][:, self.graph['test_edge_label'].long() == 1],
            y=1
        )
        test_neg_data_list = self.extract_enclosing_subgraphs(
            edge_index=self.graph['test_edge_index'],
            edge_label_index=self.graph['test_edge_label_index'][:, self.graph['test_edge_label'].long() == 0],
            y=0
        )
        print("max_z:", self._max_z)

        for data in chain(train_pos_data_list, train_neg_data_list,
                          val_pos_data_list, val_neg_data_list,
                          test_pos_data_list, test_neg_data_list):
            data.max_z = self._max_z
        # note: add node features using the computed 'z' attribute in the trainer.py

        torch.save(self.collate(train_pos_data_list + train_neg_data_list),
                   self.processed_paths[0])
        torch.save(self.collate(val_pos_data_list + val_neg_data_list),
                   self.processed_paths[1])
        torch.save(self.collate(test_pos_data_list + test_neg_data_list),
                   self.processed_paths[2])

    def extract_enclosing_subgraphs(self, edge_index, edge_label_index, y):
        data_list = []
        for src, dst in edge_label_index.t().tolist():
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, relabel_nodes=True, num_nodes=self.graph.num_nodes,
            )
            src, dst = mapping.tolist()

            # Remove target link from the subgraph.
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst,
                                        num_nodes=sub_nodes.size(0))

            data = Data(sub_nodes=sub_nodes, z=z, edge_index=sub_edge_index, y=y)
            data_list.append(data)

        return data_list

    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.

        self._max_z = max(int(z.max()), self._max_z)

        return z.to(torch.long)


class DGCNN(torch.nn.Module):
    def __init__(
            self,
            data: Data,
            train_dataset: SEALDataset,
            num_hops: int,
            gnn_hidden_channels: int,
            gnn_conv: str,  # GCNConv (default)
            mlp_hidden_channels: int,  # 128 (default)
            k,  # 0.6 (default)
    ):
        super().__init__()

        self.num_hops = num_hops
        if gnn_conv == "GCN":
            gnn_conv_class = GCNConv
        elif gnn_conv == "SAGE":
            gnn_conv_class = SAGEConv
        elif gnn_conv == "GAT":
            gnn_conv_class = GATConv
        else:
            raise ValueError(f"Unavailable: {gnn_conv}")

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            # data example: Data(edge_index=[2, 16], y=[1], sub_nodes=[7], z=[7], max_z=[1])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = int(max(10, k))

        if data.x is not None:
            num_input_feats = data.x.size(1)
        else:
            num_input_feats = data.rand_x.size(1)

        max_z = list(set([data.max_z.item() for data in train_dataset]))
        assert len(max_z) == 1, len(max_z)
        num_feats = num_input_feats + (max_z[0] + 1)

        self.gnn_convs = ModuleList()
        self.gnn_convs.append(gnn_conv_class(num_feats, gnn_hidden_channels))
        num_layers = num_hops
        for i in range(0, num_layers - 1):
            self.gnn_convs.append(gnn_conv_class(gnn_hidden_channels, gnn_hidden_channels))
        self.gnn_convs.append(gnn_conv_class(gnn_hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = gnn_hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.pool = SortAggregation(k)
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.mlp = MLP([dense_dim, mlp_hidden_channels, 1], dropout=0.5, norm=None)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for gnn_conv in self.gnn_convs:
            xs += [gnn_conv(xs[-1], edge_index).tanh()]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = self.pool(x, batch)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = self.conv1(x).relu()
        x = self.maxpool1d(x)
        x = self.conv2(x).relu()
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        return self.mlp(x)


if __name__ == '__main__':
    from performances.taskrunner import Runner, Task
    from graphs.graphset import GraphSet

    graphset = GraphSet(data_sources=['netrepo', 'pyg'], sort_graphs='num_edges')
    num_graphs = len(graphset.link_prediction_graphs())
    print("num_graphs:", num_graphs)
    runner = Runner(task=Task.LINK_PREDICTION, root=Path("~/Downloads").expanduser(), args=None, graphset=graphset)

    for graph_i in range(num_graphs):
        graph: Data = runner.load_graph(graph_i=graph_i, split_i=0)
        logger.info(f"start processing graph {graph.name} (graph_i: {graph_i}, num-nodes={graph.num_nodes}, num-edges={graph.num_edges})...")
        SEALDataset(graph, num_hops=1, mode='train')
