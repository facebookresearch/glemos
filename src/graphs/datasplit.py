# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops

import settings
from graphs.graphset import Graph
from utils import logger
from utils import set_seed


class DataSplit:
    def __init__(self, raw_graph: Graph, num_splits=5):
        self.raw_graph: Graph = raw_graph
        self.num_splits = num_splits
        self.val_ratio = 0.2  # percentage of training data to be held out for validation
        self._graph_data: Optional[Data] = None

    @property
    def graph_data(self):
        if self._graph_data is None:
            self._graph_data = DataSplit.preprocess_graph_data(self.raw_graph.pyg_graph())
        return self._graph_data

    @classmethod
    def preprocess_graph_data(cls, raw_graph_data: Data):
        graph_data: Data = copy.copy(raw_graph_data)
        edge_index = graph_data.edge_index
        """remove self-loops"""
        edge_index, edge_attr = remove_self_loops(edge_index=edge_index)
        """remove isolated nodes"""
        edge_index, edge_attr, node_mask = remove_isolated_nodes(edge_index=edge_index, num_nodes=graph_data.num_nodes)
        graph_data['edge_index'] = edge_index
        graph_data['non_isolated_node'] = node_mask
        # assert that the number of nodes does not change after preprocessing
        assert raw_graph_data.num_nodes == graph_data.num_nodes, (raw_graph_data.num_nodes, graph_data.num_nodes)
        return graph_data

    def node_split_paths(self, split_i, mkdir=False) -> Dict[str, Path]:
        node_split_root = settings.GRAPH_SPLIT_ROOT / self.raw_graph.name / f"node-{self.num_splits}splits"
        if mkdir:
            node_split_root.mkdir(parents=True, exist_ok=True)
        return {
            'train_node_index': node_split_root / f"train-node-index-split{split_i}.pt",
            'val_node_index': node_split_root / f"val-node-index-split{split_i}.pt",
            'test_node_index': node_split_root / f"test-node-index-split{split_i}.pt"
        }

    def generate_node_splits(self):
        if not self.raw_graph.is_node_labeled:
            print(f"node splits not needed: {self.raw_graph} does not have node labels")
            return

        """include nodes with labels in trainin/validation/testing for node classification task"""
        labeled_node_mask = self.raw_graph.labeled_node_mask()
        assert -1 not in torch.unique(self.raw_graph.node_labels()[labeled_node_mask]).tolist()

        """include non-isolated nodes in training/validation/testing for node classification task"""
        non_isolated_node_mask = self.graph_data['non_isolated_node'].cpu().numpy()
        assert non_isolated_node_mask.ndim == 1, non_isolated_node_mask.shape
        assert len(non_isolated_node_mask) == self.graph_data.num_nodes, (len(non_isolated_node_mask), self.graph_data.num_nodes)

        node_mask = (labeled_node_mask & non_isolated_node_mask).cpu().numpy()
        target_nodes = np.nonzero(node_mask)[0]
        print("target_nodes:", target_nodes.shape)
        assert np.max(target_nodes) <= torch.iinfo(torch.int).max, (np.max(target_nodes), torch.iinfo(torch.int).max)

        """k-fold splits"""
        kf = KFold(n_splits=self.num_splits, random_state=101, shuffle=True)
        splits = kf.split(np.arange(len(target_nodes)))

        for split_i, (train_index, test_index) in enumerate(splits):
            train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=split_i * 101, shuffle=True)

            split_paths = self.node_split_paths(split_i, mkdir=True)
            torch.save(torch.from_numpy(target_nodes[np.sort(train_index)]).int(), split_paths['train_node_index'])
            torch.save(torch.from_numpy(target_nodes[np.sort(val_index)]).int(), split_paths['val_node_index'])
            torch.save(torch.from_numpy(target_nodes[np.sort(test_index)]).int(), split_paths['test_node_index'])

        logger.info(f"node splits generated for {self.raw_graph}")

    def load_node_split(self, split_i):
        if not self.raw_graph.is_node_labeled:
            print(f"node splits not needed: {self.raw_graph} does not have node labels")
            return None
        split_paths = self.node_split_paths(split_i)
        if not all([p.exists() for p in split_paths.values()]):
            self.generate_node_splits()
        return {key: torch.load(path).long() for key, path in split_paths.items()}

    def edge_split_paths(self, split_i, mkdir=False) -> Dict[str, Path]:
        edge_split_root = settings.GRAPH_SPLIT_ROOT / self.raw_graph.name / f"edge-{self.num_splits}splits"
        if mkdir:
            edge_split_root.mkdir(parents=True, exist_ok=True)

        split_paths = {}
        for mode in ['train', 'val', 'test']:
            split_paths[f"{mode}_edge_index"] = edge_split_root / f"{mode}-edge-index-split{split_i}.pt"
            split_paths[f"{mode}_edge_label"] = edge_split_root / f"{mode}-edge-label-split{split_i}.pt"
            split_paths[f"{mode}_edge_label_index"] = edge_split_root / f"{mode}-edge-label-index-split{split_i}.pt"
        return split_paths

    def generate_edge_splits(self):
        if self.graph_data.is_directed():
            logger.info(f"directed self.graph_data: {self.graph_data}")
            graph_data = ToUndirected()(self.graph_data)
            logger.info(f"transformed into an undirected graph: {graph_data}")
        else:
            graph_data = self.graph_data
        assert graph_data.is_undirected()
        assert graph_data.num_nodes <= torch.iinfo(torch.int).max, (graph_data.num_nodes, torch.iinfo(torch.int).max)

        for split_i in range(self.num_splits):
            set_seed(seed=split_i * 101)  # use a different seed for each split

            transform = RandomLinkSplit(is_undirected=True,
                                        num_val=0.16,
                                        num_test=0.2,
                                        neg_sampling_ratio=1.0,
                                        split_labels=False)
            train_data, val_data, test_data = transform(graph_data)

            split_paths = self.edge_split_paths(split_i=split_i, mkdir=True)
            for mode, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                edge_index = data.edge_index.detach().clone().int()
                assert edge_index.ndim == 2, edge_index.shape
                edge_label = data.edge_label.detach().clone().type(torch.int8)
                assert edge_label.ndim == 1, edge_label.shape
                edge_label_index = data.edge_label_index.detach().clone().int()
                assert edge_label_index.ndim == 2, edge_label_index.shape

                torch.save(edge_index, split_paths[f"{mode}_edge_index"])
                torch.save(edge_label, split_paths[f"{mode}_edge_label"])  # 1 if pos edge, 0 if neg edge
                torch.save(edge_label_index, split_paths[f"{mode}_edge_label_index"])

        logger.info(f"edge splits generated for {self.raw_graph}")

    def load_edge_split(self, split_i):
        split_paths = self.edge_split_paths(split_i)
        if not all([p.exists() for p in split_paths.values()]):
            self.generate_edge_splits()
        edge_split = {}
        for key, path in split_paths.items():
            if key.endswith("edge_label"):
                edge_split[key] = torch.load(path).float()
            else:
                edge_split[key] = torch.load(path).long()
        return edge_split


if __name__ == '__main__':
    from graphs import load_graphs

    graphs = load_graphs()
    for graph in graphs:
        logger.info(f"Generating/loading data splits for graph {graph.name}...")
        data_split = DataSplit(graph, num_splits=5)
        edge_split = data_split.load_edge_split(split_i=0)
        if graph.is_node_labeled:
            node_split = data_split.load_node_split(split_i=0)
        break
