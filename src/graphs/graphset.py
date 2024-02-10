# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from importlib import import_module
from typing import Optional, List, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.dataset import Dataset
from torch_geometric.utils import to_networkx

from graphs.graphdomain import GraphDomain
from utils import logger


class Graph:
    def __init__(self,
                 dataset: Dataset,
                 domain: GraphDomain,
                 name: Optional[str] = None):
        if name is None and hasattr(dataset, 'name'):
            self._name = dataset.name
        else:
            self._name = name
        self.dataset = dataset
        assert len(dataset) == 1, len(dataset)
        self.data: Union[Data, HeteroData] = dataset[0]
        self.hm_data: Data = None  # set only when a heterogeneous graph is transformed into a homogeneous graph
        self.domain = domain
        self.num_nodes = self.set_and_get_num_nodes()
        self.num_classes = self.set_and_get_num_classes()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def set_and_get_num_nodes(self):
        data = self.pyg_graph()
        if not hasattr(data, "num_nodes"):
            if hasattr(data, "x"):
                data['num_nodes'] = data.x.size(0)
            else:
                data['num_nodes'] = data.edge_index.max().item() + 1

        num_nodes = data.num_nodes
        if hasattr(data, "x") and data.x is not None:
            assert num_nodes == data.x.size(0)
        assert num_nodes == data.edge_index.max().item() + 1, (num_nodes, data.edge_index.max() + 1)
        assert hasattr(self.pyg_graph(), "num_nodes")

        return num_nodes

    def set_and_get_num_classes(self):
        if not self.is_node_labeled:
            return -1

        data = self.pyg_graph()
        y = data.y if data.y is not None else data.train_y
        if not hasattr(data, "num_classes"):
            assert y.ndim <= 2, y.ndim
            if y.ndim == 2:
                num_classes = y.shape[1]
            else:
                num_classes = y.max().item() + 1
            data.num_classes = num_classes

        num_classes = data.num_classes
        if y.ndim == 1:
            assert y.min().item() in [0, -1], y.min()
            assert num_classes == y.max().item() + 1, (num_classes, y.max() + 1)
        assert hasattr(self.pyg_graph(), "num_classes")

        return num_classes

    def pyg_graph(self):
        if isinstance(self.data, HeteroData):
            if self.hm_data is None:
                self.hm_data = self.pyg_heterogeneous_to_homogeneous(self.data)
            return self.hm_data
        else:
            return self.data

    @classmethod
    def pyg_heterogeneous_to_homogeneous(cls, hetero_data: HeteroData):
        hm_data = hetero_data.to_homogeneous()
        # note:
        # - nodes of the type for which no node labels were given are assigned -1
        #   as their node label in the resulting homogeneous graph (i.e., in hm_data.y)
        # - ignore train/test node masks in the resulting homogeneous graph
        #   (the train/test masks of the nodes for which no mask information was given are all set to True by default)
        # - node features are available only when all node types have node features and all of them are of the same size
        return hm_data

    def networkx_graph(self):
        return to_networkx(self.pyg_graph())

    @property
    def is_node_labeled(self):
        return 'y' in self.pyg_graph() or ('train_y' in self.pyg_graph() and 'train_idx' in self.pyg_graph())

    def labeled_node_mask(self) -> torch.BoolTensor:
        if not self.is_node_labeled:
            return torch.full((self.num_nodes,), fill_value=False)

        if 'train_y' in self.pyg_graph() and 'train_idx' in self.pyg_graph():  # pyg Entities graph
            node_mask = torch.full((self.num_nodes,), fill_value=False)
            node_mask[self.pyg_graph()['train_idx']] = True
            node_mask[self.pyg_graph()['test_idx']] = True
            return node_mask
        else:
            if self.name == "AttributedGraph-TWeibo":  # not all nodes have node labels
                assert len(self.pyg_graph().y) <= self.num_nodes
                node_mask = torch.full((self.num_nodes,), fill_value=True)
                node_mask[len(self.pyg_graph().y):] = False
                return node_mask

            assert self.pyg_graph().y is not None and self.num_nodes == len(self.pyg_graph().y), \
                (self.name, self.num_nodes, self.pyg_graph().y.shape)
            assert self.pyg_graph().y.ndim in [1, 2], self.pyg_graph().y.ndim

            if self.pyg_graph().y.ndim == 1:
                # some nodes may have negative labels. for instance,
                # - heterogeneous graph data: nodes of the types without node labels are given -1 as their label
                # - LINKXDataset
                return self.pyg_graph().y >= 0
            else:
                return torch.full((self.num_nodes,), fill_value=True)


    def node_labels(self) -> Union[torch.LongTensor, None]:
        if not self.is_node_labeled:
            return None

        if 'train_y' in self.pyg_graph() and 'train_idx' in self.pyg_graph():  # pyg Entities graph
            node_labels = torch.full((self.num_nodes,), fill_value=-1)
            node_labels[self.pyg_graph()['train_idx']] = self.pyg_graph()['train_y']
            node_labels[self.pyg_graph()['test_idx']] = self.pyg_graph()['test_y']
            return node_labels
        else:
            assert set(torch.unique(self.pyg_graph().y).tolist()).issubset([-1] + list(range(self.num_classes)))
            if self.name == "AttributedGraph-TWeibo":  # not all nodes have node labels
                return torch.cat([self.pyg_graph().y, torch.tensor([-1] * (self.num_nodes - len(self.pyg_graph().y)))])
            else:
                return self.pyg_graph().y

    def __repr__(self):
        return self.name


class GraphSet:
    all_data_sources = ['pyg', 'netrepo']

    def __init__(self, data_sources=None, sort_graphs='num_edges'):
        if data_sources is None:
            data_sources = GraphSet.all_data_sources
        self.data_sources = data_sources
        assert any([d in self.all_data_sources for d in self.data_sources]), self.data_sources

        assert sort_graphs in ['num_edges', 'num_nodes', None], sort_graphs
        self.sort_graphs = sort_graphs
        if self.sort_graphs == 'num_nodes':  # small graphs first
            self.graphs: List[Graph] = sorted(self.load_graphs(), key=lambda g: g.pyg_graph().num_nodes, reverse=False)
        elif self.sort_graphs == 'num_edges':  # small graphs first
            self.graphs: List[Graph] = sorted(self.load_graphs(), key=lambda g: g.pyg_graph().num_edges, reverse=False)
        else:
            self.graphs: List[Graph] = self.load_graphs()

    def load_graphs(self) -> List[Graph]:
        logger.info(f"Loading graphs...")
        graphs = []
        for data_source in self.data_sources:
            m = import_module(f"graphs.{data_source}_graphs")
            graphs += m.load_graphs()
        logger.info(f"Loaded {len(graphs)} graphs")
        return graphs

    def __len__(self):
        return len(self.graphs)

    def node_classification_graphs(self):
        return [graph for graph in self.graphs if graph.is_node_labeled]

    def link_prediction_graphs(self):
        return self.graphs


def print_graph_stats(graphs):
    for g in graphs:
        pg = g.pyg_graph()
        if hasattr(pg, 'num_classes'):
            num_classes = pg.num_classes
        else:
            num_classes = None
        print(f"{g.name}:\n"
              f"\t{pg},\n"
              f"\tnum_nodes: {pg.num_nodes}, \n"
              f"\tnum_classes: {num_classes}, \n"
              f"\tis_node_labeled: {g.is_node_labeled}, \n"
              f"\tlen(dataset): {len(g.dataset)}")
    print("# graphs:", len(graphs))


if __name__ == '__main__':
    graph_set = GraphSet(['netrepo', 'pyg'])
    node_classification_graphs = graph_set.node_classification_graphs()
    print("node classification graphs:", len(node_classification_graphs), node_classification_graphs)
    num_nodes = [g.pyg_graph().num_nodes for g in node_classification_graphs]
    num_edges = [g.pyg_graph().num_edges for g in node_classification_graphs]
    print("\tnum_nodes:", min(num_nodes), max(num_nodes))
    print("\tnum_edges:", min(num_edges), max(num_edges))

    link_prediction_graphs = graph_set.link_prediction_graphs()
    print("link prediction graphs:", len(link_prediction_graphs), link_prediction_graphs)
    print_graph_stats(link_prediction_graphs)
    num_nodes = [g.pyg_graph().num_nodes for g in link_prediction_graphs]
    num_edges = [g.pyg_graph().num_edges for g in link_prediction_graphs]
    print("\tnum_nodes:", min(num_nodes), max(num_nodes))
    print("\tnum_edges:", min(num_edges), max(num_edges))
