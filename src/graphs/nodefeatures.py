# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch.nn import Embedding

import settings
from graphs.graphset import Graph
from utils import set_seed

POOL_NUM_NODES = 1000000
NUM_FEATS = 32


class RandomNodeFeatures:
    def __init__(self, pool_num_nodes, num_feats):
        self.pool_num_nodes = pool_num_nodes
        self.num_feats = num_feats
        self.rand_node_feat_pool = torch.load(self.get_random_node_feature_pool_path(pool_num_nodes=self.pool_num_nodes,
                                                                                     num_feats=self.num_feats))

    @classmethod
    def get_random_node_feature_pool_path(cls, pool_num_nodes, num_feats):
        return settings.GRAPH_SPLIT_ROOT / f"node_feature_pool_{pool_num_nodes}nodes_{num_feats}feats.pt"

    @classmethod
    def generate_random_node_feature_pool(cls, num_feats=None):
        if num_feats is None:
            num_feats = NUM_FEATS
        max_num_nodes = max_num_nodes_in_graphs_without_features()
        assert max_num_nodes <= POOL_NUM_NODES, (max_num_nodes, POOL_NUM_NODES)

        set_seed(101)
        emb = Embedding(num_embeddings=POOL_NUM_NODES, embedding_dim=num_feats)
        print(emb.weight, emb.weight.shape)

        pool_path = cls.get_random_node_feature_pool_path(pool_num_nodes=POOL_NUM_NODES, num_feats=num_feats)
        torch.save(emb.weight, pool_path)
        print(f"random node feature pool saved to {pool_path}.")

    @classmethod
    def get_random_node_feat_index_path(cls, graph: Graph):
        return settings.GRAPH_SPLIT_ROOT / graph.name / "random_node_feat_index.pt"

    def assign_random_node_features(self, graph: Graph, seed=0):
        assert graph.pyg_graph().x is None
        assert self.rand_node_feat_pool.shape[0] >= graph.num_nodes

        set_seed(seed)
        pool_randperm = torch.randperm(self.rand_node_feat_pool.shape[0])
        rand_node_feat_index = pool_randperm[:graph.num_nodes].detach().clone()  # note: clone() should be used to save only the selected subset of pool_randperm.

        torch.save(rand_node_feat_index, self.get_random_node_feat_index_path(graph))
        print(f"[{graph.name}] node feature index (shape={rand_node_feat_index.shape}) saved to {self.get_random_node_feat_index_path(graph)}")

    def load_node_features(self, graph: Graph):
        if graph.pyg_graph().x is not None:  # prioritize the input node features if available
            return graph.pyg_graph().x

        if not self.get_random_node_feat_index_path(graph).exists():
            self.assign_random_node_features(graph)

        node_feat_index = torch.load(self.get_random_node_feat_index_path(graph))
        assert node_feat_index.shape[0] == graph.num_nodes, (node_feat_index.shape[0], graph.num_nodes)
        assert node_feat_index.ndim == 1, node_feat_index.shape

        rand_node_feats = self.rand_node_feat_pool[node_feat_index].detach()
        assert len(rand_node_feats) == graph.num_nodes
        return rand_node_feats


def max_num_nodes_in_graphs_without_features():
    from graphs import load_graphs
    graphs = load_graphs()
    num_nodes_filtered = [g.pyg_graph().num_nodes for g in graphs if g.pyg_graph().x is None]
    print("num_nodes_filtered:", min(num_nodes_filtered), max(num_nodes_filtered))
    num_nodes_all = [g.pyg_graph().num_nodes for g in graphs]
    print("num_nodes_all:", min(num_nodes_all), max(num_nodes_all))

    return max(num_nodes_filtered)


if __name__ == '__main__':
    # max_num_nodes_in_graphs_without_features()
    # RandomNodeFeatures.generate_random_node_feature_pool(num_feats=32)

    random_node_features = RandomNodeFeatures(pool_num_nodes=POOL_NUM_NODES, num_feats=NUM_FEATS)
    from graphs import load_graphs
    from graphs.netrepo_graphs.netrepo_labeled_graphs import load_graphs
    from graphs.pyg_graphs import load_graphs
    for graph in load_graphs():
        if graph.name in ["NELL", "LINKX-genius"]:
            graph.pyg_graph().x = None  # use random features for NELL

        rand_node_feats = random_node_features.load_node_features(graph)
        print(graph.name, "rand_node_feats:", rand_node_feats)
        # break
