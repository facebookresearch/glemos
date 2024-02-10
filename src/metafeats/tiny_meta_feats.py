# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import networkx as nx
import numpy as np
import scipy.sparse as sps
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def tiny_meta_graph_features(data: Data):  # shape=(13,)
    """
    Compute a fixed-size k-dimensional meta-graph feature vector
    """
    # transforms the graph into an undirected graph, and output it to glet_input_graph_fn
    G_nx = to_networkx(data).to_undirected()
    A = nx.to_scipy_sparse_array(G_nx, format="csc")
    E = np.argwhere(sps.triu(A + A.T) > 0)  # shape=(# undirected edges, 2)

    A_simple = A.copy()

    A = A_simple.copy()

    V = np.unique(E)
    num_nodes = V.shape[0]
    num_edges = E.shape[0]
    density_rho = density(A)
    d_max = np.max((A + A.T).getnnz(1))
    d_avg = np.mean((A + A.T).getnnz(1))
    d_med = np.median((A + A.T).getnnz(1))

    G_nx.remove_edges_from(nx.selfloop_edges(G_nx))

    r = 0
    if d_max < 1000:
        r = nx.degree_assortativity_coefficient(G_nx)
        print('finished r')
    else:
        print('skipping r')

    core_nums = nx.core_number(G_nx)
    core_nums = np.array(list(core_nums.values()))
    print(core_nums)

    max_kcore = np.max(core_nums)
    med_kcore = np.median(core_nums)
    mean_kcore = np.mean(core_nums)
    print('max k-core = ', str(max_kcore))

    pagerank = nx.pagerank(G_nx, alpha=0.7)
    pagerank = np.array(list(pagerank.values()))

    max_pagerank = np.max(pagerank)
    med_pagerank = np.median(pagerank)
    mean_pagerank = np.mean(pagerank)

    graph_meta_features = np.array([num_nodes, num_edges, density_rho, d_max, d_med, d_avg, r,
                                    max_kcore, med_kcore, mean_kcore, max_pagerank, med_pagerank, mean_pagerank])

    return graph_meta_features


def density(A):
    n = A.shape[0]
    return A.nnz / (n * (n - 1.0))


if __name__ == '__main__':
    from graphs.netrepo_graphs.netrepo_unlabeled_graphs import load_graphs

    graphs = load_graphs()
    graph = graphs[20]
    print(graph, graph.pyg_graph(), "num_nodes:", graph.num_nodes)

    meta_feat_vec = tiny_meta_graph_features(graph.pyg_graph())
    print("meta_feat_vec:", meta_feat_vec.shape)
