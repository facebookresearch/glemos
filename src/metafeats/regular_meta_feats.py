# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from metafeats.statdist import meta_features_per_structural_property


def regular_meta_graph_features(data: Data):  # shape=(318,)
    """
    Compute a fixed-size k-dimensional meta-graph feature vector
    """
    debug = False

    G_nx = to_networkx(data).to_undirected()
    A = nx.to_scipy_sparse_array(G_nx, format="csc")
    print("A.shape = ")
    print(A.shape, np.max(A.shape))

    density_Asq = density(A * A.T)
    if debug: print("density(AA^T) = " + str(density_Asq))

    density_rho = density(A)
    d_max = np.max((A + A.T).getnnz(1))

    deg = (A + A.T).getnnz(1)
    if debug: print('deg=', deg)

    deg_meta_features = meta_features_per_structural_property(deg)
    if debug: print('deg_meta_features=', deg_meta_features)

    G_nx.remove_edges_from(nx.selfloop_edges(G_nx))

    r = 0
    if d_max < 1000:
        r = nx.degree_assortativity_coefficient(G_nx)
        if debug: print('finished r')
    else:
        if debug: print('skipping r')

    core_nums = nx.core_number(G_nx)

    core_nums = np.array(list(core_nums.values()))
    max_kcore = np.max(core_nums)
    if debug: print('max_kcore=', max_kcore)
    if debug: print('core_nums=', core_nums)

    core_num_meta_features = meta_features_per_structural_property(core_nums)
    if debug: print('core_num_meta_features=', core_num_meta_features)

    pagerank = nx.pagerank(G_nx, alpha=0.7)
    pagerank = np.array(list(pagerank.values()))
    if debug: print('pagerank=', pagerank)

    pagerank_meta_features = meta_features_per_structural_property(pagerank)
    if debug: print('pagerank_meta_features=', pagerank_meta_features)

    if True:
        wedges = deg * (deg - 1.) / 2.0
        if debug: print('wedges=', wedges)
        wedge_meta_features = meta_features_per_structural_property(wedges)
        if debug: print('wedges_meta_features=', wedge_meta_features)

        tri = np.array(list(nx.triangles(G_nx).values()))
        if debug: print('tri=', tri)
        tri_meta_features = meta_features_per_structural_property(tri)
        if debug: print('tri_meta_features=', tri_meta_features)

    graph_meta_features = [density_rho, density_Asq, r]
    graph_meta_features.extend(deg_meta_features)
    graph_meta_features.extend(core_num_meta_features)
    graph_meta_features.extend(pagerank_meta_features)

    graph_meta_features.extend(wedge_meta_features)
    graph_meta_features.extend(tri_meta_features)

    graph_meta_features = np.array(graph_meta_features).flatten()
    return graph_meta_features


def density(A):
    n = A.shape[0]
    return A.nnz / (n * (n - 1.0))


if __name__ == '__main__':
    from graphs.netrepo_graphs.netrepo_unlabeled_graphs import load_graphs
    graphs = load_graphs()
    graph = graphs[20].pyg_graph()
    print(graph, "num_nodes:", graph.num_nodes, "num_edges:", graph.num_edges)

    feats = regular_meta_graph_features(graph)
    print("feats:", feats.shape)
