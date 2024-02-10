# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import os.path
import os.path

import networkx as nx
import numpy as np
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from metafeats import GLET_PATH
from metafeats.statdist import meta_features_per_structural_property
from utils import tic, toc


def compact_meta_graph_features(data: Data, glet_input_graph_fn, output_orbits_fn):
    """
    Compute a fixed-size k-dimensional meta-graph feature vector

    INPUT:
        data:
            input graph data
        glet_input_graph_fn:
            the path to store a preprocessed version of the graph derived from E
        output_orbits_fn:
            the path to store the edge orbit counts of the graph derived from E

    OUTPUT:
        meta_feat_vec: (58,)
        meta_feat_vec_simple: (16,)
        meta_feat_vec_graphlet_dist: (6,)
    """
    if not GLET_PATH.exists():
        raise Exception(f"glet binary does not exist at {GLET_PATH}. "
                        f"Build glet binary first before calling this function.")

    # transforms the graph into an undirected graph, and output it to glet_input_graph_fn
    G_nx = to_networkx(data).to_undirected()
    A = nx.to_scipy_sparse_array(G_nx, format="csc")
    E = E_ub = np.argwhere(sps.triu(A + A.T) > 0)  # shape=(# undirected edges, 2)

    if not glet_input_graph_fn.parent.exists():
        glet_input_graph_fn.parent.mkdir(parents=True)
    np.savetxt(glet_input_graph_fn, E_ub, delimiter=',', fmt='%s')

    orbit_size = 4

    """
    Compute graphlet orbits using glet
    $ ./glet/glet -k 4 -t edge graph-toy-example/graph-toy-example-ub10.edges graph-toy-example/graph-toy-example-ub10.4-edge-orbits

    orbit_size is the orbit size in terms of the number of nodes (4,5)
    overwrite_orbits_file is a boolean flag indicating whether to recompute the orbit counts for the graph if it already exists
    """

    print("max deg, mean deg, max out-deg, mean out-deg:")
    print(np.max(np.sum(A, axis=1)), np.mean(np.sum(A, axis=1)), np.max(np.sum(A, axis=0)), np.mean(np.sum(A, axis=0)))

    A_simple = A.copy()
    A = A + A.T

    print("max deg, mean deg, max out-deg, mean out-deg:")
    print(np.max(np.sum(A, axis=1)), np.mean(np.sum(A, axis=1)), np.max(np.sum(A, axis=0)), np.mean(np.sum(A, axis=0)))

    print(os.path.isfile(output_orbits_fn), output_orbits_fn)
    cmd = f'{str(GLET_PATH)} -k ' + str(orbit_size) + ' -t edge ' + str(glet_input_graph_fn) + ' ' + str(output_orbits_fn)

    if not os.path.isfile(output_orbits_fn) or os.stat(output_orbits_fn).st_size == 0:
        exitCode = os.system(cmd)
        print(exitCode)
        print('exitCode glet: ', exitCode)
    print('glet file size: ', os.stat(output_orbits_fn).st_size)
    print(output_orbits_fn)

    while os.stat(output_orbits_fn).st_size == 0:
        print('glet file size: ', os.stat(output_orbits_fn).st_size)
        try:
            exitCode = os.system(cmd)
            print(exitCode)
            print('exitCode glet: ', exitCode)
            print('glet file size: ', os.stat(output_orbits_fn).st_size, )
            print('\n')
        except Exception:
            print('ERROR: Trying to compute graphlets from scratch again...')

    """
    4 EDGE ORBIT names:
        NOTE 'edge' is added below where each edge is set with a weight of 1
    """
    edge_orbit_names_all = ['edge', '2-stars (P_3)', 'triangles (K_3)',
                            '4-path-edge (P_4)', '4-path-center (P_4)', '3-star (claw)', '4-cycle (C_4)',
                            'tailed-tri-tailEdge (paw-tailEdge)', 'tailed-tri-edge (paw-edge)',
                            'tailed-tri-center (paw-center)',
                            'chordal-cycle-edge (diamond-edge)', 'chordal-cycle-center (diamond-center)',
                            '4-clique (K_4)']

    E_orbits = np.loadtxt(output_orbits_fn, comments=('%', '#'), delimiter=',', skiprows=0)
    E_orbits = np.array(E_orbits, dtype=int)
    print(E_orbits)
    print(E_orbits.shape)

    E_orbits_i = E_orbits[:, 0]
    E_orbits_j = E_orbits[:, 1]
    E_orbits = E_orbits[:, 2:]

    n = A.shape[0]

    sum_orbit_edge_freq = np.round(np.sum(E_orbits, axis=0), 2)

    orbit_col_names = '2-stars ($P_3$) & triangles ($K_3$) & 4-path-edge ($P_4$)  &  4-path-center ($P_4$)  &  3-star (claw) & 4-cycle ($C_4$)  &  '
    orbit_col_names += 'tailed-tri-tailEdge (paw-tailEdge) &  tailed-tri-edge (paw-edge)  &  tailed-tri-center (paw-center)  &  '
    orbit_col_names += 'chordal-cycle-edge (diamond-edge) &  chordal-cycle-center (diamond-center)  &  4-clique ($K_4$)  &  '

    orbit_weight_norm = np.array([1. / 2.,
                                  1. / 3.,
                                  1.0 / 2.0,
                                  1.0,
                                  1.0 / 3.0,
                                  1.0 / 4.0,
                                  1.0,
                                  1.0 / 2.0,
                                  1.0,
                                  1.0 / 4.0,
                                  1.0,
                                  1.0 / 6.0])

    graphlet_meta_features_vec = []

    edge_orbit_names = edge_orbit_names_all
    """
    Deriving motif-based matrices from the counts of the various motifs (\ie, graphlet/orbit/subgraph patterns)
    """
    sec_motif_matrix_time = tic()

    A_k = [A]
    if orbit_size == 4:
        for i in range(0, len(edge_orbit_names) - 1):
            print("\nderiving motif adj matrix for " + edge_orbit_names[i + 1] + ", " + str(i + 1) + "/" + str(len(edge_orbit_names) - 1))
            I = np.argwhere(E_orbits[:, i] > 0).flatten()

            graphlet_mf_tmp = meta_features_per_structural_property(E_orbits[:, i])
            graphlet_meta_features_vec.extend(graphlet_mf_tmp)

            sec_A = tic()
            A_motif = coo_matrix((E_orbits[I, i], (E_orbits_i[I], E_orbits_j[I])), dtype=float, shape=(n, n))
            A_motif = sps.csc_matrix(A_motif)

            A_motif = A_motif + A_motif.T

            A = A_motif

            sec_A = toc(sec_A)
            print("[A_motif matrix] TIME: " + str(sec_A))
            print("A_motif.shape = (" + str(A_motif.shape[0]) + "," + str(A_motif.shape[1]) + "),  max=" + str(np.max(A_motif.shape)))
            A_k.append(A_motif)

    else:
        for i in range(0, E_orbits.shape[1] - 1):
            print("\nderiving motif adj matrix for " + edge_orbit_names[i + 1] + ", " + str(i + 1) + "/" + str(len(edge_orbit_names) - 1))
            I = np.argwhere(E_orbits[:, i] > 0).flatten()

            sec_A = tic()
            A_motif = coo_matrix((E_orbits[I, i], (E_orbits_i[I], E_orbits_j[I])), dtype=float, shape=(n, n))
            A_motif = sps.csc_matrix(A_motif)

            sec_A = toc(sec_A)
            print("[A_motif matrix] TIME: " + str(sec_A))
            print("A_motif.shape = (" + str(A_motif.shape[0]) + "," + str(A_motif.shape[1]) + "),  max=" + str(np.max(A_motif.shape)))
            A_k.append(A_motif)

    sec_motif_matrix_time = toc(sec_motif_matrix_time)
    print("\noverall TIME: " + str(sec_motif_matrix_time))
    print("motifs = " + str(edge_orbit_names))
    print("number of motif-based matrices = " + str(len(A_k)))

    print(edge_orbit_names)
    print(len(A_k), A_k[0].shape)

    mean_orbit_edge_freq = np.round(np.mean(E_orbits, axis=0), 2)  # shape=(12,)
    med_orbit_edge_freq = np.array(np.median(E_orbits, axis=0), dtype=int)  # shape=(12,)
    max_orbit_edge_freq = np.max(E_orbits, axis=0)  # shape=(12,)
    print('mean_orbit_edge_freq: ', mean_orbit_edge_freq.shape, mean_orbit_edge_freq)
    print('med_orbit_edge_freq: ', med_orbit_edge_freq.shape, med_orbit_edge_freq)
    print('max_orbit_edge_freq: ', max_orbit_edge_freq.shape, max_orbit_edge_freq)

    A = A_simple.copy()
    V = np.unique(E)
    num_nodes = V.shape[0]
    num_edges = E.shape[0]
    density_rho = density(A)
    d_max = np.max((A + A.T).getnnz(1))
    d_avg = np.mean((A + A.T).getnnz(1))
    diam = -1

    if True:
        sec_diam = tic()

        G_nx.remove_edges_from(nx.selfloop_edges(G_nx))

        print('finished reading Gnx')

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

        d = (A + A.T).getnnz(1)

        tri_idx = -1
        four_clq_idx = -1
        for o_idx, orbit_name in enumerate(edge_orbit_names):
            if 'triangles (K_3)' in orbit_name: tri_idx = o_idx
            if '4-clique (K_4)' in orbit_name: four_clq_idx = o_idx

        T_G = 0
        kappa = 0
        if tri_idx != -1:
            T_G = ((A_k[tri_idx].sum() / 2.0) / 3.0)

            W_G = (np.sum((d * (d - 1.))) / 2.0)
            kappa = (T_G / W_G) * 3.0
            T_max = np.max(A_k[tri_idx])
            [src, dst, tri] = sps.find(A_k[tri_idx])
            T_avg = np.sum(tri) / tri.shape[0] * 1.0

        K4_G = 0
        K4_max = 0
        K4_avg = 0
        if four_clq_idx != -1:
            K4_G = A_k[four_clq_idx].sum()
            K4_max = np.max(A_k[four_clq_idx])
            [src, dst, four_clq] = sps.find(A_k[four_clq_idx])
            K4_avg = np.sum(four_clq) / four_clq.shape[0] * 1.0

        sec_diam = toc(sec_diam)
        print("read and computed graph invariants=" + str(diam) + ", TIME:" + str(sec_diam))

        graphlet_col_names = '3-stars  & triangles & '
        graphlet_col_names += '4-path  &  4-star & 4-cycle &  '
        graphlet_col_names += 'tailed-tri  &  '
        graphlet_col_names += 'chordal-cycle &   4-clique  &  '

        graphlet_to_count_dist = [
            False, False,
            True, False,
            True,
            True,
            True, False, False,
            False, True,
            True
        ]

        graphlet_freq_dist = []
        for i, orbit_sum in enumerate(sum_orbit_edge_freq):
            orbit_sum = orbit_weight_norm[i] * orbit_sum
            if graphlet_to_count_dist[i]:
                graphlet_freq_dist.append(orbit_sum)

        graphlet_freq_dist = np.array(graphlet_freq_dist)
        total_graphlet_count = np.sum(graphlet_freq_dist)
        print('graphlet freq: ', graphlet_freq_dist)

        graphlet_freq_dist_norm = graphlet_freq_dist / (total_graphlet_count * 1.)
        print('graphlet freq dist: ', graphlet_freq_dist_norm)  # shape=(6,)

        """
        Add graph invariant stats (as well as total orbit counts)
        """
        graph_meta_features_simple = [num_nodes, num_edges, density_rho, d_max, d_avg, r, kappa, T_G, T_max, T_avg,
                                      K4_G, K4_max, K4_avg, max_kcore, med_kcore, mean_kcore]  # shape=(16,)

        # median degree, pagerank (min, max, median)
        graph_meta_features_simple = [num_nodes, num_edges, density_rho, d_max, d_avg, r,
                                      max_kcore, med_kcore, mean_kcore,

                                      kappa,
                                      T_G, T_max, T_avg,
                                      K4_G, K4_max, K4_avg, ]  # shape=(16,)

        graph_meta_features_vec = [num_nodes, num_edges, density_rho, d_max, d_avg, r, kappa, T_G, T_max, T_avg,
                                   K4_G, K4_max, K4_avg, max_kcore, med_kcore, mean_kcore]
        graph_meta_features_vec.extend(graphlet_freq_dist_norm)
        graph_meta_features_vec.extend(med_orbit_edge_freq)
        graph_meta_features_vec.extend(mean_orbit_edge_freq)
        graph_meta_features_vec.extend(max_orbit_edge_freq)
        graph_meta_features_vec = np.array(graph_meta_features_vec)

        return graph_meta_features_vec, np.array(graph_meta_features_simple), np.array(graphlet_freq_dist_norm)


def density(A):
    n = A.shape[0]
    return A.nnz / (n * (n - 1.0))


if __name__ == '__main__':
    import settings
    from graphs.netrepo_graphs.netrepo_unlabeled_graphs import load_graphs

    graphs = load_graphs()
    graph = graphs[20]
    print(graph, graph.pyg_graph(), "num_nodes:", graph.num_nodes)

    meta_feat_vec, meta_feat_vec_simple, meta_feat_vec_graphlet_dist = compact_meta_graph_features(
        graph.pyg_graph(),
        glet_input_graph_fn=settings.META_FEAT_ROOT / graph.name / "glet_input_graph.csv",
        output_orbits_fn=settings.META_FEAT_ROOT / graph.name / "glet_output_orbits.csv"
    )
    print("meta_feat_vec:", meta_feat_vec.shape)
    print("meta_feat_vec_simple:", meta_feat_vec_simple.shape)
    print("meta_feat_vec_graphlet_dist:", meta_feat_vec_graphlet_dist.shape)
