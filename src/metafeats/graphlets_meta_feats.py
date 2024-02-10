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


def graphlets_meta_graph_features(data: Data, glet_input_graph_fn, output_orbits_fn):  # shape=(756,)
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
        meta-graph feature vector for input graph (E)
    """
    if not GLET_PATH.exists():
        raise Exception(f"glet binary does not exist at {GLET_PATH}. "
                        f"Build glet binary first before calling this function.")

    # transforms the graph into an undirected graph, and output it to glet_input_graph_fn
    G_nx = to_networkx(data).to_undirected()
    A = nx.to_scipy_sparse_array(G_nx, format="csc")
    E_ub = np.argwhere(sps.triu(A + A.T) > 0)  # shape=(# undirected edges, 2)

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
            print('glet file size: ', os.stat(output_orbits_fn).st_size)
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

    delim = ','
    E_orbits = np.loadtxt(output_orbits_fn, comments=('%', '#'), delimiter=delim, skiprows=0)
    E_orbits = np.array(E_orbits, dtype=int)
    print(E_orbits)
    print(E_orbits.shape)

    E_orbits_i = E_orbits[:, 0]
    E_orbits_j = E_orbits[:, 1]
    E_orbits = E_orbits[:, 2:]

    n = A.shape[0]

    orbit_col_names = '2-stars ($P_3$) & triangles ($K_3$) & 4-path-edge ($P_4$)  &  4-path-center ($P_4$)  &  3-star (claw) & 4-cycle ($C_4$)  &  '
    orbit_col_names += 'tailed-tri-tailEdge (paw-tailEdge) &  tailed-tri-edge (paw-edge)  &  tailed-tri-center (paw-center)  &  '
    orbit_col_names += 'chordal-cycle-edge (diamond-edge) &  chordal-cycle-center (diamond-center)  &  4-clique ($K_4$)  &  '

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

    return np.array(graphlet_meta_features_vec)


if __name__ == '__main__':
    import settings
    from graphs.netrepo_graphs.netrepo_unlabeled_graphs import load_graphs

    graphs = load_graphs()
    graph = graphs[20]
    print(graph, graph.pyg_graph(), "num_nodes:", graph.num_nodes)

    feats = graphlets_meta_graph_features(
        graph.pyg_graph(),
        glet_input_graph_fn=settings.META_FEAT_ROOT / graph.name / "glet_input_graph.csv",
        output_orbits_fn=settings.META_FEAT_ROOT / graph.name / "glet_output_orbits.csv"
    )
    print("feats:", feats.shape)
