# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Graphs from PyTorch Geometric"""
import numpy as np
import torch

import settings
from graphs.graphdomain import GraphDomain
from graphs.graphset import Graph, print_graph_stats

PYG_DATA_ROOT = settings.GRAPH_DATA_ROOT / "pyg"


def load_graphs():
    from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Flickr, IMDB, DBLP, \
        EllipticBitcoinDataset, WikiCS, AttributedGraphDataset, NELL, WebKB, WikipediaNetwork, \
        HeterophilousGraphDataset, Actor, MixHopSyntheticDataset, GitHub, FacebookPagePage, LastFMAsia, \
        DeezerEurope, Twitch, Airports, PolBlogs, EmailEUCore, LINKXDataset, Entities, KarateClub, GemsecDeezer, \
        RandomPartitionGraphDataset, Yelp, AmazonProducts, StochasticBlockModelDataset, Reddit2

    graphs = [
        Graph(name="CitationFull-Cora", dataset=CitationFull(root=f'{PYG_DATA_ROOT}/CitationFull', name='cora'), domain=GraphDomain.CITATION),
        Graph(name="CitationFull-CoraML", dataset=CitationFull(root=f'{PYG_DATA_ROOT}/CitationFull', name='cora_ml'), domain=GraphDomain.CITATION),
        Graph(name="CitationFull-CiteSeer", dataset=CitationFull(root=f'{PYG_DATA_ROOT}/CitationFull', name='citeseer'), domain=GraphDomain.CITATION),
        Graph(name="CitationFull-DBLP", dataset=CitationFull(root=f'{PYG_DATA_ROOT}/CitationFull', name='dblp'), domain=GraphDomain.CITATION),
        Graph(name="CitationFull-PubMed", dataset=CitationFull(root=f'{PYG_DATA_ROOT}/CitationFull', name='pubmed'), domain=GraphDomain.CITATION),
        Graph(name="Coauthor-CS", dataset=Coauthor(root=f'{PYG_DATA_ROOT}/Coauthor', name='cs'), domain=GraphDomain.COAUTHOR),
        Graph(name="Coauthor-Physics", dataset=Coauthor(root=f'{PYG_DATA_ROOT}/Coauthor', name='physics'), domain=GraphDomain.COAUTHOR),
        Graph(name="Amazon-Computers", dataset=Amazon(root=f'{PYG_DATA_ROOT}/Amazon', name='computers'), domain=GraphDomain.COPURCHASE),
        Graph(name="Amazon-Photo", dataset=Amazon(root=f'{PYG_DATA_ROOT}/Amazon', name='photo'), domain=GraphDomain.COPURCHASE),
        Graph(name="Flickr", dataset=Flickr(root=f'{PYG_DATA_ROOT}/Flickr'), domain=GraphDomain.COMPUTERVISION),
        Graph(name="IMDB", dataset=IMDB(root=f'{PYG_DATA_ROOT}/IMDB'), domain=GraphDomain.KNOWLEDGEBASE),  # heterogeneous: node features are available, and of the same size.
        Graph(name="DBLP", dataset=DBLP(root=f'{PYG_DATA_ROOT}/DBLP'), domain=GraphDomain.KNOWLEDGEBASE),  # heterogeneous: node features are available, yet features for different node types have different sizes. no node features are available after transformed into a homogeneous graph.
        Graph(name="EllipticBitcoin", dataset=EllipticBitcoinDataset(root=f'{PYG_DATA_ROOT}/EllipticBitcoin'), domain=GraphDomain.ECONOMIC),
        Graph(name="NELL", dataset=NELL(root=f'{PYG_DATA_ROOT}/NELL'), domain=GraphDomain.KNOWLEDGEBASE),
        Graph(name="WikiCS", dataset=WikiCS(root=f'{PYG_DATA_ROOT}/WikiCS'), domain=GraphDomain.WIKIPEDIA),
        Graph(name="AttributedGraph-Wiki", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='wiki'), domain=GraphDomain.WIKIPEDIA),
        Graph(name="AttributedGraph-Cora", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='cora'), domain=GraphDomain.CITATION),
        Graph(name="AttributedGraph-CiteSeer", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='citeseer'), domain=GraphDomain.CITATION),
        Graph(name="AttributedGraph-BlogCatalog", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='blogcatalog'), domain=GraphDomain.SOCIAL),
        Graph(name="AttributedGraph-PPI", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='ppi'), domain=GraphDomain.PROTEIN),  # multi-label multi-class node classification
        Graph(name="AttributedGraph-Flickr", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='flickr'), domain=GraphDomain.SOCIAL),
        Graph(name="AttributedGraph-Facebook", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='facebook'), domain=GraphDomain.SOCIAL),  # multi-label multi-class node classification
        Graph(name="WebKB-Cornell", dataset=WebKB(root=f'{PYG_DATA_ROOT}/WebKB', name='Cornell'), domain=GraphDomain.WEB),
        Graph(name="WebKB-Texas", dataset=WebKB(root=f'{PYG_DATA_ROOT}/WebKB', name='Texas'), domain=GraphDomain.WEB),
        Graph(name="WebKB-Wisconsin", dataset=WebKB(root=f'{PYG_DATA_ROOT}/WebKB', name='Wisconsin'), domain=GraphDomain.WEB),
        Graph(name="WikipediaNetwork-Chameleon", dataset=WikipediaNetwork(root=f'{PYG_DATA_ROOT}/WikipediaNetwork', name="chameleon"), domain=GraphDomain.WIKIPEDIA),
        Graph(name="WikipediaNetwork-Squirrel", dataset=WikipediaNetwork(root=f'{PYG_DATA_ROOT}/WikipediaNetwork', name="squirrel"), domain=GraphDomain.WIKIPEDIA),
        Graph(name="HeterophilousGraph-Roman-empire", dataset=HeterophilousGraphDataset(root=f'{PYG_DATA_ROOT}/HeterophilousGraphDataset', name="Roman-empire"), domain=GraphDomain.WIKIPEDIA),  # word dependency graph
        Graph(name="HeterophilousGraph-Amazon-ratings", dataset=HeterophilousGraphDataset(root=f'{PYG_DATA_ROOT}/HeterophilousGraphDataset', name="Amazon-ratings"), domain=GraphDomain.COPURCHASE),
        Graph(name="HeterophilousGraph-Minesweeper", dataset=HeterophilousGraphDataset(root=f'{PYG_DATA_ROOT}/HeterophilousGraphDataset', name="Minesweeper"), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="HeterophilousGraph-Tolokers", dataset=HeterophilousGraphDataset(root=f'{PYG_DATA_ROOT}/HeterophilousGraphDataset', name="Tolokers"), domain=GraphDomain.COLLABORATION),
        Graph(name="HeterophilousGraph-Questions", dataset=HeterophilousGraphDataset(root=f'{PYG_DATA_ROOT}/HeterophilousGraphDataset', name="Questions"), domain=GraphDomain.INTERACTION),
        Graph(name="LINKX-penn94", dataset=LINKXDataset(root=f'{PYG_DATA_ROOT}/LINKXDataset', name='penn94'), domain=GraphDomain.SOCIAL),  # heterophilous graph
        Graph(name="LINKX-reed98", dataset=LINKXDataset(root=f'{PYG_DATA_ROOT}/LINKXDataset', name='reed98'), domain=GraphDomain.FRIENDSHIP),  # heterophilous graph
        Graph(name="LINKX-amherst41", dataset=LINKXDataset(root=f'{PYG_DATA_ROOT}/LINKXDataset', name='amherst41'), domain=GraphDomain.FRIENDSHIP),  # heterophilous graph
        Graph(name="LINKX-cornell5", dataset=LINKXDataset(root=f'{PYG_DATA_ROOT}/LINKXDataset', name='cornell5'), domain=GraphDomain.FRIENDSHIP),  # heterophilous graph
        Graph(name="LINKX-johnshopkins55", dataset=LINKXDataset(root=f'{PYG_DATA_ROOT}/LINKXDataset', name='johnshopkins55'), domain=GraphDomain.SOCIAL),  # heterophilous graph
        Graph(name="LINKX-genius", dataset=LINKXDataset(root=f'{PYG_DATA_ROOT}/LINKXDataset', name='genius'), domain=GraphDomain.SOCIAL),  # heterophilous graph
        Graph(name="Actor", dataset=Actor(root=f'{PYG_DATA_ROOT}/Actor'), domain=GraphDomain.WIKIPEDIA),
        Graph(name="MixHopSynthetic-Homophily-0.0", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.0), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.1", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.1), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.2", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.2), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.3", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.3), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.4", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.4), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.5", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.5), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.6", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.6), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.7", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.7), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.8", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.8), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="MixHopSynthetic-Homophily-0.9", dataset=MixHopSyntheticDataset(root=f'{PYG_DATA_ROOT}/MixHopSyntheticDataset', homophily=0.9), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(name="GitHub", dataset=GitHub(root=f'{PYG_DATA_ROOT}/GitHub'), domain=GraphDomain.FRIENDSHIP),
        Graph(name="FacebookPagePage", dataset=FacebookPagePage(root=f'{PYG_DATA_ROOT}/FacebookPagePage'), domain=GraphDomain.WEB),
        Graph(name="LastFMAsia", dataset=LastFMAsia(root=f'{PYG_DATA_ROOT}/LastFMAsia'), domain=GraphDomain.FRIENDSHIP),
        Graph(name="DeezerEurope", dataset=DeezerEurope(root=f'{PYG_DATA_ROOT}/DeezerEurope'), domain=GraphDomain.FRIENDSHIP),
        Graph(name="Twitch-DE", dataset=Twitch(root=f'{PYG_DATA_ROOT}/Twitch', name="DE"), domain=GraphDomain.FRIENDSHIP),
        Graph(name="Twitch-EN", dataset=Twitch(root=f'{PYG_DATA_ROOT}/Twitch', name="EN"), domain=GraphDomain.FRIENDSHIP),
        Graph(name="Twitch-ES", dataset=Twitch(root=f'{PYG_DATA_ROOT}/Twitch', name="ES"), domain=GraphDomain.FRIENDSHIP),
        Graph(name="Twitch-FR", dataset=Twitch(root=f'{PYG_DATA_ROOT}/Twitch', name="FR"), domain=GraphDomain.FRIENDSHIP),
        Graph(name="Twitch-PT", dataset=Twitch(root=f'{PYG_DATA_ROOT}/Twitch', name="PT"), domain=GraphDomain.FRIENDSHIP),
        Graph(name="Twitch-RU", dataset=Twitch(root=f'{PYG_DATA_ROOT}/Twitch', name="RU"), domain=GraphDomain.FRIENDSHIP),
        Graph(name="Airports-USA", dataset=Airports(root=f'{PYG_DATA_ROOT}/Airports', name="USA"), domain=GraphDomain.FLIGHT),
        Graph(name="Airports-Brazil", dataset=Airports(root=f'{PYG_DATA_ROOT}/Airports', name="Brazil"), domain=GraphDomain.FLIGHT),
        Graph(name="Airports-Europe", dataset=Airports(root=f'{PYG_DATA_ROOT}/Airports', name="Europe"), domain=GraphDomain.FLIGHT),
        Graph(name="PolBlogs", dataset=PolBlogs(root=f'{PYG_DATA_ROOT}/PolBlogs'), domain=GraphDomain.WEB),
        Graph(name="EmailEUCore", dataset=EmailEUCore(root=f'{PYG_DATA_ROOT}/EmailEUCore'), domain=GraphDomain.EMAIL),
        Graph(name="Entities-AIFB", dataset=Entities(root=f'{PYG_DATA_ROOT}/Entities', name="AIFB"), domain=GraphDomain.KNOWLEDGEBASE),  # labels only for a subset of nodes. no features.
        Graph(name="Entities-MUTAG", dataset=Entities(root=f'{PYG_DATA_ROOT}/Entities', name="MUTAG"), domain=GraphDomain.KNOWLEDGEBASE),  # labels only for a subset of nodes. no features.
        Graph(name="Entities-BGS", dataset=Entities(root=f'{PYG_DATA_ROOT}/Entities', name="BGS"), domain=GraphDomain.KNOWLEDGEBASE),  # labels only for a subset of nodes. no features.
        Graph(name="KarateClub", dataset=KarateClub(), domain=GraphDomain.FRIENDSHIP),
        Graph(name="GemsecDeezer-HU", dataset=GemsecDeezer(root=f'{PYG_DATA_ROOT}/GemsecDeezer', name="HU"), domain=GraphDomain.FRIENDSHIP),  # for multi-label multi-class node classification. no features.
        Graph(name="GemsecDeezer-HR", dataset=GemsecDeezer(root=f'{PYG_DATA_ROOT}/GemsecDeezer', name="HR"), domain=GraphDomain.FRIENDSHIP),  # for multi-label multi-class node classification. no features.
        Graph(name="GemsecDeezer-RO", dataset=GemsecDeezer(root=f'{PYG_DATA_ROOT}/GemsecDeezer', name="RO"), domain=GraphDomain.FRIENDSHIP),  # for multi-label multi-class node classification. no features.
    ]
    # Graph(name="AmazonProducts", dataset=AmazonProducts(root=f'{PYG_DATA_ROOT}/AmazonProducts'), domain=GraphDomain.COPURCHASE),  # multi-label multi-class node classification. too large (x=[1569960, 200], edge_index=[2, 264339468])
    # Graph(name="AttributedGraph-Twitter", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='twitter'), domain=GraphDomain.NONE),  # multi-label multi-class node classification. too large (x=[81306, 216839, nnz=94616433], edge_index=[2, 2420766])
    # Graph(name="AttributedGraph-TWeibo", dataset=AttributedGraphDataset(root=f'{PYG_DATA_ROOT}/AttributedGraphDataset', name='tweibo'), domain=GraphDomain.NONE),  # too large: x=[2320895, 1657], edge_index=[2, 50655143]
    # Graph(name="Yelp", dataset=Yelp(root=f'{PYG_DATA_ROOT}/Yelp'), domain=GraphDomain.FRIENDSHIP),  # multi-label multi-class node classification. too large (x=[716847, 300], edge_index=[2, 13954819], y=[716847, 100])
    # Graph(name="Reddit", dataset=Reddit2(root=f'{PYG_DATA_ROOT}/Reddit2'), domain=GraphDomain.WEB),  #  too large: [232965, 602], edge_index=[2, 23213838], y=[232965]
    # Graph(name="MovieLens", dataset=MovieLens(root=f'{PYG_DATA_ROOT}/MovieLens'), domain=GraphDomain.MOVIE),  # heterogeneous. only edge labels are given.
    # Graph(name="Entities-AM", dataset=Entities(root=f'{PYG_DATA_ROOT}/Entities', name="AM"), domain=GraphDomain.ARTIFACT),  # too large (~1.7M nodes, ~12M edges). no features
    # Graph(name="AMiner", dataset=AMiner(root=f'{PYG_DATA_ROOT}/AMiner'), domain=GraphDomain.ACADEMIC),  # heterogeneous. too large (~5M nodes).
    # Graph(name="Taobao", dataset=Taobao(root=f'{PYG_DATA_ROOT}/Taobao'), domain=GraphDomain.NONE),  # no label. too large (~5M nodes).

    graphs += load_pyg_sbm_graphs()
    graphs += load_pyg_random_partition_graphs()

    return graphs


def load_pyg_sbm_graphs():
    from torch_geometric.datasets import StochasticBlockModelDataset

    sbm_graphs = []
    block_sizes = [150, 200, 300, 350]
    for i in np.linspace(0.5, 3.0, num=6):
        diag = np.array([0.10, 0.11, 0.12, 0.13])
        edge_probs = np.array([[0.10, 0.05, 0.02, 0.03],
                               [0.05, 0.11, 0.06, 0.03],
                               [0.02, 0.06, 0.12, 0.02],
                               [0.03, 0.03, 0.02, 0.13]])
        edge_probs[range(4), range(4)] = diag * i
        g = Graph(name=f"StochasticBlockModel-{i:.1f}",
                  dataset=StochasticBlockModelDataset(root=f'{PYG_DATA_ROOT}/StochasticBlockModelDataset',
                                                      block_sizes=block_sizes,
                                                      edge_probs=torch.from_numpy(edge_probs),
                                                      num_channels=64),
                  domain=GraphDomain.SYNTHETIC_SBM)
        sbm_graphs.append(g)
    return sbm_graphs


def load_pyg_random_partition_graphs():
    from torch_geometric.datasets import RandomPartitionGraphDataset

    rp_graphs = []
    for node_homophily_ratio in [0.1, 0.3, 0.5, 0.7]:
        for average_degree in [5, 10, 15]:
            g = Graph(name=f"RandomPartitionGraph-hr{node_homophily_ratio:.1f}-ad{average_degree}",
                      dataset=RandomPartitionGraphDataset(root=f'{PYG_DATA_ROOT}/RandomPartitionGraphDataset',
                                                          num_classes=10,
                                                          num_nodes_per_class=500,
                                                          node_homophily_ratio=node_homophily_ratio,
                                                          average_degree=average_degree,
                                                          num_channels=64),
                      domain=GraphDomain.SYNTHETIC_RP)
            rp_graphs.append(g)
    return rp_graphs


if __name__ == '__main__':
    print_graph_stats(load_graphs())
