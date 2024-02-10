# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Graphs from Network Repository"""
import os.path as osp
import shutil
from pathlib import Path
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

import settings
from graphs.graphdomain import GraphDomain
from graphs.graphset import Graph, print_graph_stats

UNLABELED_DATA_ROOT = settings.GRAPH_DATA_ROOT / "netrepo" / "Unlabeled"
UNLABELED_GDRIVE_ROOT = settings.GRAPH_DATA_ROOT / "netrepo" / "GDrive" / "GDriveUnlabeled"
UNLABELED_GRAPH_SUMMARY_FILE = UNLABELED_GDRIVE_ROOT / "graph_summary.pkl"


class NetRepoUnlabeledDataset(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_unlabeled_netrepo_graphs(self.root)

    @property
    def gdrive_data_dir(self) -> str:
        return osp.join(self.root, 'gdrive')

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        # A list of files in the raw_dir which needs to be found in order to skip the download.
        return [self.edge_list_file_name]

    @property
    def edge_list_file_name(self) -> str:
        return f'{self.name}.edges'

    def process(self):
        """edges"""
        edge_list_file_path = Path(osp.join(self.raw_dir, self.edge_list_file_name))
        edges = load_edges(edge_list_file_path)
        print(f"edges: {edges.shape} (file path={edge_list_file_path})\n{edges}")
        assert edges.ndim == 2 and edges.shape[1] == 2, (edges.ndim, edges.shape)

        nx_G = nx.from_edgelist(edges)  # undirected networkx graph
        nx_G = nx.convert_node_labels_to_integers(nx_G, first_label=0,
                                                  ordering="sorted")  # make node ids to be consecutive, and start from 0
        print(f"{edge_list_file_path.name}: # nodes={nx_G.number_of_nodes()}, # edges={nx_G.number_of_edges()}, "
              f"is directed={nx_G.is_directed()}")

        edges = np.array(nx.to_directed(nx_G).edges())  # shape=(# edges, 2)
        print("edges:", edges, edges.dtype)
        edge_index = torch.from_numpy(edges).t().contiguous()

        data = Data(x=None, edge_index=edge_index, y=None)
        data = data if self.pre_filter is None else self.pre_filter(data)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


def download_unlabeled_netrepo_graphs(root):
    """download all unlabeled netrepo graphs"""
    unlabeled_data_gdrive_url = 'https://drive.usercontent.google.com/download?id={}&confirm=t'
    unlabeled_data_gdrive_url = unlabeled_data_gdrive_url.format('1CoSj51rDU238nJyH5cThiOlmqhkWqJ9m')  # netrepo-unlabeled-graphs.zip
    download_path = Path(download_url(unlabeled_data_gdrive_url, UNLABELED_GDRIVE_ROOT))
    print("download_path:", download_path)
    extract_zip(download_path, UNLABELED_GDRIVE_ROOT)
    download_path.unlink()
    shutil.rmtree(download_path.parent / "__MACOSX", ignore_errors=True)

    unlabeled_gdrive_root = UNLABELED_GDRIVE_ROOT
    assert unlabeled_gdrive_root.exists() and unlabeled_gdrive_root.is_dir(), unlabeled_gdrive_root

    graph_summary = []
    for domain_dir in unlabeled_gdrive_root.iterdir():
        if not domain_dir.is_dir():
            print(f"ignore non directory: {domain_dir}")
            continue
        if domain_dir.name.startswith("__"):  # e.g., __MACOSX
            print(f"ignore non data folder: {domain_dir}")
            continue

        print("=" * 80)
        print("domain:", domain_dir.name)
        print("=" * 80)
        for domain_edge_file in domain_dir.iterdir():
            # print(domain_dir.name, domain_edge_file)
            if domain_edge_file.name.startswith("."):
                print(f"ignore {domain_edge_file}")
                continue

            if domain_edge_file.name.endswith(".edges") or domain_edge_file.name.endswith(".csv"):
                edge_file_dst = Path(root) / domain_edge_file.stem / 'raw' / f"{domain_edge_file.stem}.edges"
                edge_file_dst.parent.mkdir(parents=True, exist_ok=True)

                shutil.copyfile(src=domain_edge_file, dst=edge_file_dst)
            elif domain_edge_file.name.endswith(".mtx"):
                mtx_edges = load_edges(domain_edge_file)

                if domain_edge_file.stem in ['scc_rt_onedirection', 'ca-sandi-auths']:
                    edgelist = mtx_edges
                else:
                    if domain_edge_file.stem not in ['econ-beacxc', 'econ-beaflw', 'econ-wm3', 'econ-beause']:
                        assert mtx_edges[0][0] == mtx_edges[0][1], (mtx_edges[0], domain_edge_file)
                    # discarding the 1st row of mtx file, as it is meta data about the number of rows, columns, and nonzeros.
                    edgelist = mtx_edges[1:, :]

                edge_file_dst = Path(root) / domain_edge_file.stem / 'raw' / f"{domain_edge_file.stem}.edges"
                edge_file_dst.parent.mkdir(parents=True, exist_ok=True)

                np.savetxt(edge_file_dst, edgelist, fmt='%i')
            else:
                raise ValueError(f"Invalid file extension: {domain_edge_file}")

            edges = load_edges(edge_file_dst)
            graph_summary.append({
                'graph_name': domain_edge_file.stem,
                'num_nodes': len(np.unique(edges.reshape(-1))),
                'num_edges': len(edges),
                'graph_domain': domain_dir.name,
            })
            print(f"{edge_file_dst.name}: # nodes={graph_summary[-1]['num_nodes']}, # edges={graph_summary[-1]['num_edges']}")

    df = pd.DataFrame(graph_summary)
    df.to_pickle(UNLABELED_GRAPH_SUMMARY_FILE)

    if False:
        shutil.rmtree(unlabeled_gdrive_root)  # remove unpacked files


def load_edges(edgelist_file):
    try:
        try:
            E = np.loadtxt(edgelist_file, comments=('%', '#'), skiprows=0, usecols=(0, 1), dtype=int)
        except ValueError:
            E = np.loadtxt(edgelist_file, comments=('%', '#'), skiprows=0, usecols=(0, 1), dtype=int, delimiter=',')
    except Exception:
        print(f"error in loading {edgelist_file}")
        raise
    return E


def load_graph_summary():
    return pd.read_pickle(UNLABELED_GRAPH_SUMMARY_FILE)


def get_graphs_to_load(graph_domains=None,
                       num_nodes_min=50, num_nodes_max=1000000,
                       num_edges_min=200, num_edges_max=2500000,
                       max_graphs_per_domain=25,
                       verbose=False):
    def pr(df):
        print(df)
        print(df.groupby(['graph_domain']).agg({'num_nodes': ['min', 'max', 'count'],
                                                'num_edges': ['min', 'max', 'count']}))
        print("* num nodes range between", df['num_nodes'].min(), "and", df['num_nodes'].max())
        print("* num edges range between", df['num_edges'].min(), "and", df['num_edges'].max())

    if graph_domains is None:
        graph_domains = [
            'power', 'sc', 'bio', 'web', 'soc',  'socfb',
            'tscc', 'econ', 'ca', 'protein', 'ia', 'road',
            'rt', 'eco', 'inf', 'rec', 'proximity',
            'bn', 'chem', 'tech', 'cit', 'email',  # 'asn' (too small),
            'syn-BA', 'syn-KPGM', 'syn-CL', 'syn-ER',  # 'syn-PLC', 'syn-SW',
        ]

    df = load_graph_summary()
    if verbose:
        print("\nAVAILABLE GRAPHS:")
        pr(df)

    df = df.loc[df['graph_domain'].isin(graph_domains)]
    if verbose:
        print("\nFILTER BY DOMAIN")
        pr(df)

    df = df[df['num_nodes'].between(num_nodes_min, num_nodes_max)]
    if verbose:
        print("\nFILTER BY NUM NODES")
        pr(df)

    df = df[df['num_edges'].between(num_edges_min, num_edges_max)]
    if verbose:
        print("\nFILTER BY NUM EDGES")
        pr(df)

    from graphs.netrepo_graphs.netrepo_labeled_graphs import load_graphs as load_labeled_graphs
    labeld_graph_names = [g.name.lower() for g in load_labeled_graphs()]
    df = df[~df['graph_name'].str.lower().isin(labeld_graph_names)]
    if verbose:
        print("\nFILTER OUT GRAPHS WITH LABELS AND SOME LARGE GRAPHS")
        print(df)

    domain_df_list = []
    for domain in graph_domains:
        domain_df = df.loc[df['graph_domain'] == domain]

        if domain.startswith('syn'):
            num_limit = 15
        else:
            num_limit = max_graphs_per_domain

        if len(domain_df) > num_limit:
            # domain_df = domain_df.head(num_limit)
            indices = np.unique(np.linspace(0, len(domain_df) - 1, num_limit).astype(int))
            assert len(indices) == num_limit, (len(indices), num_limit)
            domain_df = domain_df.sort_values('num_nodes').iloc[indices]

        domain_df_list.append(domain_df)
    df = pd.concat(domain_df_list)
    if verbose:
        print("\nFILTER BY MAX NUM GRAPHS PER DOMAIN")
        pr(df)

    # filter out duplicates (tech-routers-rf, tech-internet-as, and tech-WHOIS are duplicated somehow)
    if verbose:
        print(f"\nFILTER OUT {df.duplicated(keep='first').sum()} DUPLICATES: "
              f"{df[df.duplicated(keep='first')]['graph_name'].to_list()}")
        pr(df[~df.duplicated(keep='first')])
    df = df[~df.duplicated(keep='first')]

    return df


def load_graphs(domain_mapping=None):
    if domain_mapping is None:
        domain_mapping = {
            'power': GraphDomain.POWER,
            'sc': GraphDomain.SC,
            'bio': GraphDomain.BIO,
            'web': GraphDomain.WEB,
            'soc': GraphDomain.SOCIAL,
            'socfb': GraphDomain.SOCIAL,
            'tscc': GraphDomain.TEMPREACH,
            'econ': GraphDomain.ECONOMIC,
            'ca': GraphDomain.COLLABORATION,
            'protein': GraphDomain.PROTEIN,
            'ia': GraphDomain.INTERACTION,
            'road': GraphDomain.ROAD,
            'rt': GraphDomain.RETWEET,
            'eco': GraphDomain.ECOLOGY,
            'inf': GraphDomain.INFRA,
            'rec': GraphDomain.RECOMMENDATION,
            'proximity': GraphDomain.PROXIMITY,
            'bn': GraphDomain.BRAIN,
            'chem': GraphDomain.CHEMICAL,
            'tech': GraphDomain.TECHNOLOGY,
            'cit': GraphDomain.CITATION,
            'email': GraphDomain.EMAIL,
            'syn-BA': GraphDomain.SYNTHETIC_BA,
            'syn-KPGM': GraphDomain.SYNTHETIC_KPGM,
            'syn-CL': GraphDomain.SYNTHETIC_CL,
            'syn-ER': GraphDomain.SYNTHETIC_ER,
        }

    load_df = get_graphs_to_load()
    graphs = []
    for i, row in load_df.iterrows():
        graphs.append(Graph(dataset=NetRepoUnlabeledDataset(root=f'{UNLABELED_DATA_ROOT}', name=row['graph_name']),
                            domain=domain_mapping[row['graph_domain']]))
    return graphs


if __name__ == '__main__':
    # download_unlabeled_netrepo_graphs(root=UNLABELED_DATA_ROOT)
    # get_graphs_to_load(verbose=True)
    print_graph_stats(load_graphs())
