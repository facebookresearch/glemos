# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Graphs from Network Repository"""
import os.path as osp
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

import settings
from graphs.graphdomain import GraphDomain
from graphs.graphset import Graph, print_graph_stats
from utils import remap_labels

LABELED_DATA_ROOT = settings.GRAPH_DATA_ROOT / "netrepo" / "Labeled"


class NetRepoLabeledDataset(InMemoryDataset):
    url = 'https://nrvis.com/download/data/labeled/{}.zip'

    def __init__(self, root: str, name: str, min_node_id: int, delimiter=None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        if isinstance(delimiter, list):
            self.delimiter = delimiter
        else:
            self.delimiter = [delimiter]
        self.min_node_id = min_node_id
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = Path(download_url(self.url.format(self.name), self.raw_dir))
        extract_zip(path, self.raw_dir)
        path.unlink()
        for f in path.parent.glob("readme.html"):
            f.unlink()

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        # A list of files in the raw_dir which needs to be found in order to skip the download.
        return [self.edge_list_file_name, self.node_labels_file_name]

    @property
    def edge_list_file_name(self) -> str:
        return f'{self.name}.edges'

    @property
    def node_labels_file_name(self) -> str:
        return f'{self.name}.node_labels'

    @property
    def node_attrs_file_name(self) -> str:
        return f'{self.name}.node_attrs'

    def process(self):
        """edges"""
        edge_list_file_path = osp.join(self.raw_dir, self.edge_list_file_name)
        edges = None
        for delimiter in self.delimiter:
            try:
                edges = np.loadtxt(edge_list_file_path, delimiter=delimiter, dtype=int)
                break
            except Exception:
                pass
        print(f"edges: {edges.shape} (file path={edge_list_file_path})")
        if edges.shape[1] > 2:
            print(f"using only the first two columns from {edge_list_file_path}")
            edges = edges[:, :2]

        min_node_from_edges = np.min(edges)
        assert self.min_node_id <= min_node_from_edges, (self.min_node_id, min_node_from_edges)
        edges = edges - self.min_node_id  # make sure that node ids start from 0

        """node labels"""
        node_labels_file_path = osp.join(self.raw_dir, self.node_labels_file_name)
        node_labels = None
        for delimiter in self.delimiter:
            try:
                try:
                    node_labels = np.loadtxt(node_labels_file_path, delimiter=delimiter, dtype=int, usecols=(0, 1))
                except Exception:
                    node_labels = np.loadtxt(node_labels_file_path, delimiter=delimiter, dtype=int, usecols=(0,))
                break
            except Exception as e:
                print(e)
                pass
        print(f"node_labels: {node_labels.shape} (file path={node_labels_file_path})")
        assert node_labels.ndim <= 2, node_labels.ndim
        if node_labels.ndim == 2:  # first column contains node ids
            node_ids = node_labels[:, 0]
            unique_node_ids = np.unique(node_ids)
            """ensure that node ids are consecutive & start from self.min_node_id"""
            assert np.min(node_ids) == self.min_node_id, (np.min(node_ids), self.min_node_id)
            assert len(unique_node_ids) == (np.max(node_ids) - np.min(node_ids) + 1), f"node ids are not consecutive: {len(unique_node_ids)} != {np.max(node_ids) - np.min(node_ids) + 1}"

            if self.name in ['soc-BlogCatalog-ASU', 'soc-Flickr-ASU', 'soc-YouTube-ASU']:  # some nodes have more than one labels in these datasets
                _, _, remap_dict = remap_labels(node_labels[:, 1])

                multi_label_node_labels = np.zeros((np.max(node_ids) + 1 - self.min_node_id, len(remap_dict)), dtype=int)
                print("multi_label_node_labels:", multi_label_node_labels.shape)
                for node_id, node_label in node_labels:
                    multi_label_node_labels[node_id - self.min_node_id, remap_dict[node_label]] = 1
                print("# labels per node:", set([int(row.sum()) for row in multi_label_node_labels]))

                node_labels = multi_label_node_labels
            else:
                assert np.array_equal(node_ids, unique_node_ids), (node_ids.shape, unique_node_ids.shape)
                assert np.array_equal(node_ids, np.sort(node_ids)), "node ids are not sorted"

                node_labels = node_labels[:, 1]

        assert np.max(edges) <= len(node_labels) - 1, \
            ("max node id in edges larger than max node in node labels", np.max(edges), len(node_labels) - 1)

        if node_labels.ndim == 1:
            remapped_labels, remapped, remap_dict = remap_labels(node_labels)
            if remapped:
                print(f"node labels remapped: {remap_dict}")
                node_labels = remapped_labels

            assert np.min(node_labels) == 0, f"minimum node label is larger than zero: {np.min(node_labels)}"
            assert len(np.unique(node_labels)) == (np.max(node_labels) - np.min(node_labels) + 1), "node labels are not consecutive"

        """node attributes"""
        node_attrs_file_path = Path(osp.join(self.raw_dir, self.node_attrs_file_name))
        if node_attrs_file_path.exists():
            node_attrs = None
            for delimiter in self.delimiter:
                try:
                    node_attrs = np.loadtxt(node_attrs_file_path, delimiter=delimiter, dtype=float)
                    break
                except Exception:
                    pass
            print(f"node_attrs: {node_attrs.shape} (file path={node_attrs_file_path})")
        else:
            node_attrs = None
            print(f"{self.name} does not have node attribute file: {self.node_attrs_file_name}")

        if node_attrs is not None:
            assert np.max(edges) + 1 <= len(node_attrs), (np.max(edges) + 1, len(node_attrs))
            num_nodes = len(node_attrs)
        else:
            num_nodes = np.max(edges) + 1

        assert len(node_labels) >= num_nodes, (len(node_labels), num_nodes)
        if len(node_labels) > num_nodes:
            node_labels = node_labels[:num_nodes]
            print(f"retaining the first {num_nodes} rows for the node labels (remaining labels are for non-existent nodes)")

        edge_index = torch.from_numpy(edges).t().contiguous()
        y = torch.from_numpy(node_labels)
        if node_attrs is not None:
            assert len(node_labels) == len(node_attrs), (len(node_labels), len(node_attrs))
            x = torch.from_numpy(node_attrs)
        else:
            x = None

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_filter is None else self.pre_filter(data)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


def load_graphs():
    return [
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='SW-10000-6-0d3-L2', min_node_id=1), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='SW-10000-6-0d3-L5', min_node_id=1), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='BZR', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),   # domain=chemical
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='BZR-MD', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),   # domain=chemical
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='reality-call', min_node_id=0, delimiter=","), domain=GraphDomain.PHONE),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='internet-industry-partnerships', min_node_id=1, delimiter=","), domain=GraphDomain.COLLABORATION),   # domain=collaboration (industry partnerships)
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='soc-political-retweet', min_node_id=0, delimiter=","), domain=GraphDomain.RETWEET),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='webkb-wisc', min_node_id=1), domain=GraphDomain.WEB),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='fb-CMU-Carnegie49', min_node_id=1, delimiter=[",", " "]), domain=GraphDomain.SOCIAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='Tox21-p53', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='Tox21-aromatase', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='Tox21-AHR', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='Tox21-MMP', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='Tox21-HSE', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='TerroristRel', min_node_id=1, delimiter=","), domain=GraphDomain.COLLABORATION),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='PTC-MR', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='PTC-MM', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='PTC-FR', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='PTC-FM', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='PLC-40-30-L5', min_node_id=1), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='PLC-60-30-L2', min_node_id=1), domain=GraphDomain.SYNTHETIC_OTHERS),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='Mutagenicity', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='MSRC-9', min_node_id=1, delimiter=","), domain=GraphDomain.COMPUTERVISION),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='MSRC-21', min_node_id=1, delimiter=","), domain=GraphDomain.COMPUTERVISION),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='MSRC-21C', min_node_id=1, delimiter=","), domain=GraphDomain.COMPUTERVISION),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='ER-MD', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='DHFR', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='DHFR-MD', min_node_id=1, delimiter=","), domain=GraphDomain.CHEMICAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='CL-10K-1d8-L5', min_node_id=1), domain=GraphDomain.SYNTHETIC_CL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='BA-2_24_60-L2', min_node_id=1), domain=GraphDomain.SYNTHETIC_BA),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='BA-1_10_60-L5', min_node_id=1), domain=GraphDomain.SYNTHETIC_BA),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='soc-BlogCatalog-ASU', min_node_id=1, delimiter=","), domain=GraphDomain.SOCIAL),  # multi-label multi-class data
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='soc-Flickr-ASU', min_node_id=1, delimiter=","), domain=GraphDomain.SOCIAL),  # multi-label multi-class data
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='web-spam-detection', min_node_id=1, delimiter=","), domain=GraphDomain.WEB),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='AIDS', min_node_id=1, delimiter=","), domain=GraphDomain.BIO),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='Peking-1', min_node_id=1, delimiter=","), domain=GraphDomain.SOCIAL),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='DD6', min_node_id=1), domain=GraphDomain.MISC),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='DD21', min_node_id=1), domain=GraphDomain.MISC),
        Graph(dataset=NetRepoLabeledDataset(root=LABELED_DATA_ROOT, name='DD242', min_node_id=1), domain=GraphDomain.MISC),
    ]
# Graph(dataset=NetRepoLabeledDataset(root=f'{LABELED_DATA_ROOT}/Labeled', name='soc-YouTube-ASU', min_node_id=1, delimiter=","), domain=GraphDomain.NONE),  # node ids not consecutive. multi-label multi-class data
# Graph(dataset=NetRepoLabeledDataset(root=f'{LABELED_DATA_ROOT}/Labeled', name='CL-100K-1d8-L9', min_node_id=1), domain=GraphDomain.NONE),  # not all nodes have node labels


if __name__ == '__main__':
    print_graph_stats(load_graphs())

    new_labels, remapped, remap_dict = remap_labels(np.array([1, 1, 6, 3]))
    assert np.array_equal(new_labels, np.array([0, 0, 2, 1])) and remapped, (new_labels, remapped)
    new_labels, remapped, remap_dict = remap_labels(np.array([0, 0, 2, 1]))
    assert np.array_equal(new_labels, np.array([0, 0, 2, 1])) and not remapped, (new_labels, remapped)
