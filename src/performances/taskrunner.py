# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import errno
import fcntl
import json
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pprint
import time
from datetime import datetime
from enum import Enum, unique
from logging import Logger
from pathlib import Path
from typing import Optional, Union, List

import torch
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data
from torch_geometric.nn import LabelPropagation

import settings
from graphs.datasplit import DataSplit
from graphs.graphset import GraphSet, Graph
from graphs.nodefeatures import RandomNodeFeatures, POOL_NUM_NODES
from models.linkpred_models import ClassicalLinkPredictionModel, DGCNN
from models.modelset import ModelSet
from models.trainer import Trainer, PNAPartial, DGCNNPartial
from models.unsupervised_models import SpectralEmbedding, GraRep, DGI, Node2VecModel
from utils import setup_logger, set_seed, setup_cuda
from utils.log_utils import clear_log_handlers


@unique
class Task(Enum):
    NODE_CLASSIFICATION = "NodeClassification"
    LINK_PREDICTION = "LinkPrediction"


class Runner:
    LOCK_FILE = "running.flock"
    PERF_FILE = "perf_dict.json"

    def __init__(self,
                 task: Task,
                 root: Path,
                 args: argparse.Namespace,
                 graphset: Optional[Graph] = None,
                 modelset: Optional[ModelSet] = None,
                 logger=None):
        self.task = task
        self.root = root
        self.args = args
        self.run_time = datetime.today().strftime('%y%m%d_%H%M%S')
        if logger is None:
            self.logger = setup_logger("runner_logger",
                                       log_file=self.root / f"runner_log_{self.run_time}.txt",
                                       log_format='long')
        else:
            self.logger = logger
        self.graphset = GraphSet(data_sources=['netrepo', 'pyg']) if graphset is None else graphset
        self.graphs = {
            Task.LINK_PREDICTION: self.graphset.link_prediction_graphs(),
            Task.NODE_CLASSIFICATION: self.graphset.node_classification_graphs(),
        }[task]
        self.modelset = ModelSet() if modelset is None else modelset
        self.random_node_features32: RandomNodeFeatures = RandomNodeFeatures(pool_num_nodes=POOL_NUM_NODES, num_feats=32)
        self.random_node_features16: RandomNodeFeatures = RandomNodeFeatures(pool_num_nodes=POOL_NUM_NODES, num_feats=16)


    def __repr__(self):
        return f"{self.__class__.__name__}(task={self.task}, root={self.root})"

    def run(self, split_indices: Union[int, List[int]] = 0):
        self.logger.info("\n[Runner arguments]\n" + pprint.pformat(args.__dict__))
        if isinstance(split_indices, int):
            split_indices = [split_indices]
        self.logger.info(f"Starting to run with split_indices: {split_indices}")

        for split_i in split_indices:
            self.logger.info(f"\n\nStarting to run on graph data split_i={split_i}")

            for lock_info in self.exclusive_get_graph_and_model(split_i):
                run_logger = setup_logger("run_logger", log_file=lock_info['run_root'] / f"run_log.txt",
                                          log_format='short')
                self.run_task(split_i=split_i,
                              graph_i=lock_info['graph_i'],
                              model_i=lock_info['model_i'],
                              run_root=lock_info['run_root'],
                              run_logger=run_logger)
                clear_log_handlers(run_logger)
                self.lock_release(lock_file=lock_info['lock_file'])

    def run_task(self, split_i: int, graph_i: int, model_i: int, run_root: Path, run_logger: Logger):
        perf_dict_path = run_root / self.PERF_FILE
        if perf_dict_path.exists():
            self.logger.info(f"perf_dict already exists: {perf_dict_path}.")
            return

        # set seed before loading the model and running it
        set_seed(seed=split_i * 101)

        """Load graph"""
        graph: Data = self.load_graph(graph_i=graph_i, split_i=split_i)
        self.logger.info(f"\n\n*** graph_i={graph_i} ({graph.name}: {graph})")

        """Load model"""
        in_channels = graph.x.size(1) if graph.x is not None else graph.rand_x.size(1)
        if self.task is Task.NODE_CLASSIFICATION:
            out_channels = graph.num_classes
        elif self.task is Task.LINK_PREDICTION:
            out_channels = self.args.emb_dim
        else:
            raise ValueError(f"Undefined task: {self.task}")
        self.logger.info(f"*** model_i={model_i} ({self.modelset.get_model_setting_repr(model_i)})\n\n")

        model = self.modelset.load_model(model_i=model_i, in_channels=in_channels, out_channels=out_channels)
        if isinstance(model, (SpectralEmbedding, GraRep, DGI, Node2VecModel)):
            model.train_unsupervised_node_emb(graph, device=self.args.device, logger=run_logger, run_root=run_root)
        elif isinstance(model, PNAPartial):
            model = model.instantiate(graph, self.task)
        elif isinstance(model, DGCNNPartial):
            model = model.instantiate(graph)

        """Load trainer"""
        if self.task is Task.NODE_CLASSIFICATION and graph.y.ndim == 2:  # multi-label node classification tasks
            lr = self.args.lr_multi_label
            patience = self.args.patience_multi_label
        else:
            lr = self.args.lr
            patience = self.args.patience

        if self.task is Task.NODE_CLASSIFICATION:
            batch_size = self.args.node_batch_size
            # if graph.num_edges > 5000000:
            #     batch_size = 10000
        elif self.task is Task.LINK_PREDICTION:
            if isinstance(model, DGCNN):  # SEAL link prediction method
                batch_size = self.args.seal_batch_size
            else:
                batch_size = self.args.edge_batch_size
        else:
            raise ValueError(f"Undefined task: {self.task}")
        self.logger.info(f"batch_size={batch_size}")

        # Note: model and graph move to args.device when Trainer gets instantiated
        model_trainer = Trainer(model=model, data=graph, logger=run_logger,
                                lr=lr,
                                epochs=self.args.epochs,
                                patience=patience,
                                batch_size=batch_size,
                                device=self.args.device)

        """train/test"""
        self.logger.info(f"Starting to run model-{model_i} on graph-{graph_i} (split-i={split_i})...")
        if self.task is Task.NODE_CLASSIFICATION:
            if isinstance(model, LabelPropagation):
                perf_dict = model_trainer.train_test_node_classification_with_label_prop()
            else:
                perf_dict = model_trainer.train_test_node_classification()
        elif self.task is Task.LINK_PREDICTION:
            if isinstance(model, DGCNN):  # SEAL link prediction method
                perf_dict = model_trainer.train_test_link_prediction_with_seal()
            elif isinstance(model, ClassicalLinkPredictionModel):
                perf_dict = model_trainer.train_test_link_prediction_with_classical_models()
            else:
                perf_dict = model_trainer.train_test_link_prediction()
        else:
            raise ValueError(f"Undefined task: {self.task}")
        self.logger.info(f"Finished running model-{model_i} on graph-{graph_i} (split-i={split_i}).\n\n")

        # save performances
        json.dump(perf_dict, perf_dict_path.open('w'))
        self.logger.info(f"perf_dict saved to {perf_dict_path}.")
        assert perf_dict_path.exists()

        # time.sleep(2)
        assert perf_dict_path.exists()

        return perf_dict

    def load_graph(self, graph_i: int, split_i: int) -> Data:
        graph: Graph = self.graphs[graph_i]
        data_split = DataSplit(raw_graph=graph)
        pyg_graph = graph.pyg_graph()
        pyg_graph.name = graph.name

        """load learnable node features, if no input features are provided"""
        if graph.name in ["NELL", "LINKX-genius"]:  # use random features for NELL
            pyg_graph.x = None

        if graph.name in [  # use svd embeddings for these graphs
            "AttributedGraph-BlogCatalog",  # x=[5196, 8189]
            "Coauthor-Physics",  # x=[34493, 8415]
            "AttributedGraph-Flickr",  # x=[7575, 12047], sparse tensor
            "CitationFull-Cora",  # x=[19793, 8710]
            "LINKX-cornell5",  # x=[18660, 4735]
            "LINKX-penn94",  # x=[41554, 4814]
        ]:
            import torch_sparse
            if isinstance(pyg_graph.x, torch_sparse.SparseTensor):  # AttributedGraph-Flickr
                pyg_graph.x = pyg_graph.x.to_dense()

            n_components = 128
            # noinspection PyUnresolvedReferences
            if pyg_graph.x.shape[1] > n_components:
                svd = TruncatedSVD(n_components=n_components, algorithm="arpack", random_state=42)
                # noinspection PyUnresolvedReferences
                x_emb = svd.fit_transform(pyg_graph.x.detach().cpu().numpy())
                self.logger.info(f"reduced pyg_graph.x from {pyg_graph.x.shape} into {x_emb.shape} via svd.")
                pyg_graph.x = torch.from_numpy(x_emb)

        if pyg_graph.x is None:
            pyg_graph.rand_x = self.random_node_features32.load_node_features(graph)

        assert pyg_graph.x is not None or pyg_graph.rand_x is not None, pyg_graph

        if self.task is Task.NODE_CLASSIFICATION:
            if 'train_y' in pyg_graph and 'train_idx' in pyg_graph:  # pyg Entities graph
                node_y = torch.full((pyg_graph.num_nodes,), fill_value=-1).to(pyg_graph['train_y'].device)
                node_y[pyg_graph['train_idx']] = pyg_graph['train_y']
                node_y[pyg_graph['test_idx']] = pyg_graph['test_y']
                pyg_graph.y = node_y

            assert hasattr(pyg_graph, 'y') and pyg_graph.y is not None, pyg_graph  # assert node labels
            assert hasattr(pyg_graph, 'num_classes') and pyg_graph.num_classes is not None, pyg_graph  # assert num_classes

            """load node splits"""
            node_split = data_split.load_node_split(split_i)
            for data_key, node_index in node_split.items():

                mask_key = data_key.replace("_index", "_mask")  # e.g., train_node_index -> train_node_mask
                node_mask = torch.zeros(graph.num_nodes).bool()
                node_mask[node_index] = True

                pyg_graph[mask_key] = node_mask

        elif self.task is Task.LINK_PREDICTION:
            """load edge splits"""
            edge_split = data_split.load_edge_split(split_i)
            for data_key, data_value in edge_split.items():
                pyg_graph[data_key] = data_value
        else:
            raise ValueError(f"Undefined task: {self.task}")

        return pyg_graph

    def exclusive_get_graph_and_model(self, split_i: int):
        for graph_i, graph in enumerate(self.graphs):
            for model_i in range(len(self.modelset)):
                # run_root = self.root / f"split{split_i}" / graph.name / self.modelset.get_model_setting_repr(model_i)
                run_root = self.root / graph.name / self.modelset.get_model_setting_repr(model_i)
                if not run_root.exists():
                    run_root.mkdir(parents=True, exist_ok=True)

                lock_file = self.lock_acquire(lock_path=run_root)
                if lock_file is None:  # (model_i, graph_i) is being (or was already) processed. Go to the next pair.
                    continue

                yield {
                    'graph_i': graph_i,
                    'model_i': model_i,
                    'run_root': run_root,
                    'lock_file': lock_file,
                }

    def lock_acquire(self, lock_path: Path):
        """
        Acquire the flock lock_file (adapted from https://seds.nl/notes/locking-python-scripts-with-flock)
        """
        if (lock_path / self.PERF_FILE).exists():
            # self.logger.info(f"already processed: {lock_path}")
            return None

        lock_file = open(lock_path / self.LOCK_FILE, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except BlockingIOError:
            self.logger.info(f"locking failed (being processed by another process): {lock_file}")
            return None

    # noinspection PyMethodMayBeStatic
    def lock_release(self, lock_file):
        """
        Release and remove the flock lockfile (adapted from https://seds.nl/notes/locking-python-scripts-with-flock)
        """
        # release lock
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        try:
            os.remove(lock_file.name)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        default='link-pred',
                        choices=['node-class', 'link-pred'],
                        help="graph learning task")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set to -1 to use CPU.")
    parser.add_argument("--node-batch-size", type=int, default=100000,
                        help="number of nodes in one batch")
    parser.add_argument("--edge-batch-size", type=int, default=1000000,
                        help="number of edges in one batch")
    parser.add_argument("--seal-batch-size", type=int, default=256,
                        help="batch size for the SEAL method")
    parser.add_argument("--epochs", type=int, default=300,
                        help="maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--patience', type=int, default=30,
                        help='patience for early stopping (set this to negative value to not use early stopping)')
    parser.add_argument("--lr-multi-label", type=float, default=0.01,
                        help="learning rate for multi-label classification tasks")
    parser.add_argument('--patience-multi-label', type=int, default=60,
                        help='patience for early stopping for multi-label classification tasks (set this to negative value to not use early stopping)')
    parser.add_argument("--emb-dim", type=int, default=32,
                        help="size of node embeddings (i.e., dimension of output from GNN). "
                             "used only for link prediction task.")
    args = parser.parse_args()
    setup_cuda(args)

    root = {
        'node-class': settings.NODE_CLASS_PERF_ROOT,
        'link-pred': settings.LINK_PRED_PERF_ROOT,
    }[args.task]

    graphset = GraphSet(data_sources=['netrepo', 'pyg'], sort_graphs='num_edges')

    from models.modelset import node_classification_model_setting_groups, link_prediction_model_setting_groups
    modelset = {
        'node-class': ModelSet(node_classification_model_setting_groups),
        'link-pred': ModelSet(link_prediction_model_setting_groups),
    }[args.task]

    if args.task == 'node-class':
        runner = Runner(task=Task.NODE_CLASSIFICATION, root=root, args=args, graphset=graphset, modelset=modelset)
    elif args.task == 'link-pred':
        runner = Runner(task=Task.LINK_PREDICTION, root=root, args=args, graphset=graphset, modelset=modelset)
    else:
        raise ValueError(f"Undefined task: {args.task}")
    print(runner)
    runner.run()
