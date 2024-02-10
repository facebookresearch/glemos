# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from logging import Logger
from pprint import pformat
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNA, LabelPropagation
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import degree, to_networkx

from models.linkpred_models.classical_models import ClassicalLinkPredictionModel
from models.linkpred_models.seal import DGCNN, SEALDataset
from performances.link_prediction import link_prediction_performances
from performances.node_classification import node_classification_performances
from utils import EarlyStopping


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 data: Data,
                 logger: Logger,
                 lr: float = 0.001,
                 epochs: int = 300,
                 patience: int = 30,
                 batch_size: int = 100000,  # number of nodes or edges for mini-batch training
                 device: torch.device = torch.device('cpu')):
        # graph learning model
        if isinstance(model, nn.Module):
            self.model = model.to(device)
        else:
            self.model = model
        # graph data
        if isinstance(model, DGCNN):  # SEAL link prediction method
            self.data = data
        else:
            self.data = data.to(device)
        self.device = device
        self.logger = logger

        # parameters for training
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size

    def train_test_node_classification(self):
        model, data, logger = self.model, self.data, self.logger
        assert not isinstance(model, LabelPropagation), model

        """Training"""
        logger.info(f'Starting training')
        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")
        model.train()

        if data.x is not None:
            data_x = data.x.float()
            if isinstance(data_x, SparseTensor):  # gcn edge weight normalization does not work when given a sparse feature tensor
                data_x = data_x.to_dense()
            parameters = model.parameters()
        else:
            data_x = data.rand_x.float()
            data_x = torch.nn.Parameter(data_x, requires_grad=True)  # learnable node features
            parameters = list(model.parameters()) + [data_x]

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        if data.y.ndim == 1:  # single-label classification
            criterion = torch.nn.CrossEntropyLoss()
            data_y = data.y
        else:  # multi-label classification
            criterion = torch.nn.BCEWithLogitsLoss()
            data_y = data.y.float()

        logger.info(f"criterion: {criterion}")

        start = timer()
        stopper = EarlyStopping(self.patience, minimizing_objective=False, logger=logger) if self.patience >= 0 else None
        epoch = -1
        best_val_epoch = -1
        best_data_x = None
        node_perm = torch.randperm(data.train_node_mask.sum()).to(self.device)

        for epoch in range(self.epochs):
            model.train()
            loss = torch.tensor(0.0).to(self.device)

            for batch_i, batch_offset in enumerate(range(0, len(node_perm), self.batch_size)):
                logger.info(f"[Epoch-{epoch:03d} | Batch-{batch_i:02d}]")

                optimizer.zero_grad()

                out = model(data_x, data.edge_index)

                batch_indices = node_perm[batch_offset:batch_offset + self.batch_size]
                batch_loss = criterion(out[data.train_node_mask][batch_indices], data_y[data.train_node_mask][batch_indices])
                batch_loss.backward()
                loss += batch_loss.item()
                optimizer.step()

            if stopper is not None:
                with torch.no_grad():
                    model.eval()
                    val_metric = 'accuracy' if data.y.ndim == 1 else 'weighted_ap'

                    out = model(data_x, data.edge_index)
                    val_perf_dict = node_classification_performances(
                        y_true=data.y[data.val_node_mask],
                        y_out=out[data.val_node_mask],
                        metrics=val_metric
                    )
                    val_perf = val_perf_dict[val_metric]
                    logger.info(f'[Epoch-{epoch:03d}] Train Loss={loss.item():.4f} | Val {val_metric}={val_perf:.4f}')

                early_stop = stopper.step(val_perf, model)
                if stopper.has_improved:
                    best_val_epoch = epoch
                    if data.x is None:
                        best_data_x = data_x.detach().clone()
                if early_stop:
                    logger.info(f"[Epoch-{epoch}] Early stop!")
                    break

        if stopper is not None and stopper.early_stop:
            stopper.load_checkpoint(model)
            if data.x is None:
                data_x = best_data_x

        train_time = timer() - start
        logger.info("Finished training.")

        """Testing"""
        with torch.no_grad():
            model.eval()

            out = model(data_x, data.edge_index)
            perf_dict = node_classification_performances(
                y_true=data.y[data.test_node_mask],
                y_out=out[data.test_node_mask],
            )
            perf_dict.update({
                'train_time': train_time,
                'train_epochs': epoch,
                'best_val_epoch': best_val_epoch,
            })

            self.free_up_gpu_memory(node_perm)
            logger.info("Finished testing.")
            logger.info(f"perf_dict:\n{pformat(perf_dict)}")
            return perf_dict

    def train_test_node_classification_with_label_prop(self):
        model, data, logger = self.model, self.data, self.logger
        assert isinstance(model, LabelPropagation), model

        """Training"""
        logger.info(f'Starting training')
        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")

        logger.info(f"data (before transform): {data}")
        transform = T.Compose([T.ToUndirected(), T.ToSparseTensor()])
        data = transform(data)
        logger.info(f"data (after transform): {data}")

        if data.y.ndim == 1:  # single-label classification
            data_y = data.y
        else:  # multi-label classification
            data_y = data.y.float()
        # some graphs have -1 as labels for some of the nodes. they will be ignored by masking.
        data_y = torch.clamp(data_y, 0)

        start = timer()
        out = model(data_y, data.adj_t, mask=data.train_node_mask | data.val_node_mask)
        train_time = timer() - start

        """Testing"""
        perf_dict = node_classification_performances(
            y_true=data_y[data.test_node_mask],
            y_out=out[data.test_node_mask],
        )
        perf_dict.update({
            'train_time': train_time,
            'train_epochs': -1,
            'best_val_epoch': -1,
        })

        logger.info("Finished testing.")
        logger.info(f"perf_dict:\n{pformat(perf_dict)}")
        return perf_dict

    def train_test_link_prediction(self):
        def link_predict(z, edge_label_index):
            pred = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)
            assert len(pred) == edge_label_index.shape[1], (len(pred), edge_label_index.shape[1])
            return pred

        model, data, logger = self.model, self.data, self.logger
        assert not isinstance(model, LabelPropagation), model

        """Training"""
        logger.info(f'Starting training')
        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")
        model.train()

        if data.x is not None:
            data_x = data.x.float()
            if isinstance(data_x, SparseTensor):  # gcn edge weight normalization does not work when given a sparse feature tensor
                data_x = data_x.to_dense()
            parameters = model.parameters()
        else:
            data_x = data.rand_x.float()
            data_x = torch.nn.Parameter(data_x, requires_grad=True)  # learnable node features
            parameters = list(model.parameters()) + [data_x]

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        start = timer()
        stopper = EarlyStopping(self.patience, minimizing_objective=False, logger=logger) if self.patience >= 0 else None
        epoch = -1
        best_val_epoch = -1
        best_data_x = None
        edge_perm = torch.randperm(data.train_edge_label_index.size(1)).to(self.device)

        for epoch in range(self.epochs):
            model.train()
            loss = torch.tensor(0.0).to(self.device)

            for batch_i, batch_offset in enumerate(range(0, len(edge_perm), self.batch_size)):
                logger.info(f"[Epoch-{epoch:03d} | Batch-{batch_i:02d}]")

                optimizer.zero_grad()

                batch_indices = edge_perm[batch_offset:batch_offset + self.batch_size]
                z = model(data_x, data.train_edge_index)
                out = link_predict(z=z, edge_label_index=data.train_edge_label_index[:, batch_indices]).view(-1)

                batch_loss = criterion(out, data.train_edge_label[batch_indices])
                batch_loss.backward()
                loss += batch_loss.item()
                optimizer.step()

            if stopper is not None:
                with torch.no_grad():
                    model.eval()
                    val_metric = 'rocauc'

                    z = model(data_x, data.val_edge_index)
                    val_out = link_predict(z=z, edge_label_index=data.val_edge_label_index).view(-1)

                    # if torch.any(torch.isnan(val_out)).item():  # val_out contains nan
                    #     logger.info(f"[Epoch-{epoch:03d}] Train Loss={loss.item():.4f}")
                    #     continue

                    val_perf_dict = link_prediction_performances(
                        y_true=data.val_edge_label,
                        y_out=val_out,
                        metrics=val_metric
                    )
                    val_perf = val_perf_dict[val_metric]
                    logger.info(f"[Epoch-{epoch:03d}] Train Loss={loss.item():.4f} | Val {val_metric}={val_perf:.4f}")

                early_stop = stopper.step(val_perf, model)
                if stopper.has_improved:
                    best_val_epoch = epoch
                    if data.x is None:
                        best_data_x = data_x.detach().clone()
                if early_stop:
                    logger.info(f"[Epoch-{epoch}] Early stop!")
                    break

        if stopper is not None and stopper.early_stop:
            stopper.load_checkpoint(model)
            if data.x is None:
                data_x = best_data_x

        train_time = timer() - start
        logger.info("Finished training.")

        """Testing"""
        with torch.no_grad():
            model.eval()

            z = model(data_x, data.test_edge_index)
            test_out = link_predict(z=z, edge_label_index=data.test_edge_label_index).view(-1)
            perf_dict = link_prediction_performances(
                y_true=data.test_edge_label,
                y_out=test_out,
            )
            perf_dict.update({
                'train_time': train_time,
                'train_epochs': epoch,
                'best_val_epoch': best_val_epoch,
            })

            self.free_up_gpu_memory(edge_perm)
            logger.info("Finished testing.")
            logger.info(f"perf_dict:\n{pformat(perf_dict)}")
            return perf_dict

    def train_test_link_prediction_with_classical_models(self):
        model, data, logger = self.model, self.data, self.logger
        assert isinstance(model, ClassicalLinkPredictionModel), model

        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")

        nx_G = to_networkx(data=Data(edge_index=data.test_edge_index, num_nodes=data.num_nodes))
        nx_G = nx_G.to_undirected()  # classical models require the graph to be undirected
        # nk_G = to_networkit(edge_index=to_undirected(data.test_edge_index), num_nodes=data.num_nodes, directed=False)
        test_ebunch = data.test_edge_label_index.t().tolist()

        """Testing"""
        test_out = model(G=nx_G, ebunch=test_ebunch)
        test_out = torch.tensor([score for u, v, score in test_out])
        # test_out = model(G=nk_G, ebunch=test_ebunch)

        perf_dict = link_prediction_performances(
            y_true=data.test_edge_label,
            y_out=test_out,
        )
        perf_dict.update({
            'train_time': 0.0,
            'train_epochs': -1,
            'best_val_epoch': -1,
        })

        logger.info("Finished testing.")
        logger.info(f"perf_dict:\n{pformat(perf_dict)}")
        return perf_dict

    def train_test_link_prediction_with_seal(self):
        model, data, logger = self.model, self.data, self.logger
        assert isinstance(model, DGCNN), model

        """Training"""
        logger.info(f'Starting training')
        logger.info(f"Model: {model}")
        logger.info(f"Graph: {data}")
        model.train()

        train_dataset = SEALDataset(graph=data, num_hops=model.num_hops, mode='train')
        val_dataset = SEALDataset(graph=data, num_hops=model.num_hops, mode='val')
        test_dataset = SEALDataset(graph=data, num_hops=model.num_hops, mode='test')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # node features
        max_z = list(set([data.max_z.item() for data in train_dataset]))
        assert len(max_z) == 1, len(max_z)
        max_z = max_z[0]

        if data.x is not None:
            data_x = data.x.float().to(self.device)
            if isinstance(data_x, SparseTensor):  # gcn edge weight normalization does not work when given a sparse feature tensor
                data_x = data_x.to_dense()
            parameters = model.parameters()
        else:
            data_x = data.rand_x.float().to(self.device)
            data_x = torch.nn.Parameter(data_x, requires_grad=True)  # learnable node features
            parameters = list(model.parameters()) + [data_x]

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        def get_batch_node_feats(batch_data):
            """return batch node features (input features x concatenated with SEAL structural features z)"""
            seal_z = F.one_hot(batch_data.z, max_z + 1).to(torch.float).to(self.device)
            data_batch_x = torch.cat([data_x[batch_data.sub_nodes], seal_z],
                                     dim=1)  # combine input feature with seal z feature
            return data_batch_x

        def train():
            model.train()
            total_loss = 0

            for batch_i, batch_data in enumerate(train_loader):
                # data example: Data(edge_index=[2, 16], y=[1], sub_nodes=[7], z=[7], max_z=[1])
                batch_data = batch_data.to(self.device)

                optimizer.zero_grad()

                data_batch_x = get_batch_node_feats(batch_data)
                out = model(data_batch_x, batch_data.edge_index, batch_data.batch)

                loss = criterion(out.view(-1), batch_data.y.to(torch.float))
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * batch_data.num_graphs

            return total_loss / len(train_dataset)

        @torch.no_grad()
        def test(loader, eval_metric=None):
            model.eval()

            y_pred, y_true = [], []
            for batch_data in loader:
                batch_data = batch_data.to(self.device)

                data_batch_x = get_batch_node_feats(batch_data)

                logits = model(data_batch_x, batch_data.edge_index, batch_data.batch)
                y_pred.append(logits.view(-1).cpu())
                y_true.append(batch_data.y.view(-1).cpu().to(torch.float))

            perf_dict = link_prediction_performances(
                y_true=torch.cat(y_true),
                y_out=torch.cat(y_pred),
                metrics=eval_metric,
            )
            return perf_dict

        start = timer()
        stopper = EarlyStopping(self.patience, minimizing_objective=False,
                                logger=logger) if self.patience >= 0 else None
        epoch = -1
        best_val_epoch = -1
        best_data_x = None

        for epoch in range(self.epochs):
            loss = train()

            if stopper is not None:
                val_metric = 'rocauc'
                val_perf_dict = test(val_loader, eval_metric=val_metric)
                val_perf = val_perf_dict[val_metric]
                logger.info(f"[Epoch-{epoch:03d}] Train Loss={loss:.4f} | Val {val_metric}={val_perf:.4f}")

                early_stop = stopper.step(val_perf, model)
                if stopper.has_improved:
                    best_val_epoch = epoch
                    if data.x is None:
                        best_data_x = data_x.detach().clone()
                if early_stop:
                    logger.info(f"[Epoch-{epoch}] Early stop!")
                    break

        if stopper is not None and stopper.early_stop:
            stopper.load_checkpoint(model)
            if data.x is None:
                data_x = best_data_x

        train_time = timer() - start
        logger.info("Finished training.")

        """Testing"""
        test_perf_dict = test(test_loader)
        test_perf_dict.update({
            'train_time': train_time,
            'train_epochs': epoch,
            'best_val_epoch': best_val_epoch,
        })

        logger.info("Finished testing.")
        logger.info(f"test_perf_dict:\n{pformat(test_perf_dict)}")
        return test_perf_dict

    def free_up_gpu_memory(self, additional_cuda_tensors):
        if self.device != torch.device('cpu'):
            del self.model
            del self.data
            for t in additional_cuda_tensors:
                del t
            torch.cuda.empty_cache()


class PNAPartial:
    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs

    def instantiate(self, data: Data, task):
        from performances.taskrunner import Task
        if task is Task.LINK_PREDICTION:
            edge_index = data.train_edge_index
        else:
            edge_index = data.edge_index

        # Compute the maximum in-degree in the training data.
        d = degree(edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = int(d.max())

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long).to(edge_index.device)
        deg += torch.bincount(d, minlength=deg.numel())

        return PNA(in_channels=self.in_channels, out_channels=self.out_channels, deg=deg, **self.kwargs)


class DGCNNPartial:
    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs

    def instantiate(self, data: Data):
        train_dataset = SEALDataset(data, num_hops=self.kwargs['num_hops'], mode='train')
        print("train_dataset:", train_dataset)

        return DGCNN(data,
                     train_dataset,
                     num_hops=self.kwargs['num_hops'],
                     gnn_hidden_channels=self.kwargs['gnn_hidden_channels'],
                     gnn_conv=self.kwargs['gnn_conv'],
                     mlp_hidden_channels=self.kwargs['mlp_hidden_channels'],
                     k=self.kwargs['k'])


if __name__ == '__main__':
    from utils import setup_logger
    from pathlib import Path
    logger = setup_logger('logger1', log_file=Path("~/Downloads/benchmarklog.txt").expanduser())

    import settings
    import argparse
    from graphs.graphset import GraphSet
    from performances.taskrunner import Runner, Task

    # graphset = GraphSet(data_sources=['pyg'])
    graphset = GraphSet(data_sources=['netrepo'])
    print("graphset.graphs:", graphset.graphs)

    ################################################################################
    task = Task.LINK_PREDICTION
    # task = Task.NODE_CLASSIFICATION
    ################################################################################

    args = {'lr': 0.001, 'epochs': 2, 'patience': 10, 'device': torch.device('cpu'),
            'node_batch_size': 100, 'edge_batch_size': 300}
    if task is Task.LINK_PREDICTION:
        runner = Runner(task=Task.LINK_PREDICTION, root=settings.LINK_PRED_PERF_ROOT, args=argparse.Namespace(**args),
                        graphset=graphset, logger=logger)
    else:
        runner = Runner(task=Task.NODE_CLASSIFICATION, root=settings.NODE_CLASS_PERF_ROOT, args=argparse.Namespace(**args),
                        graphset=graphset, logger=logger)
    print(runner)
    graph: Data = runner.load_graph(graph_i=0, split_i=0)
    print(graph.name, graph)

    in_channels = graph.x.shape[1] if graph.x is not None else graph.rand_x.shape[1]
    print("in_channels:", in_channels)
    # model = GCN(in_channels=in_channels, hidden_channels=64, out_channels=graph.num_classes, num_layers=2)
    # model = GAT(in_channels=in_channels, hidden_channels=64, out_channels=graph.num_classes, num_layers=2, heads=4)
    # model = SpectralEmbedding(num_components=16, out_channels=graph.num_classes, tolerance=0.001).train_unsupervised_node_emb(graph, torch.device('cpu'))
    # model = GraRep(num_components=16, out_channels=graph.num_classes, power=2).train_unsupervised_node_emb(graph)
    # model = PNAPartial(in_channels=in_channels, hidden_channels=64, out_channels=graph.num_classes, num_layers=1,
    #                    aggregators=['sum'], scalers=['amplification'], towers=1).instantiate(graph, task)
    # model = LabelPropagation(num_layers=1, alpha=0.9)
    model = ClassicalLinkPredictionModel(model_name="adamic_adar")
    print(model)

    if task is Task.LINK_PREDICTION:
        if isinstance(model, ClassicalLinkPredictionModel):
            link_pred_perf_dict = Trainer(
                model, graph, logger, epochs=300, patience=20, batch_size=100000
            ).train_test_link_prediction_with_classical_models()
        else:
            link_pred_perf_dict = Trainer(
                model, graph, logger, epochs=300, patience=20, batch_size=100000
            ).train_test_link_prediction()
        print("link_pred_perf_dict:", link_pred_perf_dict)
    else:
        if isinstance(model, LabelPropagation):
            node_class_perf_dict = Trainer(model, graph, logger).train_test_node_classification_with_label_prop()
        else:
            node_class_perf_dict = Trainer(
                model, graph, logger, lr=0.01, epochs=300, patience=50, batch_size=10000
            ).train_test_node_classification()
        print("node_class_perf_dict:", node_class_perf_dict)
