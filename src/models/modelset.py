# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import functools
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch_geometric.nn import GCN, GraphSAGE, GAT, GIN, EGConv, LabelPropagation
from torch_geometric.nn.conv import SGConv, ChebConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN

from models.linkpred_models.classical_models import ClassicalLinkPredictionModel
from models.trainer import PNAPartial, DGCNNPartial
from models.unsupervised_models import SpectralEmbedding, GraRep, DGI, Node2VecModel


class ModelSet:
    def __init__(self, model_setting_groups=None):
        if model_setting_groups is None:
            model_setting_groups = [
                GCNModelSettings(),
                GraphSAGEModelSettings(),
                GATModelSettings(),
                GINModelSettings(),
                EGCModelSettings(),
                SGCModelSettings(),
                ChebNetModelSettings(),
                PNAModelSettings(),
                SpectralEmbeddingModelSettings(),
                GraRepModelSettings(),
                DGIModelSettings(),
                Node2VecModelSettings(),
                LabelPropagationModelSettings(),
                JaccardModelSettings(),
                ResourceAllocationModelSettings(),
                AdamicAdarModelSettings(),
                SEALModelSettings(),
            ]

        self.model_setting_groups = model_setting_groups
        self.model_settings = self.load_model_settings()
        assert len(self.model_settings) == \
               sum([g.num_settings(exclude_duplicate_settings=True) for g in self.model_setting_groups])

    def load_model_settings(self):
        model_settings_list = []
        for m_group in self.model_setting_groups:
            for setting_dict in m_group.get_model_setting_dicts():
                model_settings_list.append((m_group.__class__, setting_dict))
        return model_settings_list

    def load_model(self, model_i, in_channels, out_channels):
        model_setting_cls, model_setting = self.get_model_setting(model_i)
        return model_setting_cls.load_model(in_channels=in_channels, out_channels=out_channels, **model_setting)

    def get_model_setting(self, model_i):
        return self.model_settings[model_i]

    def get_model_setting_repr(self, model_i):
        model_setting_cls, model_setting = self.model_settings[model_i]
        params = "-".join([f"{k}={model_setting[k]}" for k in sorted(model_setting.keys())])
        model_name = model_setting_cls.__name__.replace("ModelSettings", "")
        if params:
            return f"{model_name}-{params}"
        else:
            return f"{model_name}"

    def __len__(self):
        return len(self.model_settings)


class ModelSettings(ABC):
    def __init__(self):
        self.variable_hyperparams = OrderedDict()  # dict (key=parameter name, value=parameter values)

    def get_model_setting_dicts(self, exclude_duplicate_settings=True):
        param_names = list(self.variable_hyperparams.keys())
        param_values_list = list(self.variable_hyperparams.values())

        model_setting_dicts = []
        for param_values in itertools.product(*param_values_list):
            param_dict = dict(zip(param_names, param_values))
            model_setting_dicts.append(param_dict)

        if exclude_duplicate_settings:
            params = self.variable_hyperparams

            # There exist some settings that are effectively the same
            # if "num_layers" list includes 1 & "hidden_channels" has more than one option
            # since "hidden_channels" is not used when "num_layers" is 1.
            if "num_layers" in params and 1 in params["num_layers"] \
                    and "hidden_channels" in params and len(params["hidden_channels"]) > 1:

                min_hidden_channels = sorted(params["hidden_channels"])[0]
                unique_model_setting_dicts = []
                for param_dict in model_setting_dicts:
                    if param_dict["num_layers"] == 1 and param_dict["hidden_channels"] != min_hidden_channels:
                        continue
                    unique_model_setting_dicts.append(param_dict)
                model_setting_dicts = unique_model_setting_dicts

        return model_setting_dicts

    def num_settings(self, exclude_duplicate_settings=True):
        param_values = self.variable_hyperparams.values()
        if not param_values:
            return 1  # model with no hyperparameters

        if exclude_duplicate_settings:
            params = self.variable_hyperparams

            if "num_layers" in params and 1 in params["num_layers"] \
                    and "hidden_channels" in params and len(params["hidden_channels"]) > 1:
                params_other_than_num_layers = [p for p in params.keys() if p != "num_layers"]
                params_other_than_num_layers_and_hidden_channels = [p for p in params.keys()
                                                                    if p != "num_layers" and p != "hidden_channels"]

                num1, num1plus = 0, 0
                for num_layers in params["num_layers"]:
                    if num_layers == 1:
                        param_vals = [params[p] for p in params_other_than_num_layers_and_hidden_channels]
                        num1 = functools.reduce(lambda x, y: x * len(y), param_vals, 1)
                    else:
                        assert num_layers > 1, num_layers
                        param_vals = [params[p] for p in params_other_than_num_layers]
                        num1plus += functools.reduce(lambda x, y: x * len(y), param_vals, 1)
                return num1 + num1plus
            else:
                return functools.reduce(lambda x, y: x * len(y), param_values, 1)
        else:
            return functools.reduce(lambda x, y: x * len(y), param_values, 1)

    def __len__(self):
        return self.num_settings()

    @classmethod
    def alpha_ordered_dict(cls, d: dict):
        ordered_keys = sorted(d.keys())
        assert all([isinstance(k, str) for k in ordered_keys]), ordered_keys
        return OrderedDict([(k, d[k]) for k in ordered_keys])

    @classmethod
    @abstractmethod
    def load_model(cls, in_channels, out_channels, **params):
        raise NotImplementedError


class GCNModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'num_layers': [1, 2, 3],
            'dropout': [0.0, 0.5],
            'act': ['relu', 'tanh', 'elu'],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return GCN(in_channels=in_channels, out_channels=out_channels, **params)


class GraphSAGEModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'num_layers': [1, 2],
            'act': ['relu', 'tanh'],
            'aggr': ['mean', 'max'],
            # 'aggr': ['lstm'],  # with 'lstm', ValueError: Can not perform aggregation since the 'index' tensor is not sorted
            'jk': [None, 'last']
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return GraphSAGE(in_channels=in_channels, out_channels=out_channels, **params)


class GATModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'num_layers': [1, 2, 3],
            'dropout': [0.0, 0.5],
            'heads': [1, 4],
            'concat': [True, False],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return GAT(in_channels=in_channels, out_channels=out_channels, **params)


class GINModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'num_layers': [1, 2, 3],
            'train_eps': [True, False],
            'eps': [0.0],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return GIN(in_channels=in_channels, out_channels=out_channels, **params)


class SGCModelSettings(ModelSettings):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SGConv.html
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'k': [1, 2, 3, 4, 5],
            'bias': [True, False],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return SGConv(in_channels=in_channels, out_channels=out_channels, **params)



class ChebNet(BasicGNN):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.ChebConv.html
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return ChebConv(in_channels, out_channels, **kwargs)


class ChebNetModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'num_layers': [1, 2],
            'K': [1, 2, 3],
            'normalization': [None, 'sym', 'rw'],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return ChebNet(in_channels=in_channels, out_channels=out_channels, **params)


class EGC(BasicGNN):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.EGConv.html
    supports_edge_weight = False
    supports_edge_attr = False

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return EGConv(in_channels, out_channels, **kwargs)


class EGCModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'num_layers': [2],
            'aggregators': [['sum'], ['mean'], ['symnorm'], ['min'], ['max'], ['var'], ['std']],
            'num_bases': [4, 8],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return EGC(in_channels=in_channels, out_channels=out_channels, num_heads=1, **params)


class PNAModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16],
            'num_layers': [1, 2],
            'aggregators': [['sum'], ['mean'], ['max'], ['var']],
            # 'aggregators': [['sum'], ['mean'], ['min'], ['max'], ['var'], ['std']],
            'scalers': [['identity'], ['amplification'], ['attenuation'], ['linear']],
            # 'scalers': [['identity'], ['amplification'], ['attenuation'], ['linear'], ['inverse_linear']],
            'towers': [1],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return PNAPartial(in_channels=in_channels, out_channels=out_channels, **params)


class SpectralEmbeddingModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'num_components': [16, 64],
            'tolerance': [0.1, 0.01, 0.001, 0.0001],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return SpectralEmbedding(in_channels=in_channels, out_channels=out_channels, **params)


class GraRepModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'num_components': [16, 32, 64],
            'power': [1, 2],
            # 'power': [1, 2, 3, 4],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return GraRep(in_channels=in_channels, out_channels=out_channels, **params)


class DGIModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'summary': ['mean', 'max', 'min', 'var'],
            'encoder_act': ['prelu', 'relu', 'tanh'],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return DGI(in_channels=in_channels, out_channels=out_channels, **params)


class Node2VecModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'hidden_channels': [16, 64],
            'p': [1, 2, 4],
            'q': [1, 2, 4],
            'walk_length': [10, 20],
            # 'walks_per_node': [1, 10],  # not supported by grape
            'context_size': [5, 10],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return Node2VecModel(in_channels=in_channels, out_channels=out_channels, **params)


class LabelPropagationModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'num_layers': [1, 2, 3, 4],
            'alpha': [0.99, 0.9, 0.8, 0.7],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return LabelPropagation(**params)


class AdamicAdarModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({})

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return ClassicalLinkPredictionModel(model_name="adamic_adar")


class JaccardModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({})

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return ClassicalLinkPredictionModel(model_name="jaccard")


class ResourceAllocationModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({})

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return ClassicalLinkPredictionModel(model_name="resource_allocation")


class SEALModelSettings(ModelSettings):
    def __init__(self):
        super().__init__()
        self.variable_hyperparams = ModelSettings.alpha_ordered_dict({
            'num_hops': [1],
            'k': [0.6, 0.1],
            'gnn_hidden_channels': [16, 64, 128],
            'gnn_conv': ['GCN', 'SAGE', 'GAT'],
            'mlp_hidden_channels': [32, 128],
        })

    @classmethod
    def load_model(cls, in_channels, out_channels, **params):
        return DGCNNPartial(in_channels=in_channels, out_channels=out_channels, **params)


node_classification_model_setting_groups = [
    GCNModelSettings(),
    GraphSAGEModelSettings(),
    GATModelSettings(),
    GINModelSettings(),
    EGCModelSettings(),
    SGCModelSettings(),
    ChebNetModelSettings(),
    PNAModelSettings(),
    SpectralEmbeddingModelSettings(),
    GraRepModelSettings(),
    DGIModelSettings(),
    LabelPropagationModelSettings(),
    Node2VecModelSettings(),
]

link_prediction_model_setting_groups = [
    GCNModelSettings(),
    GraphSAGEModelSettings(),
    GATModelSettings(),
    GINModelSettings(),
    EGCModelSettings(),
    SGCModelSettings(),
    ChebNetModelSettings(),
    PNAModelSettings(),
    SpectralEmbeddingModelSettings(),
    GraRepModelSettings(),
    DGIModelSettings(),
    Node2VecModelSettings(),
    AdamicAdarModelSettings(),
    JaccardModelSettings(),
    ResourceAllocationModelSettings(),
    SEALModelSettings(),
]


if __name__ == '__main__':
    print("# GCNModelSettings:", GCNModelSettings().num_settings())
    for setting_dict in GCNModelSettings().get_model_setting_dicts():
        GCNModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# GraphSAGEModelSettings:", GraphSAGEModelSettings().num_settings())
    for setting_dict in GraphSAGEModelSettings().get_model_setting_dicts():
        GraphSAGEModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# GATModelSettings:", GATModelSettings().num_settings())
    for setting_dict in GATModelSettings().get_model_setting_dicts():
        GATModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# GINModelSettings:", GINModelSettings().num_settings())
    for setting_dict in GINModelSettings().get_model_setting_dicts():
        GINModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# SGCModelSettings:", SGCModelSettings().num_settings())
    for setting_dict in SGCModelSettings().get_model_setting_dicts():
        SGCModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# ChebNetModelSettings:", ChebNetModelSettings().num_settings())
    for setting_dict in ChebNetModelSettings().get_model_setting_dicts():
        ChebNetModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# EGCModelSettings:", EGCModelSettings().num_settings())
    for setting_dict in EGCModelSettings().get_model_setting_dicts():
        EGCModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# PNAModelSettings:", PNAModelSettings().num_settings())
    for setting_dict in PNAModelSettings().get_model_setting_dicts():
        PNAModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# SpectralEmbeddingModelSettings:", SpectralEmbeddingModelSettings().num_settings())
    for setting_dict in SpectralEmbeddingModelSettings().get_model_setting_dicts():
        SpectralEmbeddingModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# GraRepModelSettings:", GraRepModelSettings().num_settings())
    for setting_dict in GraRepModelSettings().get_model_setting_dicts():
        GraRepModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# DGIModelSettings:", DGIModelSettings().num_settings())
    for setting_dict in DGIModelSettings().get_model_setting_dicts():
        DGIModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# Node2VecModelSettings:", Node2VecModelSettings().num_settings())
    for setting_dict in Node2VecModelSettings().get_model_setting_dicts():
        Node2VecModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# LabelPropagationModelSettings:", LabelPropagationModelSettings().num_settings())
    for setting_dict in LabelPropagationModelSettings().get_model_setting_dicts():
        LabelPropagationModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# AdamicAdarModelSettings:", AdamicAdarModelSettings().num_settings())
    for setting_dict in AdamicAdarModelSettings().get_model_setting_dicts():
        AdamicAdarModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# JaccardModelSettings:", JaccardModelSettings().num_settings())
    for setting_dict in JaccardModelSettings().get_model_setting_dicts():
        JaccardModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# ResourceAllocationModelSettings:", ResourceAllocationModelSettings().num_settings())
    for setting_dict in ResourceAllocationModelSettings().get_model_setting_dicts():
        ResourceAllocationModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    print("# SealModelSettings:", SEALModelSettings().num_settings())
    for setting_dict in SEALModelSettings().get_model_setting_dicts():
        SEALModelSettings.load_model(in_channels=32, out_channels=16, **setting_dict)

    model_set = ModelSet()
    print("=" * 80 + "\n" + "# all models:", len(model_set))

    # model_index = 100
    # print(model_set.get_model_setting(model_index))
    # print(model_set.load_model(model_i=model_index, in_channels=32, out_channels=16))
