# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle

import numpy as np

import settings

# load all variables
workspace_file = settings.WORKSPACE_FILE
print("\nLoading workspace file: ", workspace_file)
workspace_dict = pickle.load(workspace_file.open('rb'))

link_pred_workspace = dict(workspace_dict['link-pred'])
node_class_workspace = dict(workspace_dict['node-class'])
print('\nLoading Variables (link-pred):\n=========================\n', list(link_pred_workspace.keys()))
print('\nLoading Variables (node-class):\n=========================\n', list(node_class_workspace.keys()))

# perf matrices
link_pred_P_map = link_pred_workspace['P_map']  # shape=(num graphs, num models)
link_pred_P_auc = link_pred_workspace['P_auc']
link_pred_P_ndcg = link_pred_workspace['P_ndcg']
link_pred_perf_matrices = {
    'map': link_pred_P_map,
    'auc': link_pred_P_auc,
    'ndcg': link_pred_P_ndcg,
}

node_class_P_map = node_class_workspace['P_map']
node_class_P_acc = node_class_workspace['P_acc']
node_class_P_f1 = node_class_workspace['P_f1']
node_class_perf_matrices = {
    'map': node_class_P_map,
    'acc': node_class_P_acc,
    'f1': node_class_P_f1,
}

perf_matrices = {
    'link-pred': link_pred_perf_matrices,
    'node-class': node_class_perf_matrices,
}

# meta-graph features
link_pred_M = np.nan_to_num(link_pred_workspace['M'])  # 318 features
link_pred_M_tiny = np.nan_to_num(link_pred_workspace['M_tiny'])  # 13 features
link_pred_M_compact = np.nan_to_num(link_pred_workspace['M_simple'])  # 58 features
link_pred_M_graphlets_complex = np.nan_to_num(link_pred_workspace['M_graphlets_complex'])  # 756 features
link_pred_M_regular_graphlets = np.concatenate((link_pred_M, link_pred_M_graphlets_complex), axis=1)  # 1074 features
link_pred_M_all = np.concatenate((link_pred_M, link_pred_M_compact, link_pred_M_graphlets_complex), axis=1)  # 1148 features
link_pred_meta_features = {
    'regular': link_pred_M,
    'tiny': link_pred_M_tiny,
    'compact': link_pred_M_compact,
    'graphlets_complex': link_pred_M_graphlets_complex,
    'regular_graphlets': link_pred_M_regular_graphlets,
    'all': link_pred_M_all,
}

node_class_M = np.nan_to_num(node_class_workspace['M'])  # 318 features
node_class_M_tiny = np.nan_to_num(node_class_workspace['M_tiny'])  # 13 features
node_class_M_compact = np.nan_to_num(node_class_workspace['M_simple'])  # 58 features
node_class_M_graphlets_complex = np.nan_to_num(node_class_workspace['M_graphlets_complex'])  # 756 features.
node_class_M_regular_graphlets = np.concatenate((node_class_M, node_class_M_graphlets_complex), axis=1)  # 1074 features
node_class_M_all = np.concatenate((node_class_M, node_class_M_compact, node_class_M_graphlets_complex), axis=1)  # 1148 features
node_class_meta_features = {
    'regular': node_class_M,
    'tiny': node_class_M_tiny,
    'compact': node_class_M_compact,
    'graphlets_complex': node_class_M_graphlets_complex,
    'regular_graphlets': node_class_M_regular_graphlets,
    'all': node_class_M_all,
}

meta_features = {
    'link-pred': link_pred_meta_features,
    'node-class': node_class_meta_features,
}

# other variables
all_models = {
    'link-pred': list(link_pred_workspace['all_models']),
    'node-class': list(node_class_workspace['all_models']),
}
graph_names = {
    'link-pred': list(link_pred_workspace['graphs_used']),
    'node-class': list(node_class_workspace['graphs_used']),
}
y_graph_domain = {
    'link-pred': list(link_pred_workspace['y_graph_domain']),
    'node-class': list(node_class_workspace['y_graph_domain']),
}
