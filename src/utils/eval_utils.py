# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, ndcg_score, roc_auc_score

from testbeds import CrossTaskTestbed


def create_eval_dict(metrics=None):
    if metrics is None:
        metrics = ['AUC', 'AUC@1', 'AUC@5', 'AUC@10',
                   'MAP', 'MAP@1', 'MAP@5', 'MAP@10',
                   'MRR', 'MRR@1', 'MRR@5', 'MRR@10',
                   'HR@1', 'HR@2', 'HR@3', 'HR@4', 'HR@5', 'HR@10',
                   'nDCG@1', 'nDCG@2', 'nDCG@3', 'nDCG@4', 'nDCG@5', 'nDCG@10']
    assert len(metrics) > 0
    return {metric: [] for metric in metrics}


def hit_rate_at_k(y_true, y_score, k=10):
    """
    Compute hit rate at k
    """
    y_true_flat = np.array(y_true).flatten()
    idx_true = np.argsort(y_true_flat)[::-1]

    y_score_flat = np.array(y_score).flatten()
    idx_pred_score = np.argsort(y_score_flat)[::-1]
    return np.intersect1d(idx_pred_score[0:k], idx_true[0:k]).shape[0] / (1. * k)


def eval_metrics_single_graph(y_true, y_pred, top_k=None):
    assert y_true.ndim == y_pred.ndim == 1, (y_true.shape, y_pred.shape)
    assert y_pred.shape == y_true.shape, y_pred.shape
    if top_k is None:
        top_k = [1, 10]

    eval_dict = {}
    for k in top_k:
        eval_dict[f'nDCG@{k}'] = ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)
        eval_dict[f'HR@{k}'] = hit_rate_at_k(y_true, y_pred, k=k)

    num_models = len(y_true)
    for k in list(filter(lambda x: x <= len(y_true), top_k)):
        top_k_ind = np.argpartition(y_true, -k)[-k:]
        y_true_bin = np.zeros((num_models,), dtype=int)
        y_true_bin[top_k_ind] = 1

        eval_dict[f'AUC@{k}'] = roc_auc_score(y_true_bin, y_pred)
        eval_dict[f'MAP@{k}'] = average_precision_score(y_true_bin, y_pred)
        eval_dict[f'MRR@{k}'] = label_ranking_average_precision_score(y_true_bin.reshape(1, -1), y_pred.reshape(1, -1))

    return eval_dict


def binarize_perf(Y):
    """For each row, set the maximum element to 1, and all others to 0"""
    Y = np.asarray(Y)
    Y_bin = np.zeros_like(Y, dtype=int)
    Y_bin[np.arange(len(Y)), Y.argmax(1)] = 1
    # for y_bin in Y_bin:
    #     assert y_bin.sum() == 1, y_bin.sum()
    return Y_bin


def eval_metrics(Y_true, Y_pred, Y_true_bin=None):
    assert len(Y_true.shape) == 2 and Y_true.shape == Y_pred.shape, (Y_true.shape, Y_pred.shape)
    if isinstance(Y_pred, torch.Tensor):
        Y_pred = Y_pred.cpu().detach().numpy()
    if isinstance(Y_true, torch.Tensor):
        Y_true = Y_true.cpu().detach().numpy()
    Y_pred, Y_true = np.asarray(Y_pred), np.asarray(Y_true)

    if Y_true_bin is None:
        Y_true_bin = binarize_perf(Y_true)

    eval_dict = {}
    hit_rate_at_one = partial(hit_rate_at_k, k=1)

    with mp.Pool(processes=None) as pool:
        binary_args = []
        for y_true_bin, y_pred in zip(Y_true_bin, Y_pred):
            binary_args.append((np.array(y_true_bin).flatten(), np.array(y_pred).flatten()))

        # binary_args2 = []
        # for y_true, y_pred in zip(Y_true, Y_pred):
        #     idx_best_model = np.argmax(y_true)
        #     num_models = Y_true.shape[1]
        #     y_true_bin = np.matrix(np.zeros((1, num_models), dtype=int))
        #     y_true_bin[0, idx_best_model] = 1
        #     binary_args2.append((np.array(y_true_bin).flatten(), np.array(y_pred).flatten()))

        # note: after upgrading packages, using multiprocessing is somehow slower than single-process version.
        # eval_dict['AUC'] = np.mean(pool.starmap(roc_auc_score, binary_args))
        eval_dict['AUC'] = np.mean([roc_auc_score(y_true_bin, y_pred) for y_true_bin, y_pred in binary_args])

        # eval_dict['MRR'] = eval_dict['MAP'] = np.mean(pool.starmap(average_precision_score, binary_args))
        eval_dict['MRR'] = eval_dict['MAP'] = np.mean([average_precision_score(y_true_bin, y_pred) for y_true_bin, y_pred in binary_args])

        # eval_dict['HR@1'] = np.mean(pool.starmap(partial(hit_rate_at_k, k=1), zip(Y_true, Y_pred)))
        eval_dict['HR@1'] = np.mean([hit_rate_at_one(y_true, y_pred) for y_true, y_pred in zip(Y_true, Y_pred)])

    eval_dict['nDCG@1'] = ndcg_score(Y_true, Y_pred, k=1)

    return eval_dict


def output_results(result_dir, testbed, method_names, method_eval, meta_feat, args, num_decimals=4):
    """
    Output performance results

    :param method_names: list of method names
    :param method_eval: list of eval dicts for each method given in `method_names`
    """
    assert len(method_names) == len(method_eval), (len(method_names), len(method_eval))
    metric_names = ['AUC@1', 'MAP@1', 'nDCG@1', 'HR@1']

    """Find the best meta-learner for each metric"""
    best_models_bool = {}
    for midx, metric in enumerate(metric_names):
        best_perf = -1.
        best_method_name = ''
        for meth_idx, method in enumerate(method_eval):
            if np.mean(method[metric]) >= best_perf:
                best_perf = np.mean(method[metric])
                method_name = method_names[meth_idx]
                best_method_name = method_name

        best_models_bool[metric, best_method_name] = True
        print('[best] ', metric, best_method_name)

        for meth_idx, method in enumerate(method_eval):  # Mark the methods that tie with best
            if np.mean(method[metric]) == best_perf:
                method_name = method_names[meth_idx]
                best_models_bool[metric, method_name] = True
                print('\t[best tied] ', metric, method_name)

    """Create a result table with methods on rows, and metrics on columns"""
    result_mat_tmp = np.zeros((len(method_eval), len(metric_names)), dtype=float)
    error_mat_tmp = np.zeros((len(method_eval), len(metric_names)), dtype=float)

    for meth_idx, method in enumerate(method_eval):
        for midx, metric in enumerate(metric_names):
            result_mat_tmp[meth_idx, midx] = np.round(np.mean(method[metric]), num_decimals)
            error_mat_tmp[meth_idx, midx] = np.round(np.std(method[metric]), num_decimals)

    if isinstance(testbed, CrossTaskTestbed):
        perf_metric = f"source-{testbed.source_perf_metric}_target-{testbed.target_perf_metric}"
    else:
        perf_metric = testbed.perf_metric
    result_fn = 'P=' + perf_metric + '_' + meta_feat

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    result_df = pd.DataFrame(result_mat_tmp, index=method_names, columns=metric_names)
    print("\nResults:")
    print(result_df)
    result_df.to_csv(result_dir / f'{result_fn}.csv')

    error_df = pd.DataFrame(error_mat_tmp, index=method_names, columns=metric_names)
    print("\nStd:")
    print(error_df)
    error_df.to_csv(result_dir / f'{result_fn}-error.csv')
