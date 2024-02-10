# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score


def link_prediction_performances(
        y_true,  # shape=(# pos edges + # neg edges,). binary labels (1=pos edge, 0=neg edge).
        y_out,  # shape=(# pos edges + # neg edges,)
        metrics=None
):
    assert torch.is_tensor(y_true) and torch.is_tensor(y_out), (y_true, y_out)
    assert y_true.ndim == y_out.ndim == 1, (y_true.shape == y_out.shape)

    y_true = y_true.detach().cpu().numpy()
    y_out = y_out.detach().cpu()
    y_score = y_out.sigmoid().numpy()

    if metrics is None:
        metrics = ['rocauc', 'ap', 'ndcg']
    if not isinstance(metrics, list):
        metrics = [metrics]

    perf_dict = {}
    for metric in metrics:
        if metric == 'rocauc':  # roc auc: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
            perf_dict['rocauc'] = roc_auc_score(y_true, y_score)
        elif metric == 'ap':  # average precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
            perf_dict['ap'] = average_precision_score(y_true, y_score)
        elif metric == 'ndcg':  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
            perf_dict['ndcg'] = ndcg_score(np.asarray([y_true]), np.asarray([y_score]), ignore_ties=True)
        else:
            raise ValueError(f"Invalid metric: {metric}")

    return perf_dict


if __name__ == '__main__':
    from pprint import pprint

    y_true = torch.tensor([1, 1, 1, 0, 0, 0])
    y_out = torch.tensor([10.0, 1.0, 0.5, 0.1, 0.5, -2.0])

    perf_dict = link_prediction_performances(y_true, y_out)
    pprint(perf_dict)
