# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score, \
    average_precision_score, roc_auc_score


def node_classification_performances(
        y_true,  # shape=(# nodes,) (if single-label classification) or (# nodes, # classes) (if multi-label classification)
        y_out,  # shape=(# nodes, # classes)
        metrics=None
):
    assert torch.is_tensor(y_true) and torch.is_tensor(y_out), (y_true, y_out)
    assert y_true.ndim in [1, 2], y_true.ndim  # 2-dim y_true: multi-label classification
    assert y_out.ndim == 2, y_true.ndim
    y_true = y_true.detach().cpu().numpy()
    y_out = y_out.detach().cpu()

    y_pred_single = y_out.argmax(axis=1).numpy()
    y_pred_multi = y_out.sigmoid().round().numpy()
    y_pred_multi_score = y_out.sigmoid().numpy()
    if y_true.ndim == 1:
        y_pred = y_pred_single
    else:
        y_pred = y_pred_multi

    if metrics is None:
        metrics = ['accuracy', 'jaccard', 'precision', 'recall', 'f1score', 'ap', 'rocauc']
    if not isinstance(metrics, list):
        metrics = [metrics]

    perf_dict = {}
    for metric in metrics:
        if metric == 'accuracy':  # accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
            perf_dict.update({
                'accuracy': accuracy_score(y_true, y_pred),
            })
        elif metric.endswith('jaccard'):  # Jaccard similarity coefficient:  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
            perf_dict.update({
                'micro_jaccard': jaccard_score(y_true, y_pred, average='micro'),
                'macro_jaccard': jaccard_score(y_true, y_pred, average='macro'),
                'weighted_jaccard': jaccard_score(y_true, y_pred, average='weighted'),
            })
        elif metric.endswith('precision'):  # precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
            perf_dict.update({
                'micro_precision': precision_score(y_true, y_pred, average='micro'),
                'macro_precision': precision_score(y_true, y_pred, average='macro'),
                'weighted_precision': precision_score(y_true, y_pred, average='weighted'),
            })
        elif metric.endswith('recall'):  # recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
            perf_dict.update({
                'micro_recall': recall_score(y_true, y_pred, average='micro'),
                'macro_recall': recall_score(y_true, y_pred, average='macro'),
                'weighted_recall': recall_score(y_true, y_pred, average='weighted'),
            })
        elif metric.endswith('f1score'):  # f1 score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
            perf_dict.update({
                'micro_f1score': f1_score(y_true, y_pred, average='micro'),
                'macro_f1score': f1_score(y_true, y_pred, average='macro'),
                'weighted_f1score': f1_score(y_true, y_pred, average='weighted'),
            })
        elif metric.endswith('ap'):  # average precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
            if y_true.ndim == 2:  # multi-label classification
                y_score = y_pred_multi_score  # shape=(# samples, # classes)
                y_true_2d = y_true
            else:
                y_score = torch.softmax(y_out, dim=1)  # shape=(# samples, # classes)
                y_true_2d = F.one_hot(torch.from_numpy(y_true), num_classes=y_out.shape[1])

            perf_dict.update({
                'micro_ap': average_precision_score(y_true_2d, y_score, average='micro'),
                'macro_ap': average_precision_score(y_true_2d, y_score, average='macro'),
                'weighted_ap': average_precision_score(y_true_2d, y_score, average='weighted'),
            })
        elif metric.endswith('rocauc'):  # roc auc: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
            if y_true.ndim == 2:  # multi-label classification
                y_score = y_pred_multi_score  # shape=(# samples, # classes)
            else:
                y_score = torch.softmax(y_out, dim=1)  # shape=(# samples, # classes)

            labels = np.arange(y_out.shape[1])
            try:
                perf_dict.update({
                    # 'macro_ovr_rocauc': roc_auc_score(y_true, y_score, average='macro', multi_class='ovr', labels=np.arange(y_out.shape[1])),
                    # 'weighted_ovr_rocauc': roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr', labels=labels),
                    'macro_ovo_rocauc': roc_auc_score(y_true, y_score, average='macro', multi_class='ovo', labels=labels),
                    'weighted_ovo_rocauc': roc_auc_score(y_true, y_score, average='weighted', multi_class='ovo', labels=labels),
                })
            except Exception:
                # in multi-label classification, encounters ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
                pass
        else:
            raise ValueError(f"Invalid metric: {metric}")

    return perf_dict


if __name__ == '__main__':
    from pprint import pprint

    # # test single-label classification
    y_true = torch.tensor([0, 1, 2, 0])
    y_out = torch.tensor([
        [0.8, 0, -0.5],  # 0
        [0.8, 0.6, -0.5],  # 0
        [0.2, 0.1, 0.7],  # 2
        [0.6, 0.2, -0.1],  # 0
    ])
    perf_dict = node_classification_performances(y_true, y_out)
    pprint(perf_dict)

    # test multi-label classification
    y_true = torch.tensor([
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ])
    perf_dict = node_classification_performances(y_true, y_out)
    pprint(perf_dict)
