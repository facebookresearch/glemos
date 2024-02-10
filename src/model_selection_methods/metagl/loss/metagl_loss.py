# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch.nn import functional as F

PADDED_Y_VALUE = float('nan')


def top_one_loss(y_pred, y_true, eps=1e-10, padded_value_indicator=PADDED_Y_VALUE):
    """
    Top-1 probability-based loss.
    Code adapted from https://github.com/allegro/allRank/blob/master/allrank/models/losses/listNet.py
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    if math.isnan(padded_value_indicator):
        mask = torch.isnan(y_true)
    else:
        mask = y_true == padded_value_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
