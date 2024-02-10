# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from contextlib import contextmanager
from timeit import default_timer

import numpy as np
import torch


# sec = tic()
# ...statements...
# sec = toc(sec)
def tic(): return time.time()


def toc(start_time): return time.time() - start_time  # elapsed time (from start_time) in seconds


def setup_cuda(args):
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.device)
    return args


def as_torch_tensor(X):
    if isinstance(X, torch.Tensor):
        return X
    elif isinstance(X, np.ndarray):
        return torch.from_numpy(X).float()
    else:
        raise TypeError(f"Invalid type: {type(X)}")


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def remap_labels(labels: np.ndarray):
    """remap labels to start from zero & to be consecutive"""
    assert labels.ndim == 1, labels.ndim

    uniq_labels = np.sort(np.unique(labels))
    remap_dict = {old_label: new_label for new_label, old_label in enumerate(uniq_labels)}

    remapped_labels = np.array([remap_dict[label] for label in labels])
    remapped = not np.array_equal(labels, remapped_labels)
    return remapped_labels, remapped, remap_dict
