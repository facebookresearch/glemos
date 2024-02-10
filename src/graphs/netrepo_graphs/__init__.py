# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from graphs.netrepo_graphs import netrepo_labeled_graphs
from graphs.netrepo_graphs import netrepo_unlabeled_graphs


def load_graphs():
    return netrepo_labeled_graphs.load_graphs() + netrepo_unlabeled_graphs.load_graphs()

if __name__ == '__main__':
    load_graphs()
