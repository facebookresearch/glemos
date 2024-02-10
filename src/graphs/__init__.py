# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from graphs import netrepo_graphs
from graphs import pyg_graphs


def load_graphs():
    return netrepo_graphs.load_graphs() + pyg_graphs.load_graphs()


if __name__ == '__main__':
    from graphs.graphset import print_graph_stats
    graphs = load_graphs()
    print_graph_stats(load_graphs())
