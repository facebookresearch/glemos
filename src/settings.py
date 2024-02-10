# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path

ROOT_DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / '..'
DATA_ROOT = ROOT_DIR_PATH / 'data'
TESTBED_ROOT = DATA_ROOT / 'testbeds'
GRAPH_SPLIT_ROOT = DATA_ROOT / 'graph-splits'
META_FEAT_ROOT = DATA_ROOT / 'metafeats'
GRAPH_DATA_ROOT = DATA_ROOT / 'graph-data'
LINK_PRED_PERF_ROOT = DATA_ROOT / 'link-pred-perfs'
NODE_CLASS_PERF_ROOT = DATA_ROOT / 'node-class-perfs'
RESULTS_ROOT = ROOT_DIR_PATH / 'results'
WORKSPACE_ROOT = DATA_ROOT / 'workspace'
WORKSPACE_FILE = WORKSPACE_ROOT / 'workspace.pkl'
