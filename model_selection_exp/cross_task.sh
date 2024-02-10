#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/ || exit
cd .. || exit

for meta_feat in regular graphlets_complex compact regular_graphlets; do

python src/benchmark_main.py --testbed cross-task \
  --source-task link-pred --source-perf-metric map \
  --target-task node-class --target-perf-metric map \
  --meta-feat "${meta_feat}" "$@"

python src/benchmark_main.py --testbed cross-task \
  --source-task node-class --source-perf-metric map \
  --target-task link-pred --target-perf-metric map \
  --meta-feat "${meta_feat}" "$@"

done