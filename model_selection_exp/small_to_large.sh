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

python src/benchmark_main.py --task link-pred --testbed small-to-large --perf-metric map --meta-feat "${meta_feat}" --n-splits 1 "$@"

python src/benchmark_main.py --task node-class --testbed small-to-large --perf-metric map --meta-feat "${meta_feat}" --n-splits 1 --k-dim 24 "$@"

done