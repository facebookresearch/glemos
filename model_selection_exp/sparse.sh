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

for sparsity in 0.1 0.3 0.5 0.7 0.9; do
  echo "sparsity=${sparsity}"
  python src/benchmark_main.py --task link-pred --testbed partially-observed --perf-sparsity "${sparsity}" \
    --perf-metric map --meta-feat "${meta_feat}" "$@"
done

for sparsity in 0.1 0.3 0.5 0.7 0.9; do
  echo "sparsity=${sparsity}"
  python src/benchmark_main.py --task node-class --testbed partially-observed --perf-sparsity "${sparsity}" \
    --perf-metric map --meta-feat "${meta_feat}" "$@"
done

done