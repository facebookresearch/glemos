#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/ || exit

CONDA_ENV=GLEMOS
if conda info --envs | grep -q "${CONDA_ENV} "; then
  echo "\"${CONDA_ENV}\" conda env exists.";
else
  conda create -y --name "${CONDA_ENV}" python=3.8
fi

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}"/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# we use pytorch-2.0.0
if [[ "${OSTYPE}" == "darwin"* ]]; then  # Mac OS
  conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch  # PyTorch for Mac
#  conda install -y pytorch::pytorch==2.0.0 torchvision torchaudio -c pytorch  # PyTorch for Mac
else
  conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia  # PyTorch for Linux
#  conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # PyTorch for Linux
fi

# we use pyg-2.3.0
conda install -y pyg==2.3.0 -c pyg  # PyG for Linux and Mac
conda install -y pytorch-sparse -c pyg
#conda install -y pytorch-cluster -c pyg  # used by PyG Node2Vec

conda install -y -c conda-forge rdflib  # for processing PyG Entities dataset
conda install -y pandas==1.5.3
conda install -y -c dglteam dgl=0.8.2  # used by metagl and ncf

# install other packages via pip
pip install grape
pip install threadpoolctl