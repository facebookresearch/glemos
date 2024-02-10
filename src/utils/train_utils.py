# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import os
import random

import numpy as np
import torch


# noinspection PyUnresolvedReferences
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


class EarlyStopping:
    def __init__(self,
                 patience=30,
                 minimizing_objective=False,
                 logging=True,
                 logger=None,
                 score_type='score'):
        self.patience = patience
        self.minimizing_objective = minimizing_objective
        self.counter = 0
        self.early_stop = False
        self.logging = logging
        if logger is None:
            from utils.log_utils import logger
            self.logger = logger
        else:
            self.logger = logger
        self.has_improved = None
        self.best_score = None
        self.best_model_state_dict = None
        self.score_type = score_type

    def step(self, score, model=None):
        """Return whether to early stop"""
        if self.best_score is None or self.improved(score, self.best_score):
            self.has_improved = True
            self.best_score = score
            # if self.logging:
            #     logger.info(f"[EarlyStopping-{self.score_type}] Best {self.score_type} updated to {self.best_score:.4f}")
            if model is not None:
                self.save_checkpoint(model)
            self.counter = 0
        else:
            self.has_improved = False
            self.counter += 1
            if self.logging:
                self.logger.info(f"[EarlyStopping] counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def improved(self, score, best_score):
        if self.minimizing_objective:
            return True if score < best_score else False
        else:
            return True if score > best_score else False

    def save_checkpoint(self, model):
        self.best_model_state_dict = copy.deepcopy(model.state_dict())

    def load_checkpoint(self, model):
        model.load_state_dict(self.best_model_state_dict)
