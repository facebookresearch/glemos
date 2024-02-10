# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
from logging import Logger

logger = logging.getLogger("default-logger")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


def setup_logger(logger_name: str, log_file: Path, level=logging.INFO, log_format='short'):
    logger = logging.getLogger(logger_name)
    format = {
        'short': '%(asctime)s %(message)s',
        'long': '%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s'
    }[log_format]
    formatter = logging.Formatter(format)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(formatter)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger


def clear_log_handlers(logger: Logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
