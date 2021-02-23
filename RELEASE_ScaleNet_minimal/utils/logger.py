# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # don't log results for the non-master process
    if distributed_rank > 0:
        logger.setLevel(logging.ERROR)
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    # if distributed_rank > 0:
    #     ch.setLevel(logging.WARNING)
    # else:
    ch.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(name)s %(levelname)s: %(message)s")

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename.replace('.txt', '_rank%d.txt'%distributed_rank)))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class printer():
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, distributed_rank, debug=False):
        self.distributed_rank = distributed_rank
        self.debug = debug

    def print(self, *params):
        if self.distributed_rank == 0 and self.debug:
            print(params)