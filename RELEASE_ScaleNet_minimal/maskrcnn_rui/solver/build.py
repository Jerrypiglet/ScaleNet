# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR
from termcolor import colored


def make_optimizer(cfg, model, optim_type='SGD', params_dict= {}, logger=None):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    logger.info(colored('Creating %s solver with lr=%.4f, weight_decay=%.4f...'%(optim_type, lr, weight_decay), 'white', 'on_blue'))
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr, betas=(params_dict['beta1'], 0.999), eps=1e-5)
    else:
        raise RuntimeError('Optimizer type %s not supported! (SGD/Adam)'%optim_type)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
