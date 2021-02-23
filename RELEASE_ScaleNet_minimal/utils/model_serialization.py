# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch
from termcolor import colored
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
# from maskrcnn_benchmark.utils.imports import import_file
from utils.utils_misc import *


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=None, skip_kws=[], only_load_kws=[], replace_kws=[], replace_with_kws=[]):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    rank = get_rank()

    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    current_keys_ori = current_keys.copy()
    loaded_keys_ori = loaded_keys.copy()

    current_keys_renamed = current_keys.copy()
    if not (not replace_kws or not replace_with_kws):
        print('>> replace_kws:', replace_kws, '>> replace_with_kws:', replace_with_kws)
        assert len(replace_kws) == len(replace_with_kws) and len(replace_kws) > 0, 'Length of replace_kws %d and replace_with_kws %d should equal and > 0!'%(len(replace_kws), len(replace_with_kws))
        # loaded_keys = [loaded_keys.replace('roi_heads.box', 'classifiers_head.head') for loaded_keys in loaded_keys]

        current_keys_filtered = []
        for current_key in current_keys:
            if_replace = False
            for replace_kw, replace_with_kw in zip(replace_kws, replace_with_kws):
                if replace_kw in current_key:
                    # print('--Rename', current_key, '--->', current_key.replace(replace_kw, replace_with_kw))
                    current_keys_filtered.append(current_key.replace(replace_kw, replace_with_kw))
                    if_replace = True
                    break
            if if_replace == False:
                current_keys_filtered.append(current_key)
            # current_keys = [current_keys.replace(replace_kw, replace_with_kw) for loaded_keys in loaded_keys]
        current_keys_renamed = current_keys_filtered.copy()


    logger.warning('====== current_keys %d; loaded keys %d'%(len(current_keys), len(loaded_keys)))
    # for a in sorted(current_keys_renamed):
    #     print('-', a)
    # for a in sorted(loaded_keys):
    #     print('==', a)
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.replace('.layers', '').replace('RCNN.', '').endswith(j.replace('RCNN.', '').replace('.layers', '')) else 0 for i in current_keys_renamed for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys_renamed]) if current_keys_renamed else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} <<LOADED FROM<< {: <{}} of shape {}"
    # logger = logging.getLogger(__name__)
    # print(logger)
    all_possible_loads = 0
    success_loads = 0
    for idx_new, idx_old in enumerate(idxs.tolist()):
        all_possible_loads += 1
        if idx_old == -1:
            continue
        key = current_keys[idx_new]

        break_flag = False
        for skip_kw in skip_kws:
            if skip_kw in key:
                # logger.warning(colored('====!!!!==== Skipping %s for in *skip_kws*'%key, 'cyan', 'on_blue'))
                break_flag = True
        if break_flag:
            continue

        if only_load_kws:
            at_least_one_in_flag = False
            for only_load_kw in only_load_kws:
                if only_load_kw in key:
                    at_least_one_in_flag = True
            if at_least_one_in_flag == False:
                # logger.warning(colored('====!!!!==== Skipping %s for NOT in *only_load_kws*'%key, 'cyan', 'on_blue'))
                continue

        key_old = loaded_keys_ori[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]

        if rank == 0:
            logger.warning(
                log_str_template.format(
                    key,
                    max_size,
                    key_old,
                    max_size_loaded,
                    tuple(loaded_state_dict[key_old].shape),
                )
            )
        success_loads += 1

    if success_loads == all_possible_loads:
        logger.warning(white_blue('====== Successfully loaded %d from %d possible loads.'%(success_loads, all_possible_loads)))
    else:
        logger.warning(red('====== Successfully loaded %d from %d possible loads.'%(success_loads, all_possible_loads)))
    return current_keys_ori , loaded_keys_ori



def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, logger=None, skip_kws=[], only_load_kws=[], replace_kws=[], replace_with_kws=[]):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    current_keys, loaded_keys = align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger, skip_kws=skip_kws, only_load_kws=only_load_kws, replace_kws=replace_kws, replace_with_kws=replace_with_kws)

    # use strict loading
    model.load_state_dict(model_state_dict)
    return current_keys, loaded_keys
