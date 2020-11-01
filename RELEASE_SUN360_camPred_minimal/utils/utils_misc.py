import itertools
from termcolor import colored
import torch.nn as nn
import torch
import numpy as np

def merge_list_of_lists(list_of_lists):
    merged = list(itertools.chain.from_iterable(list_of_lists))
    return merged

def batch_dict_to_list_of_dicts(batch_dict):
    list_of_dicts = []
    for idx in range(batch_dict['num_samples']):
        new_dict = {}
        for key in batch_dict:
            content = batch_dict[key]
            if key == 'num_samples':
                continue
            # print('--', idx, key, content)
            # if type(content) is dict:
            #     content = batch_dict_to_list_of_dicts(content)
            # print('----', idx, key, content)
            # if isinstance(content, list):
            #     print(len(content))
            # else:
            #     print(content.shape)
            # if isinstance(content, list):
            if key in ['yc_est_list', 'person_hs_est_list', 'vt_camEst_N_delta_est_list', 'f_pixels_est_mm_list', 'v0_est_list']:
                # if key == 'person_hs_est_list':
                # print('>>>', idx, key, content)
                new_dict[key] = [content_layer[idx] for content_layer in content] # index by layer to index by sample
                # if key == 'person_hs_est_list':
                    # print('>>>>>>>', idx,   new_dict[key])
            else:
                if key in [] or (isinstance(content, list) and not content):
                    # new_dict[key] = content[0]
                    pass
                else:
                    print('>>>', idx, key, content)
                    new_dict[key] = content[idx]
        list_of_dicts.append(new_dict)

    return list_of_dicts

def softmax_with_bins(input, bins):
    # input: [N, D], bins: [D]
    # return: [N]
    est_batch = (nn.functional.softmax(input, dim=1) * bins).sum(dim=1) #sum is reducing dims; [N]
    return est_batch

def argmax_with_bins(input, bins):
    # input: [N, D], bins: [D]
    # return: [N]
    idxx = torch.argmax(input, dim=1)
    est_batch = bins[idxx]
    return est_batch

def prob_to_est(input, bins, reduce_method='softmax', debug=False):
    if debug:
        print('[debug] prob_to_est', input.shape, bins.shape)
    if reduce_method == 'softmax':
        return softmax_with_bins(input, bins)
    elif reduce_method == 'argmax':
        return argmax_with_bins(input, bins)
    else:
        raise ValueError('reduce_method should be in [argmax, softmax]!')

def pad_zeros_to_good_num(bboxes, good_num):
    N = bboxes.shape[0]
    assert N <= good_num, 'not N %d <= good_num %d'%(N, good_num)
    N_to_pad = good_num - N
    assert len(bboxes.shape) in [1, 2]
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape((-1, 1))
    if N_to_pad > 0:
        return torch.cat((bboxes, torch.zeros((N_to_pad, bboxes.shape[1]), device=bboxes.device)))
    else:
        return bboxes

def list_of_bboxes_to_bboxes_padded(list_of_bbox, good_num, H_batch_array, normalize_with_H=True):
    bboxes_padded_list = []
    for bboxes, H in zip(list_of_bbox, H_batch_array):
        if normalize_with_H:
            bboxes = bboxes / H - 0.5
        if len(bboxes) == 1:
            bboxes = bboxes.reshape((-1, 1))
        bboxes_padded = pad_zeros_to_good_num(bboxes, good_num)
        bboxes_padded_list.append(bboxes_padded)
    return torch.stack(bboxes_padded_list)


def list_of_tensor_to_tensor_padded(list_of_delta_vt, good_num):
    bboxes_padded_list = []
    for bboxes in list_of_delta_vt:
        bboxes_padded = pad_zeros_to_good_num(bboxes, good_num)
        bboxes_padded_list.append(bboxes_padded)
    return torch.stack(bboxes_padded_list)

def bbox_list_to_list_of_bboxes(list_of_bbox_list):
    bboxes_list = []
    for boxlist in list_of_bbox_list:
        bboxes = boxlist.convert('xyxy').bbox
        bboxes_list.append(bboxes)
    return bboxes_list

def pad_bbox(bboxes, max_length):
    bboxes_padded = np.zeros((max_length, bboxes.shape[1]))
    assert bboxes.shape[0] <= max_length, 'bboxes length of %d > max_length of %d!'%(bboxes.shape[0], max_length)
    bboxes_padded[:bboxes.shape[0], :] = bboxes
    return bboxes_padded




def red(text):
    return colored(text, 'yellow', 'on_red')

def print_red(text):
    print(red(text))

def white_blue(text):
    coloredd = colored(text, 'white', 'on_blue')
    return coloredd

def print_white_blue(text):
    print(white_blue(text))

def green(text):
    coloredd = colored(text, 'blue', 'on_green')
    return coloredd

def print_green(text):
    print(green(text))

def magenta(text):
    coloredd = colored(text, 'white', 'on_magenta')
    return coloredd

def print_magenta(text):
    print(magenta(text))