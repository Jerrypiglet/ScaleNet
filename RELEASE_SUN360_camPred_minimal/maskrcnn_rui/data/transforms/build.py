# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from torchvision import transforms as T_torchvision
from torchvision.transforms import functional as F

def build_transforms_maskrcnn(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        # flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        # flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        # flip_horizontal_prob = 0.0
        # flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            # T.RandomHorizontalFlip(flip_horizontal_prob),
            # T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

def build_transform_maskrcnnPredictor(cfg):
    """
    Creates a basic transformation that was used to train the models
    """

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T_torchvision.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T_torchvision.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T_torchvision.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T_torchvision.Compose(
        [
            T_torchvision.ToPILImage(),
            Resize(min_size, max_size),
            T_torchvision.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

from torchvision import datasets, models, transforms
import torch
import numpy as np

class RandomSaturation:
    def __call__(self, sample):
        if np.random.rand() < 0.75:
            saturation_amt = np.random.triangular(-1, 0, 1)
            if np.random.rand() < 0.04: # Grayscale
                saturation_amt = 1
            im = sample[0]
            im = torch.clamp(im + (torch.max(im, 0)[0] - im) * saturation_amt, 0, 1)
            sample[0] = im
        return sample


perturb = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

def build_transforms_yannick(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        # flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        # flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        # brightness = cfg.INPUT.BRIGHTNESS
        # contrast = cfg.INPUT.CONTRAST
        # saturation = cfg.INPUT.SATURATION
        # hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        # flip_horizontal_prob = 0.0
        # flip_vertical_prob = 0.0
        # brightness = 0.0
        # contrast = 0.0
        # saturation = 0.0
        # hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    if is_train:
        transform = T.Compose(
            [
                perturb,
                T.Resize(min_size, max_size),
                T.ToTensor(),
                RandomSaturation(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
