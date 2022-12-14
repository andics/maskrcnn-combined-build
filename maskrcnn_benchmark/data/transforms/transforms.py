# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


global total_mean_tensor_value; total_mean_tensor_value = 0

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.total_mean_tensor_value = 0

    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                image = t(image)

            self.total_mean_tensor_value += image.numpy().mean()
            return image
        else:
            for t in self.transforms:
                image, target = t(image, target)

            self.total_mean_tensor_value += image.numpy().mean()
            return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


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

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        else:
            target = target.resize(image.size)
            return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if target is None:
            if random.random() < self.prob:
                image = F.hflip(image)
            return image
        else:
            if random.random() < self.prob:
                image = F.hflip(image)
                target = target.transpose(0)
            return image, target

class ToTensor(object):
    def __call__(self, image, target=None):
        if target is None:
            return F.to_tensor(image)
        else:
            return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            #print("Image shape: ", image.shape)
            if list(image.shape)[0] < 3:
                image = image.repeat(3, 1, 1)
                print("Converted image to RGB!")
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        else:
            return image, target