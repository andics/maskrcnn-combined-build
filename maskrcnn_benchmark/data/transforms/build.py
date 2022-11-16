# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
import torchvision.transforms as transforms_pretrained_resnet

from maskrcnn_benchmark.modeling import registry


def build_transforms(cfg, is_train=True):
    assert cfg.DATALOADER.TRANSFORM_FUNCTION in registry.TRANSFORM_FUNCTIONS, \
        "DATALOADER.TRANSFORM_FUNCTION: {} are not registered in registry".format(
            cfg.DATALOADER.COCO_VERSION
        )
    print("DATALOADER.TRANSFORM_FUNCTION: using {} transform function for model".format(
        cfg.DATALOADER.TRANSFORM_FUNCTION))

    return registry.TRANSFORM_FUNCTIONS[cfg.DATALOADER.TRANSFORM_FUNCTION](cfg, is_train)

@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_baseline")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_equiconst")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_variable")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_single")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_multi_mixed")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_multi_stacked_single")
def build_transforms_original(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    return transform

@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_equiconst_pretrained_resnet")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_variable_pretrained_resnet")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_multi_stacked_single_pretrained_resnet")
def build_transforms_pretrained_resnet(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        is_test = True
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    #The mean pixel values for each color channel are reversed. Original MaskRCNN expected BGR
    #Devide by 255 because Resnet was trained with such input.
    #Normalization functions have to be the same.
    pixel_means = [value/255.0 for value in cfg.INPUT.PIXEL_MEAN]

    normalize_transform = T.Normalize(
        mean=pixel_means, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    '''
    normalize = transforms_pretrained_resnet.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = T.Compose([
            T.Resize(min_size, max_size),
            transforms_pretrained_resnet.RandomHorizontalFlip(p=flip_prob),
            transforms_pretrained_resnet.ToTensor(),
            normalize,
        ])
    '''

    return transform


@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_5_mod")
@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_multi_stacked")
def build_transforms_5_mod(cfg, is_train=True):
    is_test = False

    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        is_test = True
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    # DH: multi-channel
    # TRANSFORM_FUNCTIONS with horizontal flip
    transform_hflip = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(1.0),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    # TRANSFORM_FUNCTIONS without horizontal flip
    transform_noflip = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(0),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    # return a dictionary of all composed TRANSFORM_FUNCTIONS
    out_TRANSFORM_FUNCTIONS = {
        'img_transform': transform,
        'transform_hflip': transform_hflip,
        'transform_noflip': transform_noflip
    }


    return out_TRANSFORM_FUNCTIONS


@registry.TRANSFORM_FUNCTIONS.register("dataloader_transform_function_multi_stacked_pretrained_resnet")
def build_transforms_stacked_pretrained_resnet(cfg, is_train=True):
    is_test = False

    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        is_test = True
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    #The mean pixel values for each color channel are reversed. Original MaskRCNN expected BGR

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    # DH: multi-channel
    # TRANSFORM_FUNCTIONS with horizontal flip
    transform_hflip = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(1.0),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    # TRANSFORM_FUNCTIONS without horizontal flip
    transform_noflip = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(0),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    # return a dictionary of all composed TRANSFORM_FUNCTIONS
    out_TRANSFORM_FUNCTIONS = {
        'img_transform': transform,
        'transform_hflip': transform_hflip,
        'transform_noflip': transform_noflip
    }


    return out_TRANSFORM_FUNCTIONS