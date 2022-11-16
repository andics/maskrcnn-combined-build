# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import torch

from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.model_zoo import cache_url
from collections import OrderedDict

from utils_gen import model_utils as utils
from maskrcnn_benchmark.utils import registry

from maskrcnn_benchmark.modeling import registry


def load_target_weight_state_dict(target_weight_location, cfg):
    #The highest level function combining all of the weight loading mechanisms
    # E.g: Loading from Catalog, Physical weight, Web address ect.
    loaded_checkpoint = load_file(cfg, target_weight_location)
    loaded_checkpoint = loaded_checkpoint.pop("model")

    #Triggered only while loading the pretrained ResNet weights
    if 'state_dict' in loaded_checkpoint:
        loaded_checkpoint = loaded_checkpoint['state_dict']

    loaded_state_dict = strip_prefix_if_present(loaded_checkpoint, prefix="module.")

    return loaded_state_dict


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_file(cfg, f):
    #The function returns an un-popped dictionary containing a "model" element. In there are located the model weights

    logger = logging.getLogger(__name__)

    # catalog lookup
    if f.startswith("catalog://"):
        paths_catalog = import_file(
            "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])
        logger.info("{} points to {}".format(f, catalog_f))
        f = catalog_f
    # download url files
    if f.startswith("http"):
        # if the file is a url path, download it and cache it
        cached_f = cache_url(f)
        logger.info("url {} cached in {}".format(f, cached_f))
        f = cached_f
    # convert Caffe2 checkpoint from pkl
    if f.endswith(".pkl"):
        return load_c2_format(cfg, f)

    # load native detectron.pytorch checkpoint
    loaded = _load_file(f)
    if "model" not in loaded:
        loaded = dict(model=loaded)
    return loaded


def _load_file(f):
    return torch.load(f, map_location=torch.device("cpu"))

