# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import torch

from maskrcnn_benchmark.utils.imports import import_file
from utils_gen import model_utils as utils
from utils_gen import checkpoint_utils
from maskrcnn_benchmark.utils import registry
from collections import OrderedDict

from maskrcnn_benchmark.modeling import registry


def run(cfg, model):
    #Config file format: [model_layer_name, model_layer_location, target_layer_name, target_layer_location, target_checkpoint location]
    #E.g: ["backbone.stem.conv1", "[64, 0:3, 7, 7]", "conv1", "[:]", "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/pretrained_models/resnet_baseline/checkpoint_090.pth.tar"]

    #Administrative work: setup logger
    logger = logging.getLogger(__name__)

    model_state_dict = model.state_dict()

    #Check if model is wrapped in a DDP
    if hasattr(model, 'module'):
        ddp_wrapped=True
    else:
        ddp_wrapped=False


    for custom_weight_sequence in cfg.MODEL.SERIALIZATION_SEQUENCE_CUSTOM:
        target_state_dict = checkpoint_utils.load_target_weight_state_dict(target_weight_location = custom_weight_sequence[-1],
                                                                           cfg = cfg)
        exec("model_state_dict" + "[" + custom_weight_sequence[0] + "]" + custom_weight_sequence[1] +
             " = " + "target_state_dict" + "[" + custom_weight_sequence[2] + "]" + custom_weight_sequence[3])
        logger.info("MODEL.SERIALIZATION_SEQUENCE_CUSTOM: Loaded {0} from {1} (path: {2})".format(custom_weight_sequence[0].replace('"', '') + custom_weight_sequence[1],
                                                                                      custom_weight_sequence[2].replace('"', '') + custom_weight_sequence[3], custom_weight_sequence[-1]))

    model.load_state_dict(model_state_dict)
    logger.info("MODEL.SERIALIZATION_SEQUENCE_CUSTOM: Loaded all custom_sequence weights!")

