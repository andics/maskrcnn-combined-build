# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch
from utils_gen import model_utils as utils
from maskrcnn_benchmark.utils import model_serialization_sequence_custom
from maskrcnn_benchmark.utils import registry
from maskrcnn_benchmark.modeling import registry


def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
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
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # loaded_keys: the model weights from the catalog/checkpoint
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]

    # AG: Count how many keys from the model's weight dictionary don't have a corresponding weight tensor in the catalog/checkpoint dict
    missing_keys = match_matrix.count(0)
    print("CUSTOM_INFO: {} model keys are missing weights!".format(missing_keys))

    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(cfg, model, loaded_state_dict):
    # The items in the registry match the model TYPES names. That being said, there is a parameter
    # WEIGHT.SERIALIZATION_FUNCTION, which specifies which serialization function should be used

    assert cfg.MODEL.SERIALIZATION_FUNCTION in registry.SERIALIZATION_FUNCTIONS, \
        "MODEL.SERIALIZATION_FUNCTION: {} are not registered in registry".format(
            cfg.SERIALIZATION_FUNCTION
        )
    print("MODEL.SERIALIZATION_FUNCTION: using {} serialization function for model".format(
        cfg.MODEL.SERIALIZATION_FUNCTION))
    registry.SERIALIZATION_FUNCTIONS[cfg.MODEL.SERIALIZATION_FUNCTION](cfg, model, loaded_state_dict)
    if cfg.MODEL.SERIALIZATION_SEQUENCE_CUSTOM[0] is not "Nothing" and not utils.check_has_checkpoint(cfg.OUTPUT_DIR):
        print("MODEL.SERIALIZATION_SEQUENCE_CUSTOM: Triggering custom weight-loading sequence: model checked to have no checkpoint")
        model_serialization_sequence_custom.run(cfg, model)


@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_baseline")
@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_equiconst")
@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_variable")
@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_single")
@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_multi_mixed")
@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_multi_stacked_single")
def original_load_state_dict(cfg, model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching

    #Specific for models which use a custom pretrained ResNet.
    #Only the state dict needs be extracted from the checkpoint
    if "_resnet" in cfg.MODEL.SEQUENCE:
        #Triggered only while loading the pretrained ResNet
        if 'state_dict' in loaded_state_dict:
            loaded_state_dict = loaded_state_dict['state_dict']

    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    print("Loading model weights...")

    # use strict loading
    model.load_state_dict(model_state_dict)


@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_multi_stacked")
def multi_stacked_load_state_dict(cfg, model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    #Specific for models which use a custom pretrained ResNet.
    #Only the state dict needs be extracted from the checkpoint
    if "_resnet" in cfg.MODEL.SEQUENCE:
        #Triggered only while loading the pretrained ResNet
        if 'state_dict' in loaded_state_dict:
            print("Detected custom ResNet pretrained weights. Using appropriate load function...")
            loaded_state_dict = loaded_state_dict['state_dict']

    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # DH: multi-channel
    # adjust (replicate) the loaded weights for the stem conv1 layer to match the number of input channels
    # first check if in DataParallel mode
    if hasattr(model, 'module'):
        if type(model.module._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get(
                'conv1')) is not 'NoneType':
            if model.module._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get(
                    'conv1').in_channels > 3:
                curr_weight = model_state_dict.get('module.backbone.body.stem.conv1.weight')
                rep_num = model.module._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get(
                    'conv1').in_channels / curr_weight.shape[1]
                new_weight = curr_weight.repeat(1, rep_num.__int__(), 1, 1)
                model_state_dict['module.backbone.body.stem.conv1.weight'] = new_weight
                print("Replicated model weights for multi channel input!")
    else:
        if type(model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get(
                'conv1')) is not 'NoneType':
            if model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get(
                    'conv1').in_channels > 3:
                curr_weight = model_state_dict.get('backbone.body.stem.conv1.weight')
                rep_num = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get(
                    'conv1').in_channels / curr_weight.shape[1]
                new_weight = curr_weight.repeat(1, rep_num.__int__(), 1, 1)
                model_state_dict['backbone.body.stem.conv1.weight'] = new_weight
                print("Replicated model weights for multi channel input!")

    # use strict loading
    model.load_state_dict(model_state_dict)


@registry.SERIALIZATION_FUNCTIONS.register("serialization_function_5_mod")
def load_state_dict_5_mod(cfg, model, loaded_state_dict):
    # loaded_state_dict - from the checkpoint or catalog
    model_specific_sequence = cfg.MODEL.SPECIFIC_SEQUENCE

    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")

    align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    # AG: copies the values of the state_dict tensors from the checkpoint to the state_dict of the model
    # for all weights contained both in the checkpoint and in the model
    # aka: the newly added architecture-modules are not part of the checkpoint dictionary (at least not the first checkpoint)
    # and therefore need to be added manually

    # torch.cuda.set_device(os.environ['CUDA_VISIBLE_DEVICES'])

    if loaded_state_dict.get('conv1.weight') is not None:
        # This if checks if the loaded chekpoint is the original one or one generated during training

        if hasattr(model, 'module'):
            # Gets called when traing parallel!

            conv1_weight = loaded_state_dict.get('conv1.weight')
            conv1_bias = loaded_state_dict.get('conv1.bias')

            # For PURE modules
            # rep_num_pure = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('pure_conv_ch1').in_channels / conv1_weight.shape[1]
            rep_num_pure = 1
            new_weight_pure = conv1_weight.repeat(1, rep_num_pure.__int__(), 1, 1)

            model_state_dict['module.backbone.body.stem.pure_conv_ch1.weight'] = new_weight_pure
            model_state_dict['module.backbone.body.stem.pure_conv_ch2.weight'] = new_weight_pure
            model_state_dict['module.backbone.body.stem.pure_conv_ch3.weight'] = new_weight_pure

            # For MIXED modules - 2 Channel-multi support only
            # rep_num_mixed = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('mixed_conv_ch1_ch2').in_channels / conv1_weight.shape[1]
            rep_num_mixed = 2
            new_weight_mixed = conv1_weight.repeat(1, rep_num_mixed.__int__(), 1, 1)

            model_state_dict['module.backbone.body.stem.mixed_conv_ch1_ch2.weight'] = new_weight_mixed
            model_state_dict['module.backbone.body.stem.mixed_conv_ch2_ch3.weight'] = new_weight_mixed

            # For bn1 (batch norm), bn2 and bn3
            bn1_old_weights = loaded_state_dict.get('bn1.weight')
            bn1_old_biases = loaded_state_dict.get('bn1.bias')

            if utils.network_chunk_activator("bn1", model_specific_sequence):
                model_state_dict['module.backbone.body.stem.pure_ch1_bn1.weight'] = bn1_old_weights
                model_state_dict['module.backbone.body.stem.pure_ch2_bn1.weight'] = bn1_old_weights
                model_state_dict['module.backbone.body.stem.pure_ch3_bn1.weight'] = bn1_old_weights
                model_state_dict['module.backbone.body.stem.mixed_ch1_ch2_bn1.weight'] = bn1_old_weights
                model_state_dict['module.backbone.body.stem.mixed_ch2_ch3_bn1.weight'] = bn1_old_weights

                model_state_dict['module.backbone.body.stem.pure_ch3_bn1.bias'] = bn1_old_biases
                model_state_dict['module.backbone.body.stem.pure_ch1_bn1.bias'] = bn1_old_biases
                model_state_dict['module.backbone.body.stem.pure_ch2_bn1.bias'] = bn1_old_biases
                model_state_dict['module.backbone.body.stem.mixed_ch1_ch2_bn1.bias'] = bn1_old_biases
                model_state_dict['module.backbone.body.stem.mixed_ch2_ch3_bn1.bias'] = bn1_old_biases

            # batch-norm after cat/add
            if utils.network_chunk_activator("bn2", model_specific_sequence):
                rep_num_bn2 = model._modules.get('backbone')._modules.get('body')._modules.get(
                    'stem')._modules.get('output_compressor').in_channels / bn1_old_weights.shape[
                                  0]  # Value = 320/64 or 64/64 for cat and add
                model_state_dict['module.backbone.body.stem.bn2.weight'] = bn1_old_weights.repeat(rep_num_bn2.__int__())

                model_state_dict['module.backbone.body.stem.bn2.bias'] = bn1_old_biases.repeat(rep_num_bn2.__int__())
            # Batch-norm after output compressor

            rep_num_bn3 = model.module._modules.get('backbone')._modules.get('body')._modules.get(
                'stem')._modules.get('output_compressor').out_channels / bn1_old_weights.shape[0]  # Value = 64/64
            model_state_dict['module.backbone.body.stem.bn3.weight'] = bn1_old_weights.repeat(rep_num_bn3.__int__())

            model_state_dict['module.backbone.body.stem.bn3.bias'] = bn1_old_biases.repeat(rep_num_bn3.__int__())

            # We are not assigning conv1 weights to output_compressor because it doesn't make sense -
            # The conv1 weights expect to receive an image not a featuremap

            # With output compressor
            # This assigns weight to the compressing convolution equal the the original conv1 layer but copied multiple times
            # model_state_dict['module.backbone.body.stem.output_compressor.weight'] = torch.cat((conv1_weight.repeat(1, 106, 1, 1), conv1_weight[:, 0:2, :, :]), 1)
            # conv1_weight_for_comp = conv1_weight[:, :, 0, 0][:, :, None, None]
            # model_state_dict['module.backbone.body.stem.output_compressor.weight'] = torch.cat((conv1_weight_for_comp.repeat(1, 106, 1, 1), conv1_weight_for_comp[:, 0:2, :, :]), 1)

            # model_state_dict['module.backbone.body.stem.bn2.weight'] = bn1_old_weights
            # model_state_dict['module.backbone.body.stem.bn2.bias'] = bn1_old_biases
            # Putting the old weights because the 2-nd normalization works with only 64 channels
            # ---------------------------

        else:
            # Gets called when training single GPU

            conv1_weight = loaded_state_dict.get('conv1.weight')
            conv1_bias = loaded_state_dict.get('conv1.bias')

            # For PURE modules
            # rep_num_pure = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('pure_conv_ch1').in_channels / conv1_weight.shape[1]
            rep_num_pure = 1
            new_weight_pure = conv1_weight.repeat(1, rep_num_pure.__int__(), 1, 1)

            model_state_dict['backbone.body.stem.pure_conv_ch1.weight'] = new_weight_pure
            model_state_dict['backbone.body.stem.pure_conv_ch2.weight'] = new_weight_pure
            model_state_dict['backbone.body.stem.pure_conv_ch3.weight'] = new_weight_pure

            # For MIXED modules - 2 Channel-multi support only
            # rep_num_mixed = model._modules.get('backbone')._modules.get('body')._modules.get('stem')._modules.get('mixed_conv_ch1_ch2').in_channels / conv1_weight.shape[1]
            rep_num_mixed = 2
            new_weight_mixed = conv1_weight.repeat(1, rep_num_mixed.__int__(), 1, 1)

            model_state_dict['backbone.body.stem.mixed_conv_ch1_ch2.weight'] = new_weight_mixed
            model_state_dict['backbone.body.stem.mixed_conv_ch2_ch3.weight'] = new_weight_mixed

            # For bn1 (batch norm), bn2 and bn3
            bn1_old_weights = loaded_state_dict.get('bn1.weight')
            bn1_old_biases = loaded_state_dict.get('bn1.bias')

            if utils.network_chunk_activator("bn1", model_specific_sequence):
                model_state_dict['backbone.body.stem.pure_ch1_bn1.weight'] = bn1_old_weights
                model_state_dict['backbone.body.stem.pure_ch2_bn1.weight'] = bn1_old_weights
                model_state_dict['backbone.body.stem.pure_ch3_bn1.weight'] = bn1_old_weights
                model_state_dict['backbone.body.stem.mixed_ch1_ch2_bn1.weight'] = bn1_old_weights
                model_state_dict['backbone.body.stem.mixed_ch2_ch3_bn1.weight'] = bn1_old_weights

                model_state_dict['backbone.body.stem.pure_ch3_bn1.bias'] = bn1_old_biases
                model_state_dict['backbone.body.stem.pure_ch1_bn1.bias'] = bn1_old_biases
                model_state_dict['backbone.body.stem.pure_ch2_bn1.bias'] = bn1_old_biases
                model_state_dict['backbone.body.stem.mixed_ch1_ch2_bn1.bias'] = bn1_old_biases
                model_state_dict['backbone.body.stem.mixed_ch2_ch3_bn1.bias'] = bn1_old_biases

            # batch-norm after cat/add
            if utils.network_chunk_activator("bn2", model_specific_sequence):
                rep_num_bn2 = model._modules.get('backbone')._modules.get('body')._modules.get(
                    'stem')._modules.get('output_compressor').in_channels / bn1_old_weights.shape[
                                  0]  # Value = 320/64 or 64/64 for cat and add
                model_state_dict['backbone.body.stem.bn2.weight'] = bn1_old_weights.repeat(rep_num_bn2.__int__())

                model_state_dict['backbone.body.stem.bn2.bias'] = bn1_old_biases.repeat(rep_num_bn2.__int__())
            # Batch-norm after output compressor

            rep_num_bn3 = model._modules.get('backbone')._modules.get('body')._modules.get(
                'stem')._modules.get('output_compressor').out_channels / bn1_old_weights.shape[0]  # Value = 64/64
            model_state_dict['backbone.body.stem.bn3.weight'] = bn1_old_weights.repeat(rep_num_bn3.__int__())

            model_state_dict['backbone.body.stem.bn3.bias'] = bn1_old_biases.repeat(rep_num_bn3.__int__())

    # use strict loading
    model.load_state_dict(model_state_dict)
    # ?? - Doesn't this result in infinite iteration
