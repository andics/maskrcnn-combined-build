from utils_gen import model_utils
from pathlib import Path
from maskrcnn_benchmark.utils.registry import Registry

import os
import shutil
import copy

global default_configs_location, base_config_location_path, config_file_ext, main_path
main_path = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
default_configs_location = os.path.join(main_path, "configs/default_configs")
base_config_location_path = os.path.join(main_path, "configs")
config_file_ext = ".yaml"

#-------------------CONFIG-FILE-PREP-------------------
def prepare_new_config_file_from_sequence(cfg, model_sequence, model_specific_sequence):
    '''
    :param cfg:
    :param model_sequence:
    :param model_specific_sequence:
    :return:
    Searches for a config file named using the following paradigm:
        new_config_file = configs / R-101-FPN (CONV_BODY) / model_sequence / model_sequence + model_specific_sequence (if any) + .yaml
    If a config file is found, its path is returned.

    If a config file is not found, it is created through the
    following paradigm and returned:
        - The default config file for the model_sequence is taken:
            - default_file = default_configs_location + model_sequence
        - The default_file is copied to new_config_file
        - Depending on the model architecture, the copied new_config_file is modified in the following way:
            - Default modification: baseline; equiconst; variable; multi_stacked_single; multi_stacked; multi_mixed
                OUTPUT_DIR = trained_models/ + model_sequence + / + model_specific_sequence (if any)
                MODEL.SPECIFIC_SEQUENCE = model_specific_sequence
            - 5_mod_modification: 5_mod
                OUTPUT_DIR = trained_models/ + model_sequence + / + model_specific_sequence
                MODEL.BACKBONE.MODEL_SEQUENCE_5_MOD = model_specific_sequence
                MODEL.RESNETS.COMB_OUTPUT_CHANNELS_5_MOD = 320 || 64 (depending on model_specific_sequence - _cat_ || _add_)
            - single_modification: single
                OUTPUT_DIR = trained_models/ + model_sequence + / + model_specific_sequence
                DATASETS.TEST = ch1 || ch2 || ch3 (depending on model_specific_sequence - ch1 || ch2 || ch3)
                DATASETS.TRAIN = ch1 || ch2 || ch3 (depending on model_specific_sequence - ch1 || ch2 || ch3)

    NOTES:
     - If a model_specific_sequence is provided for an architecture which does not have customization implemented,
    no problem is triggered and the model_specific_sequence is only used for a different OUTPUT_DIR and a differently named
    config_file.
    '''

    if model_sequence in _CFG_DEFAULT_FILES:
        print("Model sequence successfully verified:", model_sequence)
        default_cfg_for_model = _CFG_DEFAULT_FILES[model_sequence]
    else:
        print("Model sequence was NOT verified! Using default baseline config file.")
        default_cfg_for_model = _CFG_DEFAULT_FILES["baseline"]
        model_sequence = "baseline"

    default_config_path = default_configs_location + "/" + default_cfg_for_model + config_file_ext

    _cfg_temp = copy.deepcopy(cfg)
    _cfg_temp.merge_from_file(default_config_path)

    _sub_dir = _cfg_temp.MODEL.BACKBONE.CONV_BODY

    new_config_dir_path = os.path.join(base_config_location_path, _sub_dir, model_sequence)
    Path(new_config_dir_path).mkdir(parents=True, exist_ok=True)

    if model_specific_sequence is "Nothing":
        new_config_file_path = new_config_dir_path + "/" + model_sequence + config_file_ext
    else:
        new_config_file_path = new_config_dir_path + "/" + model_sequence + "_" + model_specific_sequence + config_file_ext

    if not os.path.isfile(new_config_file_path):
        #Create the config file ONLY if it doesn't exist
        #Also, edit its properties only under the same condition
        shutil.copy(default_config_path, new_config_file_path)
        change_custom_config_file_properties(new_config_file_path, model_sequence, model_specific_sequence)
    else:
        print("Found model-specific config file under: ", new_config_file_path)

    return new_config_file_path


def change_custom_config_file_properties(config_file_path, model_sequence, model_specific_sequence):

    current_project_base_path = Path(__file__).parents[1]

    project_save_path = os.path.join(current_project_base_path, "trained_models", model_sequence)

    if model_specific_sequence is "Nothing":
        #Just edit the output dir and model type, leaving everything else as it was in the default_config_file
        Path(os.path.join(project_save_path, "last_checkpoint")).mkdir(parents=True, exist_ok=True)
        model_utils.change_yaml_file_value(config_file_path, ['OUTPUT_DIR'], project_save_path)
        return
    else:
        project_save_path = os.path.join(project_save_path, model_specific_sequence)
        Path(os.path.join(project_save_path, "last_checkpoint")).mkdir(parents=True, exist_ok=True)
        model_utils.change_yaml_file_value(config_file_path, ['OUTPUT_DIR'], project_save_path)
        model_utils.change_yaml_file_value(config_file_path, ['MODEL', 'SPECIFIC_SEQUENCE'], model_specific_sequence)

        #Check if any special config modifications are implemented for the particular architecture type
        if model_sequence in _CFG_CUSTOMIZATION_FUNCTIONS:
            _CFG_CUSTOMIZATION_FUNCTIONS[model_sequence](config_file_path, model_sequence, model_specific_sequence)



def customize_single_config_file(config_file_path, model_sequence, model_specific_sequence):

    if model_specific_sequence in _SINGLE_CHANNEL_MODEL_TRAIN_SETS:
        model_utils.change_yaml_file_value(config_file_path, ['DATASETS', 'TRAIN'],
                                           (_SINGLE_CHANNEL_MODEL_TRAIN_SETS[model_specific_sequence],))

    if model_specific_sequence in _SINGLE_CHANNEL_MODEL_TEST_SETS:
        model_utils.change_yaml_file_value(config_file_path, ['DATASETS', 'TEST'],
                                           (_SINGLE_CHANNEL_MODEL_TEST_SETS[model_specific_sequence],))


'''
def customize_5_mod_config_file(config_file_path, model_sequence, model_specific_sequence):
    # Modify the 5 mod specific sequence in the config file
    # The train and test datasets are expected to be taken care of by the default config file for the 5_mod
    # If a model_specific_sequence is provided, which contains unrecognized parts, no problem will arise as long
    # as the basic operations of the architectures are defined in the sequence.
    # E.g:
    # conv_bn1_relu1_cat_bn2_relu2_comp will perform the same way as conv_bn1_relu1_cat_bn2_relu2_comp_test
    # Only difference will be in the name of the output directories and the config files.

    after_cat_expected_number_of_channels = 320
    after_add_expected_number_of_channels = 64

    if "_cat_" in model_specific_sequence:
        _val_to_use_ = after_cat_expected_number_of_channels
    elif "_add_" in model_specific_sequence:
        _val_to_use = after_add_expected_number_of_channels

    model_utils.change_yaml_file_value(config_file_path,
                           ['MODEL', 'RESNETS', 'COMB_OUTPUT_CHANNELS_5_MOD'],
                           _val_to_use_)
    
    model_utils.change_yaml_file_value(config_file_path,
                           ['MODEL', 'RESNETS', 'STEM_FUNC'],
                           '5_mod_base_stem')
'''

# -------------------CONFIG-FILE-PREP-------------------



_CFG_DEFAULT_FILES = Registry({
    "baseline": "e2e_maskrcnn_R_101_FPN_1x_baseline",
    "equiconst": "e2e_maskrcnn_R_101_FPN_1x_equiconst",
    "equiconst_pretrained_resnet": "e2e_maskrcnn_R_101_FPN_1x_equiconst_pretrained_resnet",
    "variable": "e2e_maskrcnn_R_101_FPN_1x_variable",
    "variable_pretrained_resnet": "e2e_maskrcnn_R_101_FPN_1x_variable_pretrained_resnet",
    "multi_mixed": "e2e_maskrcnn_R_101_FPN_1x_multi_mixed",
    "single": "e2e_maskrcnn_R_101_FPN_1x_single",
    "single_isolated_norm": "e2e_maskrcnn_R_101_FPN_1x_single_isolated_norm",
    "multi_stacked": "e2e_maskrcnn_R_101_FPN_1x_multi_stacked",
    "multi_stacked_pretrained_resnet": "e2e_maskrcnn_R_101_FPN_1x_multi_stacked_pretrained_resnet",
    "multi_stacked_single": "e2e_maskrcnn_R_101_FPN_1x_multi_stacked_single",
    "multi_stacked_single_pretrained_resnet": "e2e_maskrcnn_R_101_FPN_1x_multi_stacked_single_pretrained_resnet",
    "5_mod": "e2e_maskrcnn_R_101_FPN_1x_5_mod",
})

_CFG_CUSTOMIZATION_FUNCTIONS = Registry({
    "single": customize_single_config_file,
})

_SINGLE_CHANNEL_MODEL_TRAIN_SETS = Registry({
    "ch1": "coco_2017_ch1_train",
    "ch2": "coco_2017_ch2_train",
    "ch3": "coco_2017_ch3_train",
})

_SINGLE_CHANNEL_MODEL_TEST_SETS = Registry({
    "ch1": "coco_2017_ch1_val",
    "ch2": "coco_2017_ch2_val",
    "ch3": "coco_2017_ch3_val",
})

