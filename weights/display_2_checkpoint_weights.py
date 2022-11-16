import sys
sys.path.remove('/workspace/object_detection')
# DH: multi-channel
# adjusted path
sys.path.append('/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp')

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import logging
import functools
import tools.universal.model_utils as model_tools

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.tester import test
from demo.predictor_custom import COCO_predictor

def main():
    weight1_checkpoint = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/ch3_resnet101_ch3_equal/model_0015000.pth"
    weight2_checkpoint = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/ch3_resnet101_ch3_equal/model_0090000.pth"

    universal_config_file = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_multi_5_mod_comp/configs/e2e_mask_rcnn_R_101_FPN_1x_multi_ch.yaml"
    config_file = universal_config_file
    cfg.merge_from_file(config_file)

    print("Zdr bepce.")

    model_tools.setup_env_variables()
    model_predictor_15k = COCO_predictor(cfg = cfg, custom_config_file = universal_config_file, \
                                     weight_file_dir = weight1_checkpoint, \
                                     use_conf_threshold = False, max_num_pred = 5, min_image_size = 60, masks_per_dim = 3)
    model_predictor_90k = COCO_predictor(cfg = cfg, custom_config_file = universal_config_file, \
                                     weight_file_dir = weight2_checkpoint, \
                                     use_conf_threshold = False, max_num_pred = 5, min_image_size = 60, masks_per_dim = 3)

    weights_15k = model_predictor_15k.model.backbone.body._modules.get('stem')._modules.get('conv1')._parameters.get('weight').data
    weights_90k = model_predictor_90k.model.backbone.body._modules.get('stem')._modules.get('conv1')._parameters.get('weight').data

    print_model_weights_comparison(weights_15k, weights_90k)


def print_model_weights_comparison(weights_15k, weights_90k):
    print("Weight_1 at 0: ", weights_15k[0, 0, 0, 0])
    print("Weight_2 at 0: ", weights_90k[0, 0, 0, 0])


def load_model_weights(config_file, weight_file_path):

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Weight loading
    checkpointer = DetectronCheckpointer(cfg, model, save_dir = weight_file_path)
    _ = checkpointer.load(weight_file_path)

    return model



if __name__=="__main__":
    main()