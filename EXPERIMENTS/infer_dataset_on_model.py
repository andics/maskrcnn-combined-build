# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

#-m torch.distributed.launch --nproc_per_node=2

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys
import os
import subprocess
from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")


import argparse
import logging
import functools
import gc
import time

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
from EXPERIMENTS.predictor_custom import COCO_predictor

from maskrcnn_benchmark.utils.logger import format_step
import training.cfg_prep as cfg_prep
import utils_gen.model_utils as utils

import dllogger
from maskrcnn_benchmark.utils.logger import format_step

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    print('Failed to import AMP')
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    print('Failed to import AMP Distributed')

global dllogger_initialized; dllogger_initialized = False

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--model-sequence",
        help="Specify neural network type",
        action="store",
    )
    parser.add_argument(
        "--model-specific-sequence",
        help="For particular models, allows real-time neural network building",
        action="store",
        default="Nothing",
    )
    parser.add_argument(
        "--config-path",
        help="Specify a config file",
        default="Nothing",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    #-----UNUSED-BUT-NECESSARY-----
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
    #------------------------------

    global dllogger_initialized
    skip_test = False
    json_summary_file_name = "dllogger_inference.out"

    args = parser.parse_args()
    print("Working with the following arguments: ", args)
    model_sequence = args.model_sequence
    model_specific_sequence = args.model_specific_sequence

    #---Prepare new config_file---
    if args.config_path is "Nothing":
        custom_config_file_path = cfg_prep.prepare_new_config_file_from_sequence(cfg, model_sequence, model_specific_sequence)
    else:
        custom_config_file_path = args.config_path

    config_file = custom_config_file_path
    model_predictor = COCO_predictor(cfg = cfg, custom_config_file = config_file, \
                                     use_conf_threshold = False, max_num_pred = 5, min_image_size = 60, masks_per_dim = 3)




if __name__ == "__main__":
    main()
    if dllogger_initialized:
        dllogger.log(step=tuple(), data={})
        dllogger.flush()