# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""
import sys

sys.path.remove('/workspace/object_detection')
sys.path.append('/home/labs/waic/dannyh/work/code/docker/maskrcnn_nvidia/object_detection/')

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

####################################################################################
# requires the following Docker container: "ibdgx001:5000/dh_maskrcnn_05jul2020:v3"
####################################################################################
import argparse
import os

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
from maskrcnn_benchmark.data.transforms import build_transforms

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
# try:
# from apex import amp
# use_amp = True
# except ImportError:
# print('Use APEX for multi-precision via apex.amp')
# use_amp = False
# try:
# from apex.parallel import DistributedDataParallel as DDP
# use_apex_ddp = True
# except ImportError:
# print('Use APEX for better performance')
# use_apex_ddp = False

#USAGE parameters:--config-file configs/multi_ch_model/e2e_mask_rcnn_R_101_FPN_1x_multi_train_ch1.yaml
# --weight-file trained_models/multi_resnet101_ch1/last_checkpoint/model_final.pth
# OUTPUT_DIR inference/single_channel

use_amp = False
use_apex_ddp = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    #Define config file for loading the model structure
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/home/labs/waic/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/configs/baseline_model/e2e_mask_rcnn_R_101_FPN_1x_equal.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    #Define weight file to be applied to structure
    parser.add_argument(
        "--weight-file",
        # default="/home/labs/waic/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/original_resnet101/last_checkpoint/model_final.pth",
        default="/home/labs/waic/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/multi_stacked_single_pretrained_resnet/baseline/last_checkpoint/model_0090000.pth",
        metavar="FILE",
        help="path to model weights file",
        type=str,
    )
    parser.add_argument(
        "--max-dets",
        nargs='+',
        type=int,
        default=[1, 10, 100],
        required=False,
    )
    parser.add_argument(
        "--test-set",
        default="",
        metavar="FILE",
        help="Name of set to be used for testing",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # The defined args can be accessed with their original name but _ instead of -
    # E.g: parser.add_argument(--this_arg); parser.parse_args()
    # my_var = parser.this_arg

    global args;
    args = parser.parse_args()
    print("Inputed arguments: \n", args)
    print("Current working dir: ", os.getcwd())

    current_working_dir = os.getcwd()
    config_file = os.path.join(current_working_dir, args.config_file)
    weight_file = os.path.join(current_working_dir, args.weight_file)
    input_config_options = args.opts

    max_dets = args.max_dets
    print("Working with max_dets: ", max_dets)

    test_dataset_arg = args.test_set
    test_dataset = (test_dataset_arg,)

    cfg.merge_from_file(config_file)
    if not test_dataset_arg=="":
        cfg.merge_from_list(["DATASETS.TEST", test_dataset])
    cfg.merge_from_list(input_config_options)
    cfg.freeze()

    print('Output directory: ' + cfg.OUTPUT_DIR + '\n')

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        print("Folder didn't exists and was just created!")

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Weight loading
    checkpointer = DetectronCheckpointer(cfg, model, save_dir = weight_file)
    _ = checkpointer.load(weight_file)

    model.eval()
    model.zero_grad()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    transforms = build_transforms(cfg, False)

    #-----------------------------------------------------------------------------------------------------------------------

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()


    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    #Log state of the workspace before beginning training
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    test_model(cfg, model, args.distributed, max_dets)


def test_model(cfg, model, distributed, max_dets):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
#            max_dets = max_dets
        )
        synchronize()

if __name__ == "__main__":
    print("Inside Python Script")
    main()
