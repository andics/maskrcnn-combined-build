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


    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    utils.check_output_dir(output_dir)

    #-----Generate-command-line-----
    if cfg.COMMAND.GENERATE_COMMAND:
        line_script = utils.generate_command_line(cfg, __file__)
        print("For cluster running, use the following script: ")
        print(line_script)
    #-------------------------------

    #-----KILL-PHANTOM-PROCESSES-----
    torch.cuda.empty_cache()
    gc.collect()
    #--------------------------------

    #-----PARALLELIZE-MODEL-----
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        try:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
            synchronize()
        except Exception as e:
            #Used to fix Address already in use error
            print("Encountered the following error when initializing distributed training: ", e.__str__())
            bash_command = "$kill $(ps aux | grep " + __file__ + " | grep -v grep | awk '{print $2}')"
            process = subprocess.run(bash_command, shell=True)
            #Wait for a while for the GPU memory to get cleared
            time.sleep(1800)
            sys.exit()

    #---------------------------

    #-----Prepare Tensorboard-----
    tensorboard_writer = utils.setup_tensorboard(distributed, cfg)
    #-----------------------------

    #-----INITIALIZE-LOGGER + LOG-BASIC-INFO-----
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=json_summary_file_name),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])
    dllogger_initialized = True
    dllogger.log(step="PARAMETER", data={"gpu_count": num_gpus})
    # dllogger.log(step="PARAMETER", data={"environment_info": collect_env_info()})
    dllogger.log(step="PARAMETER", data={"config_path": config_file})
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
    dllogger.log(step="PARAMETER", data={"config": cfg})
    #---------------------------------------------

    model, iters_per_epoch = train(cfg, args.local_rank, distributed, dllogger, tensorboard_writer)

    if not skip_test:
        test_model(cfg, model, distributed, iters_per_epoch, dllogger)


def train(cfg, local_rank, distributed, dllogger, tensorboard_writer):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    use_amp_ddp_training = cfg.SOLVER.USE_APEX_DDP
    if use_amp_ddp_training:
        #Initializing distributed AMP training
        #AMP training will currently be supported for Distributed processes ONLY

        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        if use_amp_ddp_training:
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False, find_unused_parameters=True,
            )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    #Make sure that only local_rank = 0 process will try to save the file
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader, iters_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # set the callback function to evaluate and potentially
    # early exit each epoch
    if cfg.PER_EPOCH_EVAL:
        per_iter_callback_fn = functools.partial(
                mlperf_test_early_exit,
                iters_per_epoch=iters_per_epoch,
                tester=functools.partial(test, cfg=cfg, dllogger=dllogger),
                model=model,
                distributed=distributed,
                min_bbox_map=cfg.MIN_BBOX_MAP,
                min_segm_map=cfg.MIN_MASK_MAP)
    else:
        per_iter_callback_fn = None

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        use_amp_ddp_training,
        cfg,
        dllogger,
        distributed,
        tensorboard_writer,
        iters_per_epoch,
        per_iter_end_callback_fn=per_iter_callback_fn,
    )

    return model, iters_per_epoch


def test_model(cfg, model, distributed, iters_per_epoch, dllogger, current_iterations = None,
               output_folder_override = None):
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
            if current_iterations is not None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", str(current_iterations) + "_" + dataset_name)
                print("About to evaluate model with ", str(current_iterations), " number of iterations!")
            else:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)

            if not os.path.exists(output_folder):
                mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    results = []
    #-----MODIFICATION-----
    #This is used to allow for custom prediction files to be found by the inference function
    #Used for border-based evaluation
    if not (output_folder_override is None):
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            result = inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder_override,
                dllogger=dllogger,
            )
            synchronize()
            results.append(result)
        if is_main_process():
            map_results, raw_results = results[0]
            bbox_map = map_results.results["bbox"]['AP']
            segm_map = map_results.results["segm"]['AP']
            dllogger.log(step=(cfg.SOLVER.MAX_ITER, cfg.SOLVER.MAX_ITER / iters_per_epoch,), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})
            dllogger.log(step=tuple(), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})
    #----------------------
    else:
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            result = inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                dllogger=dllogger,
            )
            synchronize()
            results.append(result)
        if is_main_process():
            map_results, raw_results = results[0]
            bbox_map = map_results.results["bbox"]['AP']
            segm_map = map_results.results["segm"]['AP']
            dllogger.log(step=(cfg.SOLVER.MAX_ITER, cfg.SOLVER.MAX_ITER / iters_per_epoch,), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})
            dllogger.log(step=tuple(), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})


def test_and_exchange_map(tester, model, distributed):
    results = tester(model=model, distributed=distributed)

    # main process only
    if is_main_process():
        # Note: one indirection due to possibility of multiple test datasets, we only care about the first
        #       tester returns (parsed results, raw results). In our case, don't care about the latter
        map_results, raw_results = results[0]
        bbox_map = map_results.results["bbox"]['AP']
        segm_map = map_results.results["segm"]['AP']
    else:
        bbox_map = 0.
        segm_map = 0.

    if distributed:
        map_tensor = torch.tensor([bbox_map, segm_map], dtype=torch.float32, device=torch.device("cuda"))
        torch.distributed.broadcast(map_tensor, 0)
        bbox_map = map_tensor[0].item()
        segm_map = map_tensor[1].item()

    return bbox_map, segm_map


def mlperf_test_early_exit(iteration, iters_per_epoch, tester, model, distributed, min_bbox_map, min_segm_map):
    if iteration > 0 and iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch

        dllogger.log(step="PARAMETER", data={"eval_start": True})

        bbox_map, segm_map = test_and_exchange_map(tester, model, distributed)

        # necessary for correctness
        model.train()
        dllogger.log(step=(iteration, epoch, ), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})

        # terminating condition
        if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
            dllogger.log(step="PARAMETER", data={"target_accuracy_reached": True})
            return True

    return False


if __name__ == "__main__":
    main()
    if dllogger_initialized:
        dllogger.log(step=tuple(), data={})
        dllogger.flush()