import sys
import os
import torch
import dllogger

from pathlib import Path
try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    sys.path.append(path_main)
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from testing.test_model_classic.utils.util_functions import Utilities_helper
from utils_gen import model_utils
from testing.test_model_classic import test_functions
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process

from maskrcnn_benchmark.config import cfg


import argparse
import logging
import functools
import gc
import time

class testerObj:
    #TODO:
    #Combine this file with the test_functions one
    #Sneaky sneaky just replace the dataset name, and the predictions.pth file in the testing function
    #From what I saw all else should remain the same
    def __init__(self, model_config_file, current_bin_pth_dir_path,
                 current_bin_annotation_file_path, current_bin_dataset_name,
                 utils_helper):

        self.model_config_path = model_config_file
        self.current_bin_pth_dir_path = current_bin_pth_dir_path
        self.current_bin_annotation_file_path = current_bin_annotation_file_path
        self.current_bin_dataset_name = current_bin_dataset_name
        self.utils_helper = utils_helper

    def build_model(self):
        #add function which returns the model as well as the established CFG file
        #pass the model and the cfg to the test_model function in my_train_net
        #Perhaps make a local copy of my_train_net to keep everything modular and clean
        self.model, self.cfg = model_utils.load_model_and_cfg(self.model_config_path)

        logging.info("Successfully loaded model weights")

    def test_model(self):
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        distributed = num_gpus > 1

        dllogger.init(backends=[])

        dllogger_initialized = True
        dllogger.log(step="PARAMETER", data={"gpu_count": num_gpus})
        # dllogger.log(step="PARAMETER", data={"environment_info": collect_env_info()})
        dllogger.log(step="PARAMETER", data={"config_path": self.model_config_path})
        with open(self.model_config_path, "r") as cf:
            config_str = "\n" + cf.read()
        dllogger.log(step="PARAMETER", data={"config": self.cfg})

        dllogger.log(step="INFORMATION", data="Running evaluation...")
        self.test_model_sneaky_v2(cfg=self.cfg, model=self.model, distributed=distributed,
                                  dllogger=dllogger, iters_per_epoch=1,
                                  current_bin_annotation_file_path = self.current_bin_annotation_file_path,
                                  current_bin_dataset_name = self.current_bin_dataset_name,
                                  current_bin_pth_dir_path = self.current_bin_pth_dir_path)


    def test_model_sneaky_v2(cfg, model, distributed, iters_per_epoch, dllogger,
                             current_bin_annotation_file_path,
                             current_bin_dataset_name,
                             current_bin_pth_dir_path):

        if distributed:
            model = model.module
        torch.cuda.empty_cache()  # TODO check if it helps
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        output_folders = [current_bin_pth_dir_path]
        dataset_names = [current_bin_dataset_name]

        data_loaders_val = make_data_loader_custom(cfg, current_bin_annotation_file_path,
                                                   is_train=False, is_distributed=distributed)
        results = []
        # -----MODIFICATION-----
        # This is used to allow for custom prediction files to be found by the inference function
        # Used for border-based evaluation
        if not (current_bin_pth_dir_path is None):
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
                    output_folder=current_bin_pth_dir_path,
                    dllogger=dllogger,
                )
                synchronize()
                results.append(result)
            if is_main_process():
                map_results, raw_results = results[0]
                bbox_map = map_results.results["bbox"]['AP']
                segm_map = map_results.results["segm"]['AP']
                dllogger.log(step=(cfg.SOLVER.MAX_ITER, cfg.SOLVER.MAX_ITER / iters_per_epoch,),
                             data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})
                dllogger.log(step=tuple(), data={"BBOX_mAP": bbox_map, "MASK_mAP": segm_map})
        # ----------------------


    def run_all(self):
        self.build_model()
        self.test_model()
