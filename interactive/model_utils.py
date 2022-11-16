# Function containing custom NN model related functions.
import os
import numpy as np
import yaml
import sys
import glob
from pathlib import Path
try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[2])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print(f"Environmental paths updated successfully! Current PATH: {sys.path}")
except Exception:
    print("Tried to remove /workspace/object_detection from path but was unsuccessful!")

import re

import maskrcnn_benchmark.modeling.detector

from torch.utils.tensorboard import SummaryWriter
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer


def load_model(cutom_config_file):
    #Sets the used GPU to id 0
    default_weight_file_name = "model_final.pth"

    cfg.merge_from_file(cutom_config_file)
    cfg.freeze()

    model_weight_dir = cfg.OUTPUT_DIR

    model = maskrcnn_benchmark.modeling.detector.build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    #Weight loading
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_weight_dir)
    _ = checkpointer.load(os.path.join(model_weight_dir, default_weight_file_name))

    model.eval()

    return model


def network_chunk_activator(model_sequence, _str):
    if _str in model_sequence:
        return True
    else:
        return False


def check_output_dir(path_to_check):
    path_to_check = os.path.join(path_to_check, "last_checkpoint")
    Path(path_to_check).mkdir(parents=True, exist_ok=True)
    print("Confirmed output directory {}".format(path_to_check))


def check_has_checkpoint(path_to_check):
    save_file_folder = os.path.join(path_to_check, "last_checkpoint")
    if os.path.exists(save_file_folder):
        if len(glob.glob1(save_file_folder, '*.pth')) > 0:
            return True
        else:
            return False
    else:
        return False


def get_variable_name(my_var):
    my_var_name = [k for k, v in locals().iteritems() if v == my_var][0]
    return my_var_name


def setup_tensorboard(distributed, cfg):
    if cfg.MODEL.SPECIFIC_SEQUENCE is "Nothing":
        complete_tensorboard_log_folder_for_experiment = os.path.join(cfg.TENSORBOARD.BASE_LOG_DIR, cfg.MODEL.SEQUENCE)
    else:
        complete_tensorboard_log_folder_for_experiment = os.path.join(cfg.TENSORBOARD.BASE_LOG_DIR, cfg.MODEL.SEQUENCE + "_" + cfg.MODEL.SPECIFIC_SEQUENCE)

    #Explicitly create log dir because errors arrize in distributed training otherwise
    Path(complete_tensorboard_log_folder_for_experiment).mkdir(parents=True, exist_ok=True)

    #How long should the Writer between consequtive disk inform ation flushes (pushing local info to disk)
    tensorboard_flush_frequency_time = 10
    #How many events should the SummaryWriter be called for before it pushes the information to the disk
    tensorboard_flush_frequency_events = 3

    tensorboard_writer = SummaryWriter(log_dir = complete_tensorboard_log_folder_for_experiment,
                                       max_queue = tensorboard_flush_frequency_events,
                                       flush_secs = tensorboard_flush_frequency_time,
                                       purge_step = 2)
    return tensorboard_writer

def find_all_files_with_ext(folder, extension):
    list_of_files_full = []
    list_of_file_names = []

    for file in os.listdir(folder):
        if file.endswith(extension):
            list_of_files_full.append(os.path.join(folder, file))
            list_of_file_names.append(file)

    list_of_files_full.sort(key=lambda f: int(re.sub('\D', '', f)))
    list_of_file_names.sort(key=lambda f: int(re.sub('\D', '', f)))

    return list_of_files_full, list_of_file_names