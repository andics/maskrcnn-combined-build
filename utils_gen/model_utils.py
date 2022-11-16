# Function containing custom NN model related functions.
import re

import maskrcnn_benchmark
import maskrcnn_benchmark.modeling.detector

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from maskrcnn_benchmark.config import cfg
try:
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
except Exception as e:
    print("Tried to import DetectronCheckpointer inside model_utils.py but it seems it is already in the Path. Moving on...")


import os
import numpy as np
import yaml
import sys
import glob


def setup_env_variables():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generate_command_line(cfg, training_script_path):
    script_output_path = os.path.join(cfg.COMMAND.BASE_STORAGE_PATH, "model_" + cfg.MODEL.SEQUENCE, "model_" + cfg.MODEL.SEQUENCE + "_" + cfg.MODEL.SPECIFIC_SEQUENCE)
    Path(script_output_path).mkdir(parents=True, exist_ok=True)

    script_out_file = os.path.join(script_output_path, "useCase_out_from_train_%J.log")
    script_error_file = os.path.join(script_output_path, "useCase_err_from_train_%J.log")
    num_gpus = str(cfg.COMMAND.NUM_GPUS)
    num_jobs = str(cfg.COMMAND.NUM_JOBS)

    command = "../shared/seq_arr.sh -c \"bsub -env LSB_CONTAINER_IMAGE=\"ibdgx001:5000/maskrcnn_nvidia_pytorch:1.6.2\" -app nvidia-gpu -gpu num={0}:j_exclusive=yes -q waic-short -R rusage[mem=64000] " \
              "-R affinity[thread*24] -R select[hname!=dgxws02] -o {1} -e {2} " \
              "-J \"maskrcnn_seq[1-{3}]\" -H python3 " \
              "-m torch.distributed.launch --nproc_per_node={0} --master_port=1310 /home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/training/my_train_net.py --model-sequence {4} " \
              "--model-specific-sequence {5}\" -e {3} -d ended".format(num_gpus, script_out_file, script_error_file, num_jobs, cfg.MODEL.SEQUENCE, cfg.MODEL.SPECIFIC_SEQUENCE)

    return command


def overlay_images_from_multi_ch_tensor(numpy_tensor_img):
    # print("Numpy shape from FUNCTION: ", np.shape(numpy_tensor_img))

    image_width = np.shape(numpy_tensor_img)[1]
    image_height = np.shape(numpy_tensor_img)[0]
    num_channels = 3
    starting_selection_ch = 0
    separate_channel_averages = np.zeros([image_height, image_width, num_channels], dtype=float)

    for i in range(num_channels):
        print("Color range for image CH{}: ".format(i),
              get_data_range(list(numpy_tensor_img[:, :, i * 3:i * 3 + 3].flatten())))
        current_channel_average = np.average(numpy_tensor_img[:, :, i * 3:i * 3 + 3], axis=2)
        separate_channel_averages[:, :, i] = current_channel_average

    # Binarize the separate channel views
    # Third channel is CH3
    separate_channel_averages_bin = (separate_channel_averages != 0).astype(np.int_)
    separate_channel_averages_bin_9_ch = np.zeros([image_height, image_width, 9])
    separate_channel_averages_bin_9_ch[:, :, 6:9] = np.repeat(separate_channel_averages_bin[:, :, 2][:, :, np.newaxis],
                                                              3, axis=2)
    separate_channel_averages_bin_9_ch[:, :, 3:6] = np.repeat(separate_channel_averages_bin[:, :, 1][:, :, np.newaxis],
                                                              3, axis=2)
    separate_channel_averages_bin_9_ch[:, :, 0:3] = np.repeat(separate_channel_averages_bin[:, :, 0][:, :, np.newaxis],
                                                              3, axis=2)

    overlaid_image = np.copy(numpy_tensor_img[:, :, 6:9])
    overlaid_image[separate_channel_averages_bin_9_ch[:, :, 3:6] == 1] = numpy_tensor_img[:, :, 3:6][
        separate_channel_averages_bin_9_ch[:, :, 3:6] == 1]
    overlaid_image[separate_channel_averages_bin_9_ch[:, :, 0:3] == 1] = numpy_tensor_img[:, :, 0:3][
        separate_channel_averages_bin_9_ch[:, :, 0:3] == 1]
    overlaid_image = (overlaid_image + 128) / 255
    overlaid_image = np.flip(overlaid_image, axis=2)

    return overlaid_image


def get_data_range(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)


def change_yaml_file_value(file_location, variable_to_change, new_value):
    #E.g usage:
    #   variable_to_change = ['SOLVER', 'CATCHUP_PHASE_TRIGGERED']
    #   new_value = False

    with open(file_location) as f:
        doc = yaml.safe_load(f)

    edit_from_access_pattern(variable_to_change, doc, new_value)

    with open(file_location, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)


def edit_from_access_pattern(access_pattern, nested_dict, new_value):
    if len(access_pattern) == 1:
        nested_dict[access_pattern[0]] = new_value
    else:
        return edit_from_access_pattern(access_pattern[1:], nested_dict[access_pattern[0]], new_value)


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


def load_model_and_cfg(custom_config_file):
    #Sets the used GPU to id 0
    default_weight_file_name = "model_final.pth"

    cfg.merge_from_file(custom_config_file)
    cfg.freeze()

    model_weight_dir = cfg.OUTPUT_DIR

    model = maskrcnn_benchmark.modeling.detector.build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    #Weight loading
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_weight_dir)
    _ = checkpointer.load(os.path.join(model_weight_dir, default_weight_file_name))

    model.eval()

    return model, cfg


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