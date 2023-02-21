import sys
import os
import subprocess
from pathlib import Path
from distutils.util import strtobool

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.append(path_main)
    os.chdir(path_main)
    sys.path.remove('/workspace/object_detection')
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from EXPERIMENTS.complete_crop_dataset_rec_featuremaps_var.utils.util_functions import Utilities_helper
from EXPERIMENTS.complete_crop_dataset_rec_featuremaps_var.objects.image_and_preds_processor_obj import imageAndPredictionProcessor
from EXPERIMENTS.complete_crop_dataset_rec_featuremaps_var.objects.logger_obj import loggerObj

import argparse
import logging
import functools
import gc
import time

class flowRunner:
    _LOGS_SUBDIR = "logs"
    _LOG_LEVEL = logging.DEBUG
    _FLOW_RUNNER_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))))
    #From to and bottom, separately
    _DEFAULT_VERTICAL_CROPPING_PERCENTAGE = 0.05

    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-dl', '--dataset-locations', nargs='*',
                            type=str,
                            required = False,
                            help='The location where the dataset to be shifted is located',
                            default=['/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable_shifted_h_0.5_v_1.0',
                                     '/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable'])
        parser.add_argument('-ndbl', '--new-dataset-base-location', nargs='?',
                            type=str,
                            required = False,
                            help='The new location where the shifted dataset will be located',
                            default='/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/'
                                    'dataset_coco_2017_cropped_n_centered_w_featuremaps')
        parser.add_argument('-en', '--experiment-name', nargs='?',
                            type=str,
                            required = False,
                            help='The new location where the shifted dataset will be located',
                            default='trial_5')
        parser.add_argument('-ll', '--lower-lengths', nargs='*',
                            type=float,
                            default = [0.05, 0.55],
                            required = False,
                            help='The cropping of the image to be performed, first of the shifted then centered dataset')
        parser.add_argument('-ul', '--upper-lengths', nargs='*',
                            type=float,
                            default = [0.30, 0.80],
                            required = False,
                            help='The vertical shifting to be performed on the image -'
                                 ' counted from the top left corner as a ratio of the height')
        parser.add_argument('-mp', '--model-config-path', nargs='?',
                            type=str,
                            required = False,
                            help='The location of the configuration path of the model',
                            default='/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/configs/R-101-FPN/variable_pretrained_resnet/variable_pretrained_resnet_baseline_resnet_norm.yaml')
        parser.add_argument('-depth', '--depth', nargs='?',
                            type=str,
                            required = False,
                            help='The location from which the tensor will be extracted: stem or layer3',
                            default='stem')
        parser.add_argument('-compress', '--compress', nargs='?',
                            type=lambda x:bool(strtobool(x)),
                            const=True,
                            default=True,
                            help='Whether to record the full featuremaps in each .pth file or to record, in this order:'
                                 '[1024x50x50]: original tensor'
                                 '[slice, 0, 0]: slice mean activation'
                                 '[slice, 1, 0]: slice median activation'
                                 '[slice, 2, 0]: slice max activation'
                                 '[slice, 3, 0]: slice st. dev. activation'
                                 '[1024x4x1]: final compressed tensor')
        parser.add_argument('-batchnorm', '--batchnorm', nargs='?',
                            type=lambda x:bool(strtobool(x)),
                            const=True,
                            default=True,
                            help='Whether to take the Tensors after batch norm (True), or before  batch norm (False)')

        args = parser.parse_args()
        self.dataset_locations = args.dataset_locations
        self.new_dataset_base_location = args.new_dataset_base_location
        self.lower_lengths = args.lower_lengths
        self.upper_lengths = args.upper_lengths
        self.model_config_path = args.model_config_path
        self.experiment_name = args.experiment_name
        self.tensor_depth = args.depth
        self.compress_tensors = args.compress
        self.batch_norm = args.batchnorm

        self.objects_setup_complete = False
        self.setup_objects_and_variables()


    def run_all(self):
        self.image_processor.load_model()
        self.image_processor.attach_hooks_to_model()
        self.image_processor.read_all_images_in_org_dataset()
        self.image_processor.process_all_images()


    def setup_objects_and_variables(self):
        self.utils_helper = Utilities_helper()

        _compression_suffix = "_compressed" if self.compress_tensors else "_uncompressed"
        _bn_or_conv_suffix = "_bn" if self.batch_norm else "_conv"

        self.new_dataset_location = os.path.join(self.new_dataset_base_location,
                                                 self.experiment_name + "_shifted_" + str(self.lower_lengths[0])
                                                 + "_" + str(self.upper_lengths[0]) + "_centered_"
                                                 + str(self.lower_lengths[1]) + "_" + str(self.upper_lengths[1]) +
                                                 f"_{self.tensor_depth}" +
                                                 _compression_suffix +
                                                 _bn_or_conv_suffix)

        self.utils_helper.check_dir_and_make_if_na(self.new_dataset_location)
        self.logger = loggerObj(#logs_subdir = self.new_dataset_base_location,
                                logs_subdir=self.new_dataset_location,
                                log_file_name = "log",
                                utils_helper = self.utils_helper,
                                log_level=flowRunner._LOG_LEVEL)
        logging.info("Successfully setup logger!")

        self.image_processor = imageAndPredictionProcessor(utils_helper=self.utils_helper, org_dataset_folder=self.dataset_locations,
                                                           new_dataset_folder=self.new_dataset_location, lower_lengths=self.lower_lengths,
                                                           upper_lengths= self.upper_lengths,
                                                           model_config_path=self.model_config_path,
                                                           tensor_depth = self.tensor_depth,
                                                           compress_tensors = self.compress_tensors,
                                                           batch_norm = self.batch_norm,
                                                           default_vertical_cropping_percentage = flowRunner._DEFAULT_VERTICAL_CROPPING_PERCENTAGE)


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
