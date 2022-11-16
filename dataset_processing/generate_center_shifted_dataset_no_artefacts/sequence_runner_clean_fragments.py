import sys
import os
import subprocess
from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from dataset_processing.generate_center_shifted_dataset.objects.logger_obj import Logger
from dataset_processing.generate_center_shifted_dataset.utils.util_functions import Utilities_helper
from dataset_processing.generate_center_shifted_dataset.objects.image_processor_obj import imageProcessor

import argparse
import logging
import functools
import gc
import time

class flowRunner:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-dl', '--dataset-location', nargs='?',
                            type=str,
                            required = False,
                            help='The location where the dataset to be shifted is located',
                            default='/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/Variable_shifted_h_0.5_v_1.0')
        #/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/Variable_shifted_h_0.5_v_1.0
        parser.add_argument('-ndbl', '--new-dataset-base-location', nargs='?',
                            type=str,
                            required = False,
                            help='The new location where the shifted dataset will be located',
                            default='/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/filtered_no_fragments/Variable_shifted_h_0.5_v_1.0')
        #/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/filtered_no_fragments/Variable_shifted_h_0.5_v_1.0
        parser.add_argument('-hs', '--horizontal-shift', nargs='?',
                            type=float,
                            default=.0,
                            required=False,
                            help='The horizontal shifting to be performed on the image -'
                                 ' counted from the top left corner as a ratio of the width')
        parser.add_argument('-vs', '--vertical-shift', nargs='?',
                            type=float,
                            default=.0,
                            required = False,
                            help='The vertical shifting to be performed on the image -'
                                 ' counted from the top left corner as a ratio of the height')
        #---MODIFICATION---
        #Warning: the hs, vs variables are not used in this script. Their function was to shift the dataset in
        #the other script (generate_dataset_shifted), but I've left them here for two reasons:
        #1. In case we want to remove fragments of various shifting configurations
        #2. To make this script as fast as possible
        #------------------

        args = parser.parse_args()
        self.dataset_location = args.dataset_location
        self.new_dataset_base_location = args.new_dataset_base_location
        self.horizontal_shift = args.horizontal_shift
        self.vertical_shift = args.vertical_shift

        self.objects_setup_complete = False
        self.setup_objects_and_variables()


    def run_all(self):
        self.image_processor.read_all_images_in_org_dataset()
        self.image_processor.process_all_images()

    def setup_objects_and_variables(self):
        self.utils_helper = Utilities_helper()

        self.new_dataset_location = self.new_dataset_base_location
        self.utils_helper.check_dir_and_make_if_na(self.new_dataset_location)

        self.image_processor = imageProcessor(utils_helper=self.utils_helper, org_dataset_folder=self.dataset_location,
                                              new_dataset_folder=self.new_dataset_location, horizontal_shift=self.horizontal_shift,
                                              vertical_shift = self.vertical_shift)


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
