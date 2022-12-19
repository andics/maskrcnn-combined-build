import sys
import os
import subprocess
from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.append(path_main)
    os.chdir(path_main)
    sys.path.remove('/workspace/object_detection')
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from dataset_processing.crop_dataset_for_featuremap_recording.utils.util_functions import Utilities_helper
from dataset_processing.crop_dataset_for_featuremap_recording.objects.image_processor_obj import imageProcessor

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
                            default='w:/bagon/dannyh/data/coco_filt/val2017/Variable_shifted_h_0.5_v_1.0')
        parser.add_argument('-ndbl', '--new-dataset-base-location', nargs='?',
                            type=str,
                            required = False,
                            help='The new location where the shifted dataset will be located',
                            default='w:/bagon/andreyg/Projects/Variable_Resolution/Datasets/dataset_coco_2017_cropped_n_centered')
        parser.add_argument('-ll', '--lower-length', nargs='?',
                            type=float,
                            default = 0.05,
                            required = False,
                            help='The horizontal shifting to be performed on the image -'
                                 ' counted from the top left corner as a ratio of the width')
        parser.add_argument('-up', '--upper-length', nargs='?',
                            type=float,
                            default = 0.25,
                            required = False,
                            help='The vertical shifting to be performed on the image -'
                                 ' counted from the top left corner as a ratio of the height')

        args = parser.parse_args()
        self.dataset_location = args.dataset_location
        self.new_dataset_base_location = args.new_dataset_base_location
        self.lower_length = args.lower_length
        self.upper_length = args.upper_length

        self.objects_setup_complete = False
        self.setup_objects_and_variables()


    def run_all(self):
        self.image_processor.read_all_images_in_org_dataset()
        self.image_processor.process_all_images()

    def setup_objects_and_variables(self):
        self.utils_helper = Utilities_helper()

        self.new_dataset_location = os.path.join(self.new_dataset_base_location,
                                                 self.utils_helper.extract_folder_name_from_path(self.dataset_location) +
                                                 "_shifted_h_" + str(self.lower_length) + "_v_"
                                                 + str(self.upper_length))
        self.utils_helper.check_dir_and_make_if_na(self.new_dataset_location)

        self.image_processor = imageProcessor(utils_helper=self.utils_helper, org_dataset_folder=self.dataset_location,
                                              new_dataset_folder=self.new_dataset_location, lower_length=self.lower_length,
                                              upper_length= self.upper_length)


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
