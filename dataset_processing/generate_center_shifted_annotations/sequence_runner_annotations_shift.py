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

from dataset_processing.generate_center_shifted_annotations.objects.logger_obj import Logger
from dataset_processing.generate_center_shifted_annotations.utils.util_functions import Utilities_helper
from dataset_processing.generate_center_shifted_annotations.objects.annotations_processor_obj import annotationProcessor

import argparse
import logging
import functools
import gc
import time

class flowRunner:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-al', '--annotations-location', nargs='?',
                            type=str,
                            required = False,
                            help='The location where the annotations to be shifted are located',
                            default='/home/projects/bagon/shared/coco/annotations/instances_val2017.json')
        parser.add_argument('-nabl', '--new-annotations-base-location', nargs='?',
                            type=str,
                            required = False,
                            help='The new location where the shifted dataset will be located',
                            default='/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations_polygon')
        parser.add_argument('-hs', '--horizontal-shift', nargs='?',
                            type=float,
                            required = True,
                            help='The horizontal shifting to be performed on the image -'
                                 ' counted from the top left corner as a ratio of the width')
        parser.add_argument('-vs', '--vertical-shift', nargs='?',
                            type=float,
                            required = True,
                            help='The vertical shifting to be performed on the image -'
                                 ' counted from the top left corner as a ratio of the height')

        args = parser.parse_args()
        self.annotations_location = args.annotations_location
        self.new_annotations_base_location = args.new_annotations_base_location
        self.horizontal_shift = args.horizontal_shift
        self.vertical_shift = args.vertical_shift

        self.objects_setup_complete = False
        self.setup_objects_and_variables()

    def run_all(self):
        self.annotation_processor.read_annotations()
        self.annotation_processor.shift_annotations()
        self.annotation_processor.write_new_annotations_to_disk()

    def setup_objects_and_variables(self):
        self.utils_helper = Utilities_helper()

        self.new_annotations_location = os.path.join(self.new_annotations_base_location,
                                                     self.utils_helper.extract_filename_and_ext(self.annotations_location)[0] +
                                                 "_shifted_h_" + str(self.horizontal_shift) + "_v_"
                                                     + str(self.vertical_shift) + "." + self.utils_helper.extract_filename_and_ext(self.annotations_location)[1])
        self.utils_helper.check_dir_and_make_if_na(self.new_annotations_base_location)

        self.annotation_processor = annotationProcessor(utils_helper=self.utils_helper, org_annotations_location=self.annotations_location,
                                                        new_annotations_location=self.new_annotations_location, horizontal_shift=self.horizontal_shift,
                                                        vertical_shift = self.vertical_shift)


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()