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

from dataset_processing.generate_resolution_bin_annotations.objects.logger_obj import Logger
from dataset_processing.generate_resolution_bin_annotations.utils.util_functions import Utilities_helper
from dataset_processing.generate_resolution_bin_annotations.objects.annotation_processor_obj import annotationProcessor

import argparse
import logging
import functools
import gc
import time

class flowRunner:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-ap', '--annotations-path', nargs='?',
                            type=str,
                            required = True,
                            help='Path to annotation file to be filtered')
        parser.add_argument('-mb', '--middle-boundary', nargs='+',
                            required=False,
                            default=[80],
                            help='The edge size of the middle square we define to have high-resolution')
        parser.add_argument('-lt', '--lower-threshold', nargs='?',
                            type=float,
                            default=0.0,
                            required = False,
                            help='(% / 100) The lower boundary of area we want to'
                                 ' allow a non-filtered object to have in middle')
        parser.add_argument('-ut', '--upper-threshold', nargs='?',
                            type=float,
                            default=0.1,
                            required = True,
                            help='(% / 100) The upper boundary of area we want to'
                                 ' allow a non-filtered object to have in middle')
        #---FILE-STRUCTURE-ARGS--
        parser.add_argument('-fp', '--filtered-path', nargs='?',
                            type=str,
                            required=True,
                            help='Directory where the filtered annotation file will be stored')
        parser.add_argument('-fp', '--filtered-path', nargs='?',
                            type=str,
                            required = True,
                            help='Directory where the filtered annotation file will be stored')



        args = parser.parse_args()
        self.config_location = args.config_path

        self.objects_setup_complete = False
        self.setup_objects()


    def run_all(self):
        self.annotation_processor.setup_new_annotations_folder_structure()
        self.annotation_processor.read_annotations()
        self.annotation_processor.filter_annotations_outside_border()
        self.annotation_processor.write_new_annotations_to_disk()


    def setup_objects(self):
        self.utils_helper = Utilities_helper()

        self.utils_helper.change_yaml_file_value(self.config_location, ['LOGGING', 'logs_subdir'],
                                                 self.utils_helper.extract_filename_and_ext(self.config_location)[0])
        self.config_file = self.utils_helper.read_yaml_data(self.config_location)

        self.logger = Logger(config_file=self.config_file, utils_helper=self.utils_helper)
        self.annotation_processor = annotationProcessor(self.config_file, self.utils_helper, self.logger)
        self.logger.log("Finished setting up objects")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
