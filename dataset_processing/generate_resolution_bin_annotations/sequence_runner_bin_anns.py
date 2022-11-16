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

from dataset_processing.generate_resolution_bin_annotations.objects.logger_obj import Logger
from dataset_processing.generate_resolution_bin_annotations.utils.util_functions import Utilities_helper
from dataset_processing.generate_resolution_bin_annotations.objects.annotation_processor_obj import annotationProcessor

import argparse

class flowRunner:
    #Some default (usually unnecessary to change) parameters of the logger object
    _LOGS_SUBDIR = "logs"
    _LOGGER_NAME = "main_logger"

    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-ap', '--annotations-path', nargs='?',
                            type=str,
                            required = False,
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
                            required = False,
                            help='(% / 100) The upper boundary of area we want to'
                                 ' allow a non-filtered object to have in middle')
        #---FILE-STRUCTURE-ARGS--
        parser.add_argument('-fp', '--filtered-path', nargs='?',
                            type=str,
                            default = "Q:/Projects/Variable_resolution/Programming/maskrcnn-combined-build/dataset_processing"
                                      "/generate_resolution_bin_annotations/filtered_annotations",
                            required=False,
                            help='Directory where the filtered annotation file will be stored')
        parser.add_argument('-en', '--experiment-name', nargs='?',
                            type=str,
                            default = "variable_resolution_test",
                            required = False,
                            help='The name which the new annotation file will assume (based on the experiment name')


        args = parser.parse_args()
        self.experiment_name = args.experiment_name

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def run_all(self):
        pass
        #self.annotation_processor.setup_new_annotations_folder_structure()
        #self.annotation_processor.read_annotations()
        #self.annotation_processor.filter_annotations_outside_border()
        #self.annotation_processor.write_new_annotations_to_disk()


    def setup_objects(self):
        self.utils_helper = Utilities_helper()

        #Setting up logger file structure
        self.logs_subdir = os.path.join(self.main_file_dir, flowRunner._LOGS_SUBDIR)
        self.utils_helper.check_dir_and_make_if_na(self.logs_subdir)

        #Setting up the logger
        self.logger = Logger(logger_name = flowRunner._LOGGER_NAME,
                             logs_subdir = self.logs_subdir,
                             log_file_name = self.experiment_name,
                             utils_helper = self.utils_helper)
        #self.annotation_processor = annotationProcessor(self.config_file, self.utils_helper, self.logger)
        self.logger.log("Finished setting up objects")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects()
    flow_runner.run_all()
