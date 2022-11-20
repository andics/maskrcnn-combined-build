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
    #Some default (usually unnecessary to change) parameters
    _LOGS_SUBDIR = "logs"
    _LOGGER_NAME = "main_logger"
    _ORIGINAL_ANNOTATIONS_SUBDIR = "original_annotations"
    _PROCESSED_ANNOTATIONS_SAVE_SUBDIR = "filtered_annotations"
    _OVERRIDE_ANNOTATIONS = False

    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-ap', '--annotations-path', nargs='?',
                            type=str,
                            default=os.path.join(str(Path(os.path.dirname(os.path.realpath(__file__)))),
                                                   flowRunner._ORIGINAL_ANNOTATIONS_SUBDIR, "instances_val2017.json"),
                            required = False,
                            help='Path to annotation file to be filtered')
        parser.add_argument('-mb', '--middle-boundary', nargs='+',
                            required=False,
                            default=[100],
                            help='The edge size of the middle square we define to have high-resolution')
        parser.add_argument('-lt', '--lower-threshold', nargs='+',
                            type=float,
                            default=[0.9],
                            required = False,
                            help='(% / 100) The lower boundary of area we want to'
                                 ' allow a non-filtered object to have in middle')
        parser.add_argument('-ut', '--upper-threshold', nargs='+',
                            type=float,
                            default=[1.0],
                            required = False,
                            help='(% / 100) The upper boundary of area we want to'
                                 ' allow a non-filtered object to have in middle'
                                 'IMPORTANT: There need to be as may upper-threshold elements as lower')
        #---FILE-STRUCTURE-ARGS--
        parser.add_argument('-fd', '--filtered-dir', nargs='?',
                            type=str,
                            default = os.path.join(str(Path(os.path.dirname(os.path.realpath(__file__)))),
                                                   flowRunner._PROCESSED_ANNOTATIONS_SAVE_SUBDIR),
                            required=False,
                            help='Directory where the filtered annotation file will be stored')
        parser.add_argument('-en', '--experiment-name', nargs='?',
                            type=str,
                            default = "variable_resolution_test",
                            required = False,
                            help='The name which the new annotation file will assume (based on the experiment name')


        args = parser.parse_args()
        self.original_annotations_path = args.annotations_path
        self.middle_boundary = args.middle_boundary[0]
        self.area_threshold_array = list(zip(args.lower_threshold,
                                             args.upper_threshold))
        self.annotations_save_dir = args.filtered_dir
        self.experiment_name = args.experiment_name

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def run_all(self):
        for _current_threshold_array in self.area_threshold_array:
            annotations_save_path = os.path.join(self.annotations_save_dir,
                                                 self.experiment_name +
                                                 f"_{str(_current_threshold_array[0])}" +
                                                 f"_{str(_current_threshold_array[1])}" +
                                                 "_instances_val2017" + "." +
                                                 self.utils_helper.extract_filename_and_ext(
                                                 self.original_annotations_path)[1])
            if (not flowRunner._OVERRIDE_ANNOTATIONS) and (os.path.exists(annotations_save_path)):
                self.logger.log(f"Skipping annotation bin {_current_threshold_array} as file already exists ...")
                continue
            else:
                if os.path.exists(annotations_save_path): self.logger.log(f"Overriding annotation bin {_current_threshold_array}")
                self.annotation_processor = annotationProcessor(original_annotations_path = self.original_annotations_path,
                                                                filter_threshold_array = _current_threshold_array,
                                                                middle_boundary=self.middle_boundary,
                                                                experiment_name= self.experiment_name,
                                                                new_annotations_file_path = annotations_save_path,
                                                                utils_helper= self.utils_helper,
                                                                logger= self.logger)

            self.annotation_processor.read_annotations()
            self.annotation_processor.filter_annotations_w_wrong_area_ratio()
            self.annotation_processor.write_new_annotations_to_disk()


    def setup_objects_and_file_structure(self):
        self.utils_helper = Utilities_helper()

        #Setting up logger file structure
        self.logs_subdir = os.path.join(self.main_file_dir, flowRunner._LOGS_SUBDIR)
        self.utils_helper.check_dir_and_make_if_na(self.logs_subdir)

        #Setting up the logger
        self.logger = Logger(logger_name = flowRunner._LOGGER_NAME,
                             logs_subdir = self.logs_subdir,
                             log_file_name = self.experiment_name,
                             utils_helper = self.utils_helper)
        self.logger.log("Finished setting up logger object")

        _tmp = self.utils_helper.check_dir_and_make_if_na(self.annotations_save_dir)
        self.logger.log(f"Finished setting up new annotations folder structure! Created new annotations sub-dir: {not _tmp}")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects_and_file_structure()
    flow_runner.run_all()