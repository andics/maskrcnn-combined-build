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

from dataset_processing.calculate_object_count_preds.objects.logger_obj import Logger
from dataset_processing.calculate_object_count_preds.utils.util_functions import Utilities_helper
from dataset_processing.calculate_object_count_preds.objects.prediction_processor_obj import predictionProcessor

import argparse

class flowRunner:
    #Some default (usually unnecessary to change) parameters
    _LOGS_SUBDIR = "logs"
    _LOGGER_NAME = "main_logger"
    _ORIGINAL_PREDICTIONS_SUBDIR = "original_predictions"
    _PROCESSED_PREDICTIONS_SAVE_SUBDIR = "filtered_predictions"
    _OVERRIDE_PREDICTIONS = False

    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-pp', '--predictions-path', nargs='?',
                            type=str,
                            default="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/equiconst_pretrained_resnet/baseline_resnet_norm/inference",
                            required = False,
                            help='Path to prediction file(s) to be analyzed')

        args = parser.parse_args()
        self.main_predictions_path = args.predictions_path

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def run_all(self):
        for pred_file in sorted(self.pth_file_paths_full):
            pred_processor = predictionProcessor(pred_file, self.logger)
            pred_processor.read_predictions()
            total_grand = pred_processor.count_predictions()
            self.logger.log(f"Objects in {pred_file}:\n {total_grand}")


    def setup_objects_and_file_structure(self):
        self.utils_helper = Utilities_helper()

        #Setting up logger file structure
        self.logs_subdir = os.path.join(self.main_file_dir, flowRunner._LOGS_SUBDIR)
        self.utils_helper.check_dir_and_make_if_na(self.logs_subdir)

        #Setting up the logger
        self.logger = Logger(logger_name = flowRunner._LOGGER_NAME,
                             logs_subdir = self.logs_subdir,
                             log_file_name = "log",
                             utils_helper = self.utils_helper)
        self.logger.log("Finished setting up logger object")
        self.logger.log("Gathering all .pth files ...")
        self.pth_file_paths_full = self.utils_helper.list_all_files_w_ext_in_a_parent_folder_recursively(self.main_predictions_path,
                                                                                                         ("predictions.pth"))
        self.logger.log(self.pth_file_paths_full)



if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects_and_file_structure()
    flow_runner.run_all()