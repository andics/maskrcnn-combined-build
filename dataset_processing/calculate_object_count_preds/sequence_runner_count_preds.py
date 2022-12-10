import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
        parser.add_argument('-afp', '--annotation-file-path', nargs='?',
                            type=str,
                            default="/home/projects/bagon/shared/coco/annotations/instances_val2017.json",
                            required = False,
                            help='Path to the  parent COCO annotation file used to generate the predictions'
                                 'This is needed only for information about the images - E.g. size, filename etc.'
                                 'Does not need to be the specific annotation file used for a bin: only the general annotation file'
                                 'for the parent model - Variable, Equiconst')
        parser.add_argument('-ip', '--images-path', nargs='?',
                            type=str,
                            default="/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
                            required = False,
                            help='Path to the folder in which the images used for generating the'
                                 ' predictions.pth file are located')
        parser.add_argument('-cp', '--config-path', nargs='?',
                            type=str,
                            default = os.path.join(str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1]),
                                                   "configs/R-101-FPN/variable_pretrained_resnet/variable_pretrained_resnet_baseline_resnet_norm.yaml"),
                            required=False,
                            help='The configuration file used to construct the model. Can be any generic config file, '
                                 'even if the test set IS NOT the name of the set you wish to filter now. The'
                                 'important part is that the config file is of the type of the model you used'
                                 ' to generate the original predictions.pth you wish to filter now.'
                                 'E.g. If the predictions.pth is of the pretrained_resnet_var, use any'
                                 'config file of pretrained_resnet_var')

        args = parser.parse_args()
        self.main_predictions_path = args.predictions_path
        self.dataset_images_path = args.images_path
        self.annotation_file_path = args.annotation_file_path
        self.config_path = args.config_path

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def run_all(self):
        for pred_file in sorted(self.pth_file_paths_full):
            pred_processor = predictionProcessor(pred_file,
                                                 self.annotation_file_path,
                                                 self.dataset_images_path,
                                                 self.config_path,
                                                 self.logger)
            pred_processor.read_predictions()
            pred_processor.setup_coco_image_reader()
            total_counts = pred_processor.count_predictions()
            self.logger.log(f"Objects in {pred_file}:\n {total_counts}")


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