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

from dataset_processing.generate_resolution_bin_predictions.objects.logger_obj import Logger
from dataset_processing.generate_resolution_bin_predictions.utils.util_functions import Utilities_helper
from dataset_processing.generate_resolution_bin_predictions.objects.prediction_processor_obj import predictionProcessor

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
                            default="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/coco_2017_variable_val/predictions.pth",
                            required = False,
                            help='Path to prediction file to be filtered')
        parser.add_argument('-mb', '--middle-boundary', nargs='+',
                            required=False,
                            default=[100],
                            help='The edge size of the middle square we define to have high-resolution')
        parser.add_argument('-lt', '--lower-threshold', nargs='+',
                            type=float,
                            default=[0.0],#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            required = False,
                            help='(% / 100) The lower boundary of area we want to'
                                 ' allow a non-filtered object to have in middle')
        parser.add_argument('-ut', '--upper-threshold', nargs='+',
                            type=float,
                            default=[0.1],#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            required = False,
                            help='(% / 100) The upper boundary of area we want to'
                                 ' allow a non-filtered object to have in middle'
                                 'IMPORTANT: There need to be as may upper-threshold elements as lower')
        parser.add_argument('-cp', '--config-path', nargs='?',
                            type=str,
                            default = os.path.join(str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1]),
                                                   "configs/R-101-FPN/variable_pretrained_resnet/variable_pretrained_resnet_baseline_resnet_norm.yaml"),
                            required=False,
                            help='The configuration file used to construct the model. Can be any generic config file, '
                                 'even if the test set IS NOT the name of the set you wish to filter now. The'
                                 'important part is that the config file is of the type of the model you used'
                                 ' to generate the original predictions.pth you wish to filter now.')
        parser.add_argument('-dn', '--dataset-name', nargs='?',
                            type=str,
                            default = "coco_2017_res_bin_0.0_0.1_var",
                            required = False,
                            help='Determines the location from which the images will be read as the predictions file'
                                 'gets filtered, as well as the name of the directory in which the new predictions file will'
                                 'be stored.')
        #---FILE-STRUCTURE-ARGS--
        parser.add_argument('-fd', '--filtered-dir', nargs='?',
                            type=str,
                            default = os.path.join(str(Path(os.path.dirname(os.path.realpath(__file__)))),
                                                   flowRunner._PROCESSED_PREDICTIONS_SAVE_SUBDIR),
                            required=False,
                            help='Directory where the filtered prediction file will be stored')

        print("Here")
        args = parser.parse_args()
        self.original_predictions_path = args.predictions_path
        self.middle_boundary = args.middle_boundary[0]
        self.area_threshold_array = list(zip(args.lower_threshold,
                                             args.upper_threshold))
        self.dataset_name = args.dataset_name
        self.config_path = args.config_path
        self.predictions_save_dir = os.path.join(args.filtered_dir, self.dataset_name)

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def run_all(self):
        for _current_threshold_array in self.area_threshold_array:
            predictions_save_path = os.path.join(self.predictions_save_dir,
                                                 self.utils_helper.extract_filename_and_ext(
                                                     self.original_predictions_path)[0] + "." +
                                                 self.utils_helper.extract_filename_and_ext(
                                                 self.original_predictions_path)[1])
            if (not flowRunner._OVERRIDE_PREDICTIONS) and (os.path.exists(predictions_save_path)):
                self.logger.log(f"Skipping prediction bin {_current_threshold_array} as file already exists ...")
                continue
            else:
                if os.path.exists(predictions_save_path): self.logger.log(f"Overriding prediction bin {_current_threshold_array}")
                self.prediction_processor = predictionProcessor(org_predictions_path=self.original_predictions_path,
                                                                new_predictions_path=predictions_save_path,
                                                                area_threshold_array=_current_threshold_array,
                                                                middle_boundry=self.middle_boundary,
                                                                model_cfg_path=self.config_path,
                                                                dataset_name=self.dataset_name,
                                                                logger=self.logger,
                                                                utils_helper=self.utils_helper)

            self.prediction_processor.setup_objects_and_misk_variables()
            self.prediction_processor.read_predictions()
            self.prediction_processor.filter_predictions_w_wrong_area_ratio()
            #self.prediction_processor.write_new_predictions_to_disk()


    def setup_objects_and_file_structure(self):
        self.utils_helper = Utilities_helper()

        #Setting up logger file structure
        self.logs_subdir = os.path.join(self.main_file_dir, flowRunner._LOGS_SUBDIR)
        self.utils_helper.check_dir_and_make_if_na(self.logs_subdir)

        #Setting up the logger
        self.logger = Logger(logger_name = flowRunner._LOGGER_NAME,
                             logs_subdir = self.logs_subdir,
                             log_file_name = self.dataset_name,
                             utils_helper = self.utils_helper)
        self.logger.log("Finished setting up logger object")

        _tmp = self.utils_helper.check_dir_and_make_if_na(self.predictions_save_dir)
        self.logger.log(f"Finished setting up new predictions folder structure! Created new predictions sub-dir: {not _tmp}")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects_and_file_structure()
    flow_runner.run_all()