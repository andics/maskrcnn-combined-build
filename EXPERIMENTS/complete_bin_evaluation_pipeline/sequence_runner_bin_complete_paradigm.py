import sys
import os
import logging
import subprocess
import numpy as np
from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.append(path_main)
    os.chdir(path_main)
    sys.path.remove('/workspace/object_detection')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.logger_obj import loggerObj
from EXPERIMENTS.complete_bin_evaluation_pipeline.utils.util_functions import Utilities_helper
from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.annotation_processor_obj import annotationProcessor
from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.prediction_processor_obj import predictionProcessor
from EXPERIMENTS.complete_bin_evaluation_pipeline.objects.tester_obj import testerObj

import argparse

class flowRunner:
    #Some default (usually unnecessary to change) parameters
    _LOG_LEVEL = logging.DEBUG
    _ORIGINAL_ANNOTATIONS_SUBDIR = "original_annotations"
    _PROCESSED_ANNOTATIONS_SAVE_SUBDIR = "filtered_annotations"
    _OVERRIDE_ANNOTATIONS = False
    _MASKRCNN_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    _FLOW_RUNNER_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))))
    _GENERATED_ANNOTATION_FILES_NAME = "instances_val2017.json"
    _GENERATED_PRECITIONS_FILES_NAME = "predictions.pth"
    _GENERATED_EVALUTION_FILES_NAME = "coco_results.json"
    #This parameter determines whether the script will filter predictions having masks with no Logit score > 0.5
    #If FALSE: predictions regardless of their mask logits score will be kept
    #If TRUE: only predictions having at least 1 mask logit score > 0.5 will be kept
    _FILTER_MASK_LOGITS = False

    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for complete resolution-bin evaluation pipeline')
        parser.add_argument('-mn', '--model-name', nargs='?',
                            type=str,
                            default = "variable_resolution_pretrained_resnet_norm",
                            required = False,
                            help='This name will be used as: '
                                 '1. Name of the sub-directory in which the experiment files will be stored'
                                 '2. Prefix to the log file')
        parser.add_argument('-mcf', '--model-config-file', nargs='?',
                            type=str,
                            default = os.path.join(flowRunner._MASKRCNN_PARENT_DIR_ABSOLUTE,
                                                   "configs/R-101-FPN/variable_pretrained_resnet/variable_pretrained_resnet_baseline_resnet_norm.yaml"),
                            required = False,
                            help='This parameter is used: '
                                 '1. For constructing the model during testing'
                                 '2. NOT for its test set: this parameter is irrelevant')
        parser.add_argument('-mb', '--middle-boundary', nargs='+',
                            required=False,
                            default=[100],
                            help='The edge size of the middle square we define to have high-resolution')
        parser.add_argument('-bs', '--bin-spacing', nargs='?',
                            type=float,
                            default=0.04,
                            required = False,
                            help='(% / 100) The space between each resolution bin.'
                                 'E.g. If this paramter is set to 0.1, one can expct that the paradigm'
                                 'will generate 10 and evaluate 10 bins, starting from 0.0-0.1 and ending with'
                                 '0.9-1.0'
                                 'IMPORTANT: This parameter is also appended to the name of the'
                                 'folder in which this experiment is stored')
        parser.add_argument('-oal', '--org-annotations-location', nargs='?',
                            type=str,
                            default = os.path.normpath(os.path.join(flowRunner._MASKRCNN_PARENT_DIR_ABSOLUTE,
                                                   "annotations/original_annotations/instances_val2017.json")),
                            required = False,
                            help='The location of the original annotation file to be filtered')
        parser.add_argument('-il', '--images-location', nargs='?',
                            type=str,
                            default = "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
                            required = False,
                            help='The location of the images for the parent dataset'
                                 'E.g. The Variable images')
        parser.add_argument('-opl', '--org-predictions-location', nargs='?',
                            type=str,
                            default = os.path.normpath(os.path.join(flowRunner._MASKRCNN_PARENT_DIR_ABSOLUTE,
                                                   "trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/coco_2017_variable_val/predictions.pth")),
                            required = False,
                            help='The location of the original prediction file to be filtered')
        parser.add_argument('-psl', '--parent-storage-location', nargs='?',
                            type=str,
                            default = os.path.normpath(os.path.join(flowRunner._FLOW_RUNNER_PARENT_DIR_ABSOLUTE,
                                                   "evaluations")),
                            required = False,
                            help='The location in which the newly generated annotation file'
                                 ' as well as the newly generated predictions file will be stored')

        self.args = parser.parse_args()

        self.model_name = self.args.model_name
        self.model_config_file = self.args.model_config_file
        self.middle_boundary = self.args.middle_boundary[0]
        self.bin_spacing = self.args.bin_spacing
        self.org_annotations_location = self.args.org_annotations_location
        self.images_location = self.args.images_location
        self.org_predictions_location = self.args.org_predictions_location
        self.parent_storage_location = self.args.parent_storage_location

        self.experiment_name = self.model_name + "_" + str(float(self.bin_spacing)) + "_" + str(self.middle_boundary)
        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False


    def setup_objects_and_file_structure(self):
        self.utils_helper = Utilities_helper()
        #Setting up logger file structure
        self.experiment_dir = os.path.join(self.parent_storage_location, self.experiment_name)
        self.utils_helper.check_dir_and_make_if_na(self.experiment_dir)

        #Setting up the logger
        self.logger = loggerObj(logs_subdir = self.experiment_dir,
                                log_file_name = "log",
                                utils_helper = self.utils_helper,
                                log_level=flowRunner._LOG_LEVEL)
        logging.info("Finished setting up logger...")
        logging.info("Passed arguments -->>")
        logging.info('\n  -  '+ '\n  -  '.join(f'{k}={v}' for k, v in vars(self.args).items()))
        #---SETUP-BINS---
        self.bins_lower_threshold = list(np.around(np.linspace(0, 1-self.bin_spacing,
                                                               int(1/self.bin_spacing)),
                                                   decimals=4))
        self.bins_upper_threshold = list(np.around(np.linspace(self.bin_spacing, 1,
                                                               int(1 / self.bin_spacing)),
                                                   decimals=4))
        assert len(self.bins_upper_threshold) == len(self.bins_lower_threshold)
        logging.info("Bin pairs setup complete -->>")

        #---SETUP-EVALUATION-FOLDER-NAMES---
        self.evaluation_foders = []
        for lower_threshold, upper_threshold in zip(self.bins_lower_threshold, self.bins_upper_threshold):
            _current_dir = os.path.join(self.experiment_dir, str("{:.4f}".format(lower_threshold))
                                        + "_" + str("{:.4f}".format(upper_threshold))
                                        + "_eval")
            _current_dir = os.path.normpath(_current_dir)
            self.utils_helper.check_dir_and_make_if_na(_current_dir)
            self.evaluation_foders.append(_current_dir)
        logging.info("Setup individual bin evaluation folders -->>")
        #-----------------------------------

        #---SETUP-GENERATED-FILES-PATHS---
        #This portion of the script generates the complete paths each new annotation and predictions file will assume
        self.generated_annotation_files_paths = []
        self.generated_predictions_files_paths = []
        self.generated_test_sets_names = []
        for evaluation_folder in self.evaluation_foders:
            self.generated_annotation_files_paths.append(os.path.join(evaluation_folder,
                                                                      flowRunner._GENERATED_ANNOTATION_FILES_NAME))
            self.generated_predictions_files_paths.append(os.path.join(evaluation_folder,
                                                                      flowRunner._GENERATED_PRECITIONS_FILES_NAME))


        for lower_threshold, upper_threshold in zip(self.bins_lower_threshold, self.bins_upper_threshold):
            self.generated_test_sets_names.append(self.model_name + "_" + str(float(self.bin_spacing)) +
                                                  "_" + str(self.middle_boundary) +
                                                  str("{:.4f}".format(lower_threshold)) + "_" +
                                                  str("{:.4f}".format(upper_threshold)) + "_eval")

        #---------------------------------
        logging.info('\n  -  '+ '\n  -  '.join(f'({l} | {u}) \n  -  Evaluation dir: {f}'
                                               f' \n  -  Annotation file: {a}'
                                               f' \n  -  Predictions file: {p}'
                                               f' \n  -  Test set name: {t}' for l, u, f, a, p, t in
                                               zip(self.bins_lower_threshold,
                                                self.bins_upper_threshold,
                                                self.evaluation_foders,
                                                self.generated_annotation_files_paths,
                                                self.generated_predictions_files_paths,
                                                self.generated_test_sets_names)))


    def run_all(self):
        logging.info("Running per bin evaluation -->>")
        for lower_threshold, upper_threshold,\
            evaluation_folder, gen_annotation_file_path,\
            gen_prediction_file_path, gen_test_set_name in zip(self.bins_lower_threshold,
                                                                     self.bins_upper_threshold,
                                                                     self.evaluation_foders,
                                                                     self.generated_annotation_files_paths,
                                                                     self.generated_predictions_files_paths,
                                                                     self.generated_test_sets_names):
            logging.info(f"Working on bin {lower_threshold}-{upper_threshold} in:\n{evaluation_folder}")
            self.logger.add_temp_file_handler_and_remove_main_file_handler(evaluation_folder)
            #---ANNOTATION-PREP---
            if not os.path.exists(gen_annotation_file_path):
                annotation_processor_object = annotationProcessor(original_annotations_path= self.org_annotations_location,
                                                                  new_annotations_file_path = gen_annotation_file_path,
                                                                  filter_threshold_array = (lower_threshold, upper_threshold),
                                                                  middle_boundary= self.middle_boundary,
                                                                  utils_helper= self.utils_helper)
                annotation_processor_object.read_annotations()
                annotation_processor_object.filter_annotations_w_wrong_area_ratio()
                annotation_processor_object.write_new_annotations_to_disk()
            else: logging.info("Bin annotation file exists. Moving to prediction file processing -->>")
            #---PREDICTION-PROCESSING---
            if not os.path.exists(gen_prediction_file_path):
                prediction_processor_object = predictionProcessor(
                    org_predictions_location = self.org_predictions_location,
                    new_predictions_path = gen_prediction_file_path,
                    images_location = self.images_location,
                    annotation_file_location = gen_annotation_file_path,
                    area_threshold_array = (lower_threshold, upper_threshold),
                    middle_boundary = self.middle_boundary,
                    model_cfg_path = self.model_config_file,
                    utils_helper = self.utils_helper,
                    mask_logit_threshold = 0.5 if flowRunner._FILTER_MASK_LOGITS else 0.0)
                prediction_processor_object.setup_objects_and_misk_variables()
                prediction_processor_object.read_predictions()
                prediction_processor_object.filter_predictions_w_wrong_area_ratio()
                prediction_processor_object.write_new_predictions_to_disk()
            else: logging.info("Bin prediction file exists. Moving to evaluation -->>")
            logging.info("Finished prediction file processing ->>")
            #---------------------------
            #---BIN-EVALUATION---
            if not os.path.exists(os.path.join(evaluation_folder, "coco_results.json")):
                tester_obj = testerObj(model_config_file = self.model_config_file,
                                       current_bin_pth_dir_path = gen_prediction_file_path,
                                       current_bin_annotation_file_path = gen_annotation_file_path,
                                       current_bin_dataset_name = gen_test_set_name,
                                       current_bin_images_path = self.images_location,
                                       utils_helper = self.utils_helper)
                tester_obj.build_model()
                tester_obj.test_model()
                tester_obj.write_results_to_disk()
            else: logging.info("Evaluation file exists. Moving to next bin (if any) -->>")
            #--------------------
            #-----------------
            self.logger.remove_temp_file_handler_and_add_main_file_handler()
            logging.info(f"Finished working on bin {lower_threshold}-{upper_threshold} in:\n{evaluation_folder}")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects_and_file_structure()
    flow_runner.run_all()