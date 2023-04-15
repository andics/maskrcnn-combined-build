import sys
import os
import logging
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

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

from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.trail_runner_obj import trialRunnerObj
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.utils.util_functions import Utilities_helper
from EXPERIMENTS.bin_eval_per_obj_type_ann_norm_small_med.objects.main_logger_obj import loggerObj

import argparse

class flowRunner:
    #Some default (usually unnecessary to change) parameters
    _MASKRCNN_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    _FLOW_RUNNER_PARENT_DIR_ABSOLUTE = str(Path(os.path.dirname(os.path.realpath(__file__))))
    _TRIAL_SUBFOLDERS_TEMPLATE = "trial_%s"
    _LOG_LEVEL = logging.DEBUG


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
                            type = int,
                            default=[100],
                            help='The edge size of the middle square we define to have high-resolution')
        parser.add_argument('-bs', '--bin-spacing', nargs='?',
                            type=float,
                            default=0.04,
                            required = False,
                            help='(% / 100) The space between each resolution bin.'
                                 'E.g. If this paramter is set to 0.1, one can expect that the paradigm'
                                 'will generate 10 and evaluate 10 bins, starting from 0.0-0.1 and ending with'
                                 '0.9-1.0'
                                 'IMPORTANT: This parameter is also appended to the name of the'
                                 'folder in which this experiment is stored')
        parser.add_argument('-fp', '--filter-preds', nargs='?',
                            type=str,
                            default="False",
                            required = False,
                            help='Whether to filter the prediction files (True),'
                                 ' or only the annotation files (False).'
                                 'This measure was implemented due to suspected bias in the eval'
                                 'stemming from the different number of objects in each pred. bin')
        parser.add_argument('-pan', '--perform-annotation-normalization', nargs='?',
                            type=str,
                            default="False",
                            required = False,
                            help='The annotation normalization includes taking the smallest number of objects'
                                 'present in each annotation file after filtering, and selecting from ALL'
                                 'other annotation files a random subset of the same number of objects.'
                                 'Thereafter, preforming the evaluations on those subsets.')
        parser.add_argument('-anf', '--annotation-normalization-factor', nargs='?',
                            default=0.9,
                            type=float,
                            required = False,
                            help='The smallest number of annotations for some obj. size present in single bin will'
                                 'be the size of the subsample we randomly extract from each other bin, for each size,'
                                 'in order to normalize the annotations. However, we multiply this size by the'
                                 'normalization ratio: so that there is some randomness in the smallest'
                                 ' bin also. '
                                 'However, this parameter can also be an integer > 1. Then: exactly that '
                                 'many random objects will be picked from each annotation bin')
        parser.add_argument('-anrs', '--annotation-normalization-random-seed', nargs='?',
                            default=-1,
                            type=int,
                            required = False,
                            help='The Misk annotation processor picks out an anf*smallest_num_objs_in_bin number of'
                                 'objects of each type (small, med, large) from each bin. Those objects can be the'
                                 ' same every time if the random seed is the same. If anrs is not set (is -1),'
                                 'the set of picked objects will be the random every time.'
                                 'This parameter should be provided in decile numbers only, due to'
                                 'the internal double-randomization (e.g. anrs = 10, 20...)')
        parser.add_argument('-anlp', '--annotation-normalization-large-objects-present', nargs='?',
                            type = str,
                            default = "False",
                            required = False,
                            help='The filtered bin annotation files sometimes have 0 large objects,'
                                 ' or very few. Those few large objects impede the annotation normalization'
                                 'process, as they enforce a smaller number of annotations for small/medium'
                                 'objects. By default, we do not include large objects')
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
        parser.add_argument('-efi', '--experiment-folder-identificator', nargs='?',
                            type=str,
                            default = "ann_norm",
                            required = False,
                            help='As the amount of tunable parameters of this script grows, it is needed'
                                 'to have some idea of what an experiment folder contains. This variable'
                                 'allows one to append any string the the final experiment folder name')
        #---
        parser.add_argument('-nt', '--num-trials', nargs='?',
                            default=1,
                            type=int,
                            required = False,
                            help='The number of trials with different subsets of the normalized annotations '
                                 'which will be performed. Each trials will be stored in its own trial_n folder')

        self.args = parser.parse_args()
        assert(self.args.perform_annotation_normalization == "True" or
               self.args.perform_annotation_normalization == "False")
        assert(self.args.filter_preds == "True" or
               self.args.filter_preds == "False")
        assert(self.args.annotation_normalization_large_objects_present == "True" or
               self.args.annotation_normalization_large_objects_present == "False")

        self.model_name = self.args.model_name
        self.model_config_file = self.args.model_config_file
        self.middle_boundary = self.args.middle_boundary[0]
        self.bin_spacing = self.args.bin_spacing
        self.filter_preds = True if \
            self.args.filter_preds == "True" else False
        self.perform_annotation_norm = True if \
            self.args.perform_annotation_normalization == "True" else False
        self.annotation_normalization_large_objects_present = True if \
            self.args.annotation_normalization_large_objects_present == "True" else False
        self.annotation_normalization_factor = self.args.annotation_normalization_factor
        self.annotation_normalization_random_seed = self.args.annotation_normalization_random_seed if \
            not self.args.annotation_normalization_random_seed == -1 else "None"
        self.org_annotations_location = self.args.org_annotations_location
        self.images_location = self.args.images_location
        self.org_predictions_location = self.args.org_predictions_location
        self.parent_storage_location = self.args.parent_storage_location
        self.experiment_folder_identificator = self.args.experiment_folder_identificator
        self.num_trials = self.args.num_trials

        self.experiment_name = self.model_name + "_" + str(float(self.bin_spacing)) + "_" +\
                               str(self.middle_boundary) + "_" + self.experiment_folder_identificator

        self.main_file_dir = str(Path(os.path.dirname(os.path.realpath(__file__))))
        self.objects_setup_complete = False

    def setup_objects_and_file_structure(self):
        self.utils_helper = Utilities_helper()
        self.main_experiment_dir = os.path.join(self.parent_storage_location,
                                                 self.experiment_name)
        self.utils_helper.check_dir_and_make_if_na(self.main_experiment_dir)

        #TODO: set up a logger
        self.logger_ref = loggerObj(logs_subdir=self.main_experiment_dir,
                                log_file_name="log",
                                utils_helper=self.utils_helper,
                                log_level=flowRunner._LOG_LEVEL,
                                name="flow_logger")
        self.logger = self.logger_ref.setup_logger()
        self.logger.info("Passed arguments -->>")
        self.logger.info('\n  -  '+ '\n  -  '.join(f'{k}={v}' for k, v in vars(self.args).items()))
        self.logger.info(f"  -  Main experiment folder: {self.main_experiment_dir}")

    def generate_trial_folders(self):
        self.trial_folders = []

        for trial_i in range(self.num_trials):
            _current_trial_folder = os.path.join(self.main_experiment_dir,
                                                 flowRunner._TRIAL_SUBFOLDERS_TEMPLATE % str(trial_i))
            self.trial_folders.append(_current_trial_folder)


    def run_all_trails(self):
        self.trial_objects = []

        for i, trial_folder in enumerate(self.trial_folders):
            self.logger.info(f"Working on Trial #{i};")
            current_trial_object = trialRunnerObj(model_name = self.model_name,
                 model_config_file = self.model_config_file,
                 middle_boundary = self.middle_boundary,
                 bin_spacing = self.bin_spacing,
                 filter_preds = self.filter_preds,
                 perform_annotation_norm = self.perform_annotation_norm,
                 annotation_normalization_large_objects_present = self.annotation_normalization_large_objects_present,
                 annotation_normalization_factor = self.annotation_normalization_factor,
                 annotation_normalization_random_seed = self.annotation_normalization_random_seed,
                 org_annotations_location = self.org_annotations_location,
                 images_location = self.images_location,
                 org_predictions_location = self.org_predictions_location,
                 experiment_dir = trial_folder,
                 num_trials = self.num_trials,
                 utils_helper = self.utils_helper,
                 current_trial_number = i)

            _prev_trail_folder = self.trial_folders[i-1] if i>0 else None

            current_trial_object.setup_objects_and_file_structure()
            current_trial_object.run_recycler(_prev_trail_folder)
            current_trial_object.run_all_vanilla()
            current_trial_object.run_all_misk()
            current_trial_object.logger.factory_reset_logger()

            self.logger.info(f"Finished working on Trial #{i}. Moving to next (if any)...")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects_and_file_structure()
    flow_runner.generate_trial_folders()
    flow_runner.run_all_trails()