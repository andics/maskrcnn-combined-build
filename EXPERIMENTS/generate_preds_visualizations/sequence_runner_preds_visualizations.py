import sys
import os
import dllogger

from pathlib import Path
from EXPERIMENTS.generate_preds_visualizations.utils.util_functions import Utilities_helper
from utils_gen import model_utils
from utils_gen import dataset_utils

from maskrcnn_benchmark.config import cfg
from EXPERIMENTS.generate_preds_visualizations.objects.model_class import Model_single

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")


import argparse
import logging
import functools
import gc
import time

class flowRunner:
    #The following script infers a dataset on a model and saves the predictions along with ground-truth visualizations
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-cp', '--config-path', nargs='?',
                            type=str,
                            required = True,
                            help='Path to configuration file used to build the model')
        parser.add_argument('-td', '--test-dataset', nargs='?',
                            type=str,
                            required = True,
                            help='The name of the dataset as written in the paths_catalog file')
        parser.add_argument('-sd', '--save-dir', nargs='?',
                            type=str,
                            required = False,
                            default= '/home/projects/bagon/andreyg/Projects/Variable_Resolution/Experiment_visualization/FiftyoneApp_debugging',
                            help='The folder in which the predictions should be saved')

        args = parser.parse_args()
        self.config_location_model = args.config_path
        self.test_dataset = args.test_dataset
        self.save_dir = args.save_dir

        self.objects_setup_complete = False


    def build_model(self):
        #add function which returns the model as well as the established CFG file
        #pass the model and the cfg to the test_model function in my_train_net
        #Perhaps make a local copy of my_train_net to keep everything modular and clean
        self.model, self.cfg = model_utils.load_model_and_cfg(self.config_location_model)
        self.test_dataset = (self.test_dataset,)
        self.cfg.merge_from_list(["DATASETS.TEST", self.test_dataset])

        print("Successfully loaded model weights")
        print("Using the following dataset for inference: ", cfg.DATASETS.TEST)

    def setup_objects_and_variables(self):
        self.utils_helper = Utilities_helper()
        self.predictions_save_directory = os.path.join(self.save_dir, self.test_dataset[0])
        self.utils_helper.check_dir_and_make_if_na(self.predictions_save_directory)
        datasets_ann_path, dataset_images_path = dataset_utils.get_dataset_info_from_cfg(self.cfg)

        self.modelObj = Model_single(cfg=self.cfg, model=self.model,
                                     images_location=dataset_images_path,
                                     preserve_original_dir_content=True,
                                     plot_save_dir=self.predictions_save_directory,
                                     confidence_threshold=0.5)

        print("We are here")

    def run_all(self):
        self.build_model()
        self.setup_objects_and_variables()
        self.modelObj.run()


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
