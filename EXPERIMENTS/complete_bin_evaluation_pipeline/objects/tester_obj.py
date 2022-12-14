import sys
import os
import dllogger

from pathlib import Path
try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    sys.path.append(path_main)
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from testing.test_model_classic.utils.util_functions import Utilities_helper
from utils_gen import model_utils
from testing.test_model_classic import test_functions

from maskrcnn_benchmark.config import cfg


import argparse
import logging
import functools
import gc
import time

class flowRunner:
    #TODO:
    #Combine this file with the test_functions one
    #Sneaky sneaky just replace the dataset name, and the predictions.pth file in the testing function
    #From what I saw all else should remain the same
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-cm', '--config-path-model', nargs='?',
                            type=str,
                            required = True,
                            help='Path to configuration file used to build the model')
        parser.add_argument('-td', '--test-dataset', nargs='?',
                            type=str,
                            required = True,
                            help='The name of the borderised dataset as written in the paths_catalog file')

        args = parser.parse_args()
        self.config_location_model = args.config_path_model
        self.test_dataset = args.test_dataset

        self.objects_setup_complete = False
        self.setup_objects()


    def build_model(self):
        #add function which returns the model as well as the established CFG file
        #pass the model and the cfg to the test_model function in my_train_net
        #Perhaps make a local copy of my_train_net to keep everything modular and clean
        self.model, self.cfg = model_utils.load_model_and_cfg(self.config_location_model)
        self.test_dataset = (self.test_dataset,)
        self.cfg.merge_from_list(["DATASETS.TEST", self.test_dataset])

        print("Successfully loaded model weights")
        print("Using the following dataset for testing: ", cfg.DATASETS.TEST)

    def test_model(self):
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        distributed = num_gpus > 1

        dllogger.init(backends=[])

        dllogger_initialized = True
        dllogger.log(step="PARAMETER", data={"gpu_count": num_gpus})
        # dllogger.log(step="PARAMETER", data={"environment_info": collect_env_info()})
        dllogger.log(step="PARAMETER", data={"config_path": self.config_location_model})
        with open(self.config_location_model, "r") as cf:
            config_str = "\n" + cf.read()
        dllogger.log(step="PARAMETER", data={"config": self.cfg})

        dllogger.log(step="INFORMATION", data="Running evaluation...")
        test_functions.test_model(cfg=self.cfg, model=self.model, distributed=distributed,
                                  dllogger=dllogger, iters_per_epoch=1,
                                  current_iterations=self.cfg.SOLVER.MAX_ITER)

    def setup_objects(self):
        self.utils_helper = Utilities_helper()


    def run_all(self):
        self.setup_objects()
        self.build_model()
        self.test_model()


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
