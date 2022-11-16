import argparse
import os

from EXPERIMENTS.record_neuronal_activations.model_predictor_obj import Model_single
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.tester import test

from maskrcnn_benchmark.utils.logger import format_step

from EXPERIMENTS.record_neuronal_activations.utils.util_functions import Utilities_helper as utils_exp_specific
# Import experiment-specific utils function

import training.cfg_prep as cfg_prep
import utils_gen.model_utils as utils

class sequenceRunner():

    def __init__(self):
        #Define essentials
        self.script_rood_dir = os.path.realpath(__file__)

        parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference & Neuronal Activation recording")
        parser.add_argument(
            "--model-sequence",
            help="Specify neural network type",
            action="store",
        )
        parser.add_argument(
            "--model-specific-sequence",
            help="For particular models, allows real-time neural network building",
            action="store",
            default="Nothing",
        )
        parser.add_argument(
            "--config-path",
            help="Specify a config file",
            default="Nothing",
        )
        parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )
        #-----UNUSED-BUT-NECESSARY-----
        parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
        args = parser.parse_args()

        print("Working with the following arguments: ", args)
        self.model_sequence = args.model_sequence
        self.model_specific_sequence = args.model_specific_sequence

        #---Prepare new config_file---
        if args.config_path is "Nothing":
            self.custom_config_file_path = cfg_prep.prepare_new_config_file_from_sequence(cfg, self.model_sequence, self.model_specific_sequence)
        else:
            self.custom_config_file_path = args.config_path

        config_file = self.custom_config_file_path
        self.cfg = cfg.clone()
        self.cfg.merge_from_file(config_file)
        self.cfg.merge_from_list(args.opts)
        self.cfg.freeze()

        self.setup_helper_obj()

        self.model = self.initialize_model()
        self.model.run()


    def initialize_model(self):

        self.model_name = self.model_sequence + "_" + self.model_specific_sequence
        self.model_final_weight = self.ut_exp.get_last_checkpoint_in_dir(self.cfg.OUTPUT_DIR)
        self.universal_plot_save_path = self.cfg.MODEL_PREDICTOR.UNIVERSAL_SAVE_DIR
        #The dataset's image folder and anootation_file location are
        #recorded as dictionary elements of the get() method of this class
        self.dataset_catalog = DatasetCatalog()

        self.dataset_location_images = self.dataset_catalog.get(self.cfg.MODEL_PREDICTOR.DATASET_NAME)["args"]["root"]
        self.dataset_location_annotation_file = self.dataset_catalog.get(self.cfg.MODEL_PREDICTOR.DATASET_NAME)["args"]["ann_file"]
        #TODO: Make utils script for finding model weight
        #Input the correct cfg parameters into this constructor
        #MAYBE: chenge the default parameters inside the class to also be inferred from the config file

        return Model_single(config_path = self.custom_config_file_path,
                     model_weight = self.model_final_weight,
                     model_name = self.model_name,
                     images_location = self.dataset_location_images,
                     preserve_original_dir_content = False,
                     base_plot_save_dir = self.universal_plot_save_path,
                     confidence_threshold = 0.75)


    def setup_helper_obj(self):
        #Util functions for the specific experiment
        self.ut_exp = utils_exp_specific()


if __name__=="__main__":
    sequence_runner = sequenceRunner()