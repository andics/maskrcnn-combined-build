import sys
import os
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

from archive.record_and_save_featuremaps_variable_v2.utils.util_functions import Utilities_helper
from archive.record_and_save_featuremaps_variable_v2.objects.inference_processor_obj import inferenceProcessor

import argparse


class flowRunner:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-dl', '--dataset-location', nargs='?',
                            type=str,
                            required = False,
                            help='The location where the dataset to be shifted is located',
                            default='/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable_shifted_h_0.5_v_1.0')
        parser.add_argument('-sl', '--storage-location', nargs='?',
                            type=str,
                            required = False,
                            help='The new location where the prediction featuremaps will be stored',
                            default='/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/EXPERIMENTS/record_and_save_featuremaps_variable_v2/test_1')
        parser.add_argument('-mp', '--model-config-path', nargs='?',
                            type=str,
                            required = False,
                            help='The location where the dataset to be shifted is located',
                            default='/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/configs/R-101-FPN/variable_pretrained_resnet/variable_pretrained_resnet_baseline_resnet_norm.yaml')


        args = parser.parse_args()
        self.dataset_location = args.dataset_location
        self.storage_location = args.storage_location
        self.model_config_path = args.model_config_path

        self.objects_setup_complete = False
        self.setup_objects_and_variables()


    def run_all(self):
        self.image_processor.load_model()
        self.image_processor.attach_hooks_to_model()
        self.image_processor.read_all_images_in_org_dataset()
        self.image_processor.process_all_images()

    def setup_objects_and_variables(self):
        self.utils_helper = Utilities_helper()

        self.complete_storage_location = os.path.join(self.storage_location,
                                                      self.utils_helper.extract_folder_name_from_path(self.dataset_location))
        self.utils_helper.check_dir_and_make_if_na(self.complete_storage_location)

        self.image_processor = inferenceProcessor(utils_helper=self.utils_helper, org_dataset_folder=self.dataset_location,
                                                  new_dataset_folder=self.complete_storage_location,
                                                  model_config_path=self.model_config_path)


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
