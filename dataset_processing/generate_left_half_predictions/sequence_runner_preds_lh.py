import sys
import os
from pathlib import Path
try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from dataset_processing.generate_left_half_predictions.objects.logger_obj import Logger
from dataset_processing.generate_left_half_predictions.utils.util_functions import Utilities_helper
from dataset_processing.generate_left_half_predictions.objects.prediction_processor_obj import predictionProcessor

import argparse
import logging
import functools
import gc
import time

class flowRunner:
    def __init__(self):
        '''
        The following script is used to trim the predictions produced by the default model-image inference to only the left-half of the image.
        This is useful for removing hallucinations of the model.
        '''
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-f', '--config-path', nargs='?',
                            type=str,
                            required = True,
                            help='Path to configuration file to be used for prediction half-image generation')

        args = parser.parse_args()
        self.config_location = args.config_path

        self.objects_setup_complete = False
        self.setup_objects()


    def run_all(self):
        self.prediction_processor.setup_new_predictions_folder_structure()
        self.prediction_processor.read_predictions()
        self.prediction_processor.setup_objects()
        self.prediction_processor.filter_predictions_outside_border()
        self.prediction_processor.write_new_predictions_to_disk()


    def setup_objects(self):
        self.utils_helper = Utilities_helper()

        self.utils_helper.change_yaml_file_value(self.config_location, ['LOGGING', 'logs_subdir'],
                                                 self.utils_helper.extract_filename_and_ext(self.config_location)[0])
        self.config_file = self.utils_helper.read_yaml_data(self.config_location)

        self.logger = Logger(config_file=self.config_file, utils_helper=self.utils_helper)
        self.prediction_processor = predictionProcessor(self.config_file, self.utils_helper, self.logger)
        self.logger.log("Finished setting up objects")


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
