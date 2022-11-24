import sys
import os
import time
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

from interactive.visualize_gt_and_predictions.utils.util_functions import Utilities_helper
from interactive.visualize_gt_and_predictions.objects.visualizor_dataset_gt_and_predictions import datasetVisualizer

import argparse

class flowRunner:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for script')

        parser.add_argument('-al', '--annotations-location', nargs='*',
                            type=str,
                            required = False,
                            help='The location(s) where the annotation file(s) are',
                            default=['/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations_polygon/instances_val2017_shifted_h_0.5_v_1.0.json'])
        parser.add_argument('-il', '--images-location', nargs='*',
                            type=str,
                            required = False,
                            help='The location where the GT(s) images are',
                            #default='/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable_shifted_h_0.5_v_1.0')
                            default=['/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/val2017_shifted_h_0.5_v_1.0'])
        parser.add_argument('-pl', '--predictions-location', nargs='*',
                            type=str,
                            required = False,
                            help='The location where the predictions are. If no predictions are wanted, assign value None',
                            default=['None'])
        parser.add_argument('-sl', '--session-location', nargs='*',
                            type=str,
                            required = False,
                            help='The location where the FiftyOne session(s) are stored OR where you wish them to be stored now',
                            default=['None'])
        parser.add_argument('-port', '--port', nargs='*',
                            type=int,
                            required = False,
                            default = [6001],
                            help='The port(s) to be used for running the visualization sessions')

        args = parser.parse_args()
        self.annotations_location = args.annotations_location
        self.images_location = args.images_location
        self.predictions_location = args.predictions_location
        #The session saving functionality is not yet implemented.
        #This is placed here tentatively in case we wish to implement it in the future
        self.session_location = args.session_location
        self.port = args.port

        #Ensure we are given the same number of sessions to start
        _prev = self.annotations_location
        for argument in [self.annotations_location, self.images_location,
                         self.predictions_location, self.port]:
            assert len(_prev) == len(argument)
            _prev = argument

        self.objects_setup_complete = False

    def run_all(self):
        if len(self.session_location) != len(self.images_location):
            self.session_location = self.session_location * len(self.images_location)

        for annotations_location, images_location,\
            port, predictions_location, session_location in zip(self.annotations_location,
                                                                                 self.images_location,
                                                                                 self.port,
                                                                                 self.predictions_location,
                                                                                 self.session_location):
            print(f"Visualizing on port: {port} \n")
            print(f"Annotations: {annotations_location} \n")
            print(f"Images: {images_location} \n")
            print(f"Predictions: {predictions_location} \n")
            dataset_visualizer = self.setup_objects_and_variables_for_one_session(annotations_location,
                                                                                  images_location, port,
                                                                                  predictions_location, session_location)
            print(f"Successfully initialized visualiser with port {port}! Proceeding to dataset loading ...")

            dataset_visualizer.load_dataset()
            if not predictions_location == "None":
                dataset_visualizer.load_predictions()
                dataset_visualizer.add_prediction_fields_to_dataset()
            dataset_visualizer.run_visualization_gt()
            print(f"Successfully ran visualiser with port {port}! You may now connect!")

        time.sleep(43200)


    def setup_objects_and_variables_for_one_session(self, annotations_location, images_location,\
            port, predictions_location, session_location):
        utils_helper = Utilities_helper()

        dataset_visualizer = datasetVisualizer(annotation_file_path=annotations_location,
                                                    gt_images_base_path = images_location,
                                                    port = port,
                                                    predictions_file_path = predictions_location,
                                                    utils_helper = utils_helper,
                                                    pickle_save_path = session_location)
        return dataset_visualizer


if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.run_all()
