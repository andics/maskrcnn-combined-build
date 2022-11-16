#Before filtering the annotation files based on the border region, the mechanism by which this is done requires
#that a few parameters get changed in the image info files
import argparse
import os
import sys
import glob
import json
from pathlib import Path

from dataset_processing.generate_annotation_file_hitmap.utils.util_functions import Utilities_helper
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

def main():
    utils_helper = Utilities_helper()
    dataset_catalogue = DatasetCatalog()
    debugging = True

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference & Neuronal Activation recording")
    parser.add_argument(
        "--crop-info-files-folder",
        help="Specify the folder containing the crop-info files for each image",
        default="Nothing",
    )
    parser.add_argument(
        "--border-thickness-ratio",
        help="Specify the thickness of the border as a percentage of the image width/height",
        default=0.1,
        type=float,
    )
    #Define main parameters for hitmap generation
    args = parser.parse_args()

    crop_files_folder = args.crop_info_files_folder
    border_thickness_ratio = args.border_thickness_ratio

    print("Current file path: ", Path(__file__).parent.resolve())
    print("Crop info files: ", crop_files_folder)

    for filename in glob.glob(os.path.join(crop_files_folder, '*.json')):  # only process .JSON files in folder.
        print("Processing file: ", filename)
        with open(filename, 'r+') as f:
            data = json.load(f)
            new_bbox_array = generate_new_bbox_array([data["width"], data["height"]], border_thickness_ratio)
            data["bbox"] = new_bbox_array
            f.seek(0)  # <--- should reset file position to the beginning.
            json.dump(data, f, indent=4)


def generate_new_bbox_array(current_bbox_array, border_thickness_ratio):
    return [round(number) for number in [current_bbox_array[0]*border_thickness_ratio, current_bbox_array[1]*border_thickness_ratio,
            current_bbox_array[0]*(1-border_thickness_ratio), current_bbox_array[1]*(1-border_thickness_ratio)]]


if __name__=="__main__":
    main()