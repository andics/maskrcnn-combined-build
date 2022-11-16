import numpy as np
import sys
import os
import argparse
import random
from pathlib import Path

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
    image_extensions = (".jpg", ".jpeg", ".JPEG", ".JPG")

    #Folders should be given to include the train/val direcotories
    parser = argparse.ArgumentParser(description="Dataset statistics calculator")
    parser.add_argument(
        "--datasets-base-path",
        help="Specify the datasets Base path",
        default="Nothing",
    )
    parser.add_argument(
        "--channels-subdirs",
        nargs="+",
        help="Within the Dataset Base Path, which are the separate channel sub-directories?"
             "Provide from channel 1 to channel 3",
        default=["ch1", "ch2", "ch3"],
        #Channel sub-dirs can be a chain of sub-dirs
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        help="Where should the new Dataset be created",
        type=str,
    )
    args = parser.parse_args()

    datasets_base_path = args.datasets_base_path
    dataset_type = "imagenet" if "imagenet" in datasets_base_path else "coco"

    folder_ch1 = os.path.join(datasets_base_path, args.channels_subdirs[0])
    folder_ch2 = os.path.join(datasets_base_path, args.channels_subdirs[1])
    folder_ch3 = os.path.join(datasets_base_path, args.channels_subdirs[2])

    list_of_files_full = collect_desired_subdirs_w_images(datasets_base_path, image_extensions)


def collect_desired_subdirs_w_images(dataset_path, images_extensions):
    all_subdirs_w_images = []

    print("Gathering images from {}".format(dataset_path))

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for filename in [f for f in filenames if f.endswith(images_extensions)]:
            all_subdirs_w_images.append(dirnames + "/" + filename)

    print("Gathered " + str(len(all_subdirs_w_images)) + " images for the sample")

    return all_subdirs_w_images


if __name__=="__main__":
    main()