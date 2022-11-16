import os
import sys
import argparse
import time
import numpy as np
import cv2

from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from dataset_processing.generate_annotation_file_hitmap.utils.util_functions import Utilities_helper

#Appending the modules' path as necessary
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
        "--dataset-name",
        help="Specify the name of the dataset for which an annotation file is needed",
        default="Nothing",
    )
    parser.add_argument(
        "--hitmap-dimensions",
        help="Specify the dimensions of the heatmap (it is a square)",
        default=100,
        type=int,
    )

    #Define main parameters for hitmap generation
    args = parser.parse_args()
    hitmap_dimensions = (args.hitmap_dimensions, args.hitmap_dimensions)
    dataset_name = dataset_catalogue.get(args.dataset_name)["args"]

    annotation_file_path = dataset_name["ann_file"]
    images_folder = dataset_name["root"]

    coco = COCO(annotation_file_path)

    #Load images from coco
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    print("Total number of images: ", len(img_ids), "\n")
    images = coco.loadImgs(img_ids)

    errors = []

    total_binary_mask_small = np.zeros(shape=hitmap_dimensions)
    total_binary_mask_medium = np.zeros(shape=hitmap_dimensions)
    total_binary_mask_large = np.zeros(shape=hitmap_dimensions)
    total_binary_mask_all = np.zeros(shape=hitmap_dimensions)


    for img in images:
        #Start timing
        t = time.time()

        img_file_path = os.path.join(images_folder, img['file_name'])

        try:
            #Load image and turn into a tensor
            print("Working on image {}".format(img['file_name']))

            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)

            for annotation in anns:
                current_image_binary_mask = coco.annToMask(annotation)
                current_image_binary_mask_resized = cv2.resize(current_image_binary_mask,
                                                               dsize=hitmap_dimensions,
                                                               interpolation=cv2.INTER_CUBIC)
                if debugging:
                    plt.imshow(current_image_binary_mask_resized)
                    plt.show()
                total_binary_mask_all += current_image_binary_mask_resized
                if annotation['area'] <= 32**2:
                    total_binary_mask_small += current_image_binary_mask_resized
                    continue
                elif annotation['area'] <= 96**2:
                    total_binary_mask_medium += current_image_binary_mask_resized
                    continue
                else:
                    total_binary_mask_large += current_image_binary_mask_resized
                    continue

            if debugging:
                img_org = Image.open(img_file_path)
                #Display prediction and compare to org
                prediction_image_width = 640
                prediction_image_height = 750
                # DPI is redundant, can be any value
                dpi = 100
                num_plt_rows = 1
                num_plt_cols = 1
                fig, axs = plt.subplots(nrows = num_plt_rows, ncols = num_plt_cols)

                #Set axis off for all subplots
                axs.set_axis_off()
                fig.subplots_adjust(top=0.95, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)

                fig.suptitle('Ground Truth')

                axs.imshow(img_org)
                plt.axes(axs)
                coco.showAnns(anns)

                figure_size_inches_width, figure_size_inches_height = utils_helper.calculate_figure_size_inches(
                    prediction_image_width,
                    prediction_image_height,
                    dpi)
                fig.set_size_inches(figure_size_inches_width, figure_size_inches_height)

                fig.show()
                time.sleep(7)

        except Exception as e:
            print("Error with image {}".format(img['file_name']))
            e.with_traceback()

    image_bundle_pack = (
        (total_binary_mask_all, "All objects"),
        (total_binary_mask_large, "Large objects"),
        (total_binary_mask_medium, "Medium objects"),
        (total_binary_mask_small, "Small objects"),
    )
    utils_helper.display_multi_image_collage(image_bundle_pack, (1, 4))

if __name__ == "__main__":
    main()