import sys, os
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

from archive.generate_image_collage.utils.util_functions import Utilities_helper as ut
from PIL import Image
from pycocotools.coco import COCO

import time
import matplotlib.pyplot as plt

def generate_gt_visualizations(plot_save_directory = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Experiment_visualization/gt_with_0.1_border_cropping"):
    overide_folder_content = False
    #Measured in pixels
    prediction_image_width = 640
    prediction_image_height = 750
    #DPI is redundant, can be any value
    dpi = 100

    annotation_file = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/annotations/gen_ann_frame_0.1_thickness/instances_val2017.json"
    images_folder = "/home/projects/bagon/shared/coco/val2017"

    ut_helper = ut()
    ut_helper.to_delete_or_not_to_delete_content(plot_save_directory, not overide_folder_content)

    coco = COCO(annotation_file)

    #Load images from coco
    img_ids = coco.getImgIds()
    print("Total number of images: ", len(img_ids), "\n")
    images = coco.loadImgs(img_ids)

    errors = []

    for img in images:
        #Start timing
        t = time.time()

        img_file_path = os.path.join(images_folder, img['file_name'])
        new_plot_path = os.path.join(plot_save_directory, img['file_name'])

        try:
            #Load image and turn into a tensor
            print("Working on image {}".format(img['file_name']))

            img_org = Image.open(img_file_path)

            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)

            original_image = Image.open(img_file_path)

            #Display prediction and compare to org
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

            figure_size_inches_width, figure_size_inches_height = ut_helper.calculate_figure_size_inches(
                prediction_image_width,
                prediction_image_height,
                dpi)
            fig.set_size_inches(figure_size_inches_width, figure_size_inches_height)

            fig.savefig(new_plot_path, dpi=dpi)
            plt.close(fig)

        except Exception as e:
            print("Error with image {}".format(img['file_name']))
            e.with_traceback()

if __name__=="__main__":
    generate_gt_visualizations()