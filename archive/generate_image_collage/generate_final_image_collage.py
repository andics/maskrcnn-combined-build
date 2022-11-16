from utils.util_functions import Utilities_helper as ut
from PIL import Image

import os
import sys
import matplotlib.pyplot as plt

def generate_final_image_collage(general_folder_with_models_predictions = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Experiment_visualization/comparative_visualization_min_5",
    folder_to_save_prediction_collage = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Experiment_visualization/comparative_visualization_min_5/Final_collage"):
    util_helper = ut()

    collage_image_width = 2700
    collage_image_height = 4200
    dpi = 100

    model_names_in_order_of_display = (
        "Baseline_resnet101",
        "Constant_blurry_resnet101",
        "Variable_res_resnet101",
        "Multi_stacked_resnet101",
        "Single_CH3_resnet101",
        "Single_CH2_resnet101",
        "Single_CH1_resnet101",
        "Ground_truth",
        "MultiMixed_CH3_resnet101",
        "MultiMixed_CH2_resnet101",
        "MultiMixed_CH1_resnet101",
        "Original_images",
    )

    lists_of_common_folder_files = find_images_in_common_for_models(general_folder_with_models_predictions,
                                                                    (model_names_in_order_of_display[0], model_names_in_order_of_display[3]),
                                                                    util_helper)
    print("Number of images in common: ", len(lists_of_common_folder_files))

    for image_name in lists_of_common_folder_files:
        image_collage_save_path = os.path.join(folder_to_save_prediction_collage, image_name)

        if os.path.exists(image_collage_save_path):
            print("Image {} already exists! Skipping...".format(image_name))
            continue

        print("Working on image: ", image_name)
        # Display prediction and compare to org
        num_plt_rows = 3
        num_plt_cols = 4
        fig, axs = plt.subplots(nrows=num_plt_rows, ncols=num_plt_cols)

        [axi.set_axis_off() for axi in axs.ravel()]
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0.05, 0)

        figure_size_inches_width, figure_size_inches_height = util_helper.calculate_figure_size_inches(
            collage_image_width,
            collage_image_height,
            dpi)
        fig.set_size_inches(figure_size_inches_width, figure_size_inches_height)

        ax_counter_horizontal = 0
        ax_counter_vertical = 0
        for model in model_names_in_order_of_display:
            path_of_current_image_to_add = os.path.join(general_folder_with_models_predictions, model, image_name)
            current_image_to_add = Image.open(path_of_current_image_to_add)

            if ax_counter_horizontal % num_plt_cols == 0 and ax_counter_horizontal != 0:
                ax_counter_vertical += 1
                ax_counter_horizontal = 0

            #print("Trying to access: ", ax_counter_vertical, " - ", ax_counter_horizontal)
            plt.axes(axs[ax_counter_vertical, ax_counter_horizontal])
            axs[ax_counter_vertical, ax_counter_horizontal].imshow(current_image_to_add)

            ax_counter_horizontal += 1


        fig.savefig(image_collage_save_path, dpi=dpi)
        plt.close(fig)


def find_images_in_common_for_models(general_folder_with_models_predictions, model_names_in_order_of_display, util_helper):
    list_of_lists_of_folder_files_in_order = []

    for model_folder in model_names_in_order_of_display:
        current_model_predictions_folder = os.path.join(general_folder_with_models_predictions, model_folder)
        current_subfolder_image_files = util_helper.gather_subfiles(current_model_predictions_folder)

        list_of_lists_of_folder_files_in_order.append(current_subfolder_image_files)
        print("Finished gathering images for model", model_folder)

    lists_of_common_folder_files = util_helper.find_common_elements_between_n_lists(list_of_lists_of_folder_files_in_order)
    print("Number of files found in common: ", len(lists_of_common_folder_files))

    return lists_of_common_folder_files


if __name__=="__main__":
    generate_final_image_collage()