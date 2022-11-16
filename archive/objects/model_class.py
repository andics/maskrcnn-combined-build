import sys

sys.path.append('/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp')

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from maskrcnn_benchmark.config import cfg
from demo.predictor_custom import COCO_predictor
from utils.CustomTextBox import ExtendedTextBox
from matplotlib.path import Path
from matplotlib.patches import BoxStyle
from utils.util_functions import Utilities_helper as ut

BoxStyle._style_list["ext"] = ExtendedTextBox

import math
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import itertools

class Model_single(object):
    """
    Class used to infer individual images through MaskRCNN
    Likely used in older project builds; currently not used in of the maskrcnn_combined files
    """

    expected_image_extensions = (".jpg", ".png")
    static_variables = ["cfg"]

    columns_on_pred_table = 3

    #Measured in pixels
    prediction_image_width = 640
    prediction_image_height = 1100
    #DPI is redundant, can be any value
    dpi = 100

    #NOT including total_num_predictions
    max_num_preds = 23
    show_bboxes = False

    choose_top_n_if_none = True
    #Under how many predictions should the tresholded ones be to take the top N ones instead?
    critical_number_to_select_top_n_predictions = 5
    top_n_if_none = 5

    def __init__(self,
                 config_path,
                 model_weight,
                 model_name,
                 images_location,
                 preserve_original_dir_content,
                 base_plot_save_dir,
                 confidence_threshold
                 ):

        #Config path is irrelevant - can be any config
        self.config_path = config_path

        self.model_weight = model_weight
        self.model_name = model_name
        self.cfg = cfg
        self.ut = ut()

        self.images_location = images_location

        self.plot_save_dir = os.path.join(base_plot_save_dir, model_name)
        self.ut.create_folder_if_none_exists(self.plot_save_dir)
        self.preserve_original_dir_content = preserve_original_dir_content
        self.ut.to_delete_or_not_to_delete_content(self.plot_save_dir, self.preserve_original_dir_content)

        self.confidence_threshold = confidence_threshold

        self.recall_specified_parameters()


    def run(self):
        self.load_model()
        self.gather_images()
        self.run_pred_on_all_images()

    def load_model(self):
        self.model_predictor = COCO_predictor(cfg=self.cfg, custom_config_file=self.config_path,\
                                         weight_file_dir=self.model_weight, \
                                         use_conf_threshold=True, confidence_threshold=self.confidence_threshold, \
                                              max_num_pred=Model_single.max_num_preds, min_image_size=60,
                                              masks_per_dim=3, show_bboxes=Model_single.show_bboxes,
                                              top_if_none_critical_number = Model_single.critical_number_to_select_top_n_predictions,
                                              choose_top_n_if_none=Model_single.choose_top_n_if_none,
                                              top_n_if_none=Model_single.top_n_if_none)

        print("Loaded model %s!" % self.model_name)


    def recall_specified_parameters(self):

        print("------------ PASSED ARGUMENTS ------------")

        attrs = vars(self)

        for variable in attrs.items():
            if variable[0] not in Model_single.static_variables:
                print("%s = %s" % variable)


    def gather_images(self):
        self.images_names_w_extensions = []

        for file in os.listdir(self.images_location):
            if file.lower().endswith(self.expected_image_extensions):
                self.images_names_w_extensions.append(file)

        print("Gathered images for model %s!" % self.model_name)


    def make_pandas_prediction_dataframe(self, data_for_columns):
        number_of_columns = len(data_for_columns)

        print("Preparing Pandas dataframe!")
        #print("----DATA FOR COLUMNS----")
        #print(data_for_columns)

        column_names = []
        for i in range(number_of_columns):
            column_names.append("Predictions " + str(i+1))

        #print("Prepared column names: ", column_names)

        prediction_dataframe = pd.DataFrame((_ for _ in itertools.zip_longest(*data_for_columns)), columns=column_names)

        dfStyler = prediction_dataframe.style.set_properties(**{'text-align': 'left'})
        dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

        return prediction_dataframe

    def transform_predictions_dict(self, pred_dictionaries):
        print("Transforming prediction dictionary!")

        columns_to_prepare = Model_single.columns_on_pred_table

        #If predictions is < than num_columns, the columns that can't be filled are returned empty!
        prediction_sublists = self.ut.split_list_into_chunks(pred_dictionaries, columns_to_prepare)

        prediction_dataframe = self.make_pandas_prediction_dataframe(prediction_sublists)
        prediction_table_string_format = prediction_dataframe.to_string(header = False, index=False)

        return prediction_dataframe


    def run_pred_on_all_images(self):
        images = self.images_names_w_extensions
        images_base_dir = self.images_location
        model_predictor = self.model_predictor
        new_images_base_dir = self.plot_save_dir
        self.ut.create_folder_if_none_exists(new_images_base_dir)

        errors = []
        images_saved_paths = []

        for img in images:
            # Start timing
            t = time.time()

            current_fig_path = os.path.join(new_images_base_dir, img)
            if self.ut.path_exists(path_to_check=current_fig_path):
                print("Skipped  image {} as it already exists!".format(img))
                images_saved_paths.append(current_fig_path)
                continue

            img_file_path = os.path.join(images_base_dir, img)
            print("Predicting on image: ", img_file_path, "\n")

            try:
                # Load image and turn into a tensor
                img_org = Image.open(img_file_path)
                img_org_rgb = img_org.convert("RGB")
                # convert to BGR format
                tensor_image = np.array(img_org_rgb)  # [:, :, [2, 1, 0]]

                # Generate model predictions with predeictor_custom
                predictions, predictions_dictionary = model_predictor.run_on_opencv_image(img_org)

                # Display prediction and compare to org
                num_plt_rows = 2
                num_plt_cols = 1
                fig, axs = plt.subplots(nrows=num_plt_rows, ncols=num_plt_cols)

                # Set axis off for all subplots
                [axi.set_axis_off() for axi in axs.ravel()]

                fig.suptitle('Model {}'.format(self.model_name))

                axs[0].imshow(img_org_rgb)
                plt.axes(axs[0])
                axs[0].imshow(predictions)

                # Prepare dictionary text box
                plt.axes(axs[1])

                box_annotations = self.transform_predictions_dict(predictions_dictionary)

                plt.axes(axs[1])
                cell_text = []
                for row in range(len(box_annotations)):
                    cell_text.append(box_annotations.iloc[row])

                axs[1].table(cellText=cell_text, colLabels=None, loc='center')

                #Determines the ration of space that each subplot element should take as part of the entire subplot area
                fig.subplots_adjust(top=0.95, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                for axi in axs.ravel():
                    #Removes the scales from the axis
                    axi.xaxis.set_major_locator(plt.NullLocator())
                    axi.yaxis.set_major_locator(plt.NullLocator())

                figure_size_inches_width, figure_size_inches_height = self.ut.calculate_figure_size_inches(Model_single.prediction_image_width,
                                                                                                   Model_single.prediction_image_height,
                                                                                                   Model_single.dpi)
                fig.set_size_inches(figure_size_inches_width, figure_size_inches_height)

                fig.savefig(current_fig_path, bbox_inches='tight',
                            pad_inches=0, dpi=Model_single.dpi)
                plt.close(fig)

                images_saved_paths.append(current_fig_path)

                print('Saved : ', img, '\n')
                elapsed = time.time() - t
                print('Elapsed time: ', str(elapsed), '\n')
            except Exception as e:
                e.with_traceback()
                print("Error during the processing of image ", img, " : ", e)
                errors.append([e, img])

        self.images_saved_paths = images_saved_paths

        print(errors)







