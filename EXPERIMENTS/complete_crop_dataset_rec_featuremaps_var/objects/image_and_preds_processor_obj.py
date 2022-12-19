import json, os
import copy
import numpy as np
import cv2
import glob
import math
import logging
import torch

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter

from maskrcnn_benchmark.config import cfg
from EXPERIMENTS.complete_crop_dataset_rec_featuremaps_var.objects.predictor_custom import COCO_predictor

featuremap_current = []

def hook_module_backbone_layer3_last_bn(m, i, o):
    global featuremap_current

    featuremap_current = []
    featuremap_current = copy.deepcopy(o[0, :, :, :]).cpu().squeeze()


class imageAndPredictionProcessor:
    _DEBUGGING = True
    _VALID_IMG_EXT = ['jpg', 'png']

    def __init__(self, org_dataset_folder, new_dataset_folder, utils_helper,
                 lower_lengths, upper_lengths, model_config_path, default_vertical_cropping_percentage):
        self.org_dataset_folder = org_dataset_folder
        self.new_dataset_folder = new_dataset_folder
        self.utils_helper = utils_helper
        self.lower_lengths = lower_lengths
        self.upper_lengths = upper_lengths
        self.model_config_path = model_config_path
        self.default_vertical_cropping_percentage = default_vertical_cropping_percentage

    def load_model(self):
        cfg.merge_from_file(self.model_config_path)
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
        self.model_predictor = COCO_predictor(cfg=cfg, custom_config_file=self.model_config_path, \
                                         weight_file_dir = cfg.OUTPUT_DIR, \
                                         use_conf_threshold=False, max_num_pred=5, min_image_size=60, masks_per_dim=3)
        self.model = self.model_predictor.model
        logging.info("Model loaded! \n")
        logging.info(self.model)


    def attach_hooks_to_model(self):
        self._module_backbone_layer3_last_bn = self.model._modules.get('backbone')._modules.get('body')._modules.get('layer3').\
            _modules.get('22')._modules.get('bn3')
        self.hook_module = self._module_backbone_layer3_last_bn.register_forward_hook(hook_module_backbone_layer3_last_bn)

    def read_all_images_in_org_dataset(self):
        dir_path = self.org_dataset_folder[0]
        # list to store files
        res = []
        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                res.append(path)

        self.all_images_filenames = res


    def process_all_images(self):
        global featuremap_current
        total_img_num = len(self.all_images_filenames)
        for i, current_image_filename in enumerate(self.all_images_filenames):
            org_img_complete_path_dataset_0_shifted = os.path.join(self.org_dataset_folder[0], current_image_filename)
            org_img_complete_path_dataset_1_centered = os.path.join(self.org_dataset_folder[1], current_image_filename)

            assert os.path.exists(org_img_complete_path_dataset_0_shifted)
            assert os.path.exists(org_img_complete_path_dataset_1_centered)

            #---PREPARE-SAVE-FILE-PATHS---
            org_img_file_name, org_img_ext = self.utils_helper.extract_filename_and_ext(org_img_complete_path_dataset_0_shifted)
            current_img_general_storage_dir = os.path.join(self.new_dataset_folder,
                                        org_img_file_name)
            if os.path.exists(current_img_general_storage_dir):
                logging.info(f"Folder for image {org_img_file_name} exists here: \n {current_img_general_storage_dir}"
                             "\nProceeding...")
                continue
            else:
                self.utils_helper.check_dir_and_make_if_na(current_img_general_storage_dir)

            img_0_shifted_cropped_and_pasted_save_dir = os.path.join(current_img_general_storage_dir,
                                                                     org_img_file_name + "_0." + org_img_ext)
            img_1_centered_cropped_and_pasted_save_dir = os.path.join(current_img_general_storage_dir,
                                                                     org_img_file_name + "_1." + org_img_ext)
            tensor_0_shifted_cropped_and_pasted_save_dir = os.path.join(current_img_general_storage_dir,
                                                                     "predictions_0.pth")
            tensor_1_centered_cropped_and_pasted_save_dir = os.path.join(current_img_general_storage_dir,
                                                                     "predictions_1.pth")
            tensor_plot_0_shifted_cropped_and_pasted_save_dir = os.path.join(current_img_general_storage_dir,
                                                                     "predictions_0_vis.jpg")
            tensor_plot_1_centered_cropped_and_pasted_save_dir = os.path.join(current_img_general_storage_dir,
                                                                     "predictions_1_vis.jpg")
            img_collage_plot_save_path =  os.path.join(current_img_general_storage_dir,
                                                       "visual_comp.jpg")
            #-----------------------------

            #---IMAGE-PROCESSING-AND-SAVING---
            img_img_format_0_shifted = Image.open(org_img_complete_path_dataset_0_shifted).convert("RGB")
            img_img_format_1_centered = Image.open(org_img_complete_path_dataset_1_centered).convert("RGB")

            img_img_format_0_shifted_cropped_and_pasted_PIL_format = self.crop_and_paste_image(img_img_format_0_shifted,
                                                                                               self.lower_lengths[0],
                                                                                               self.upper_lengths[0])
            img_img_format_0_shifted_cropped_and_pasted_PIL_format.save(img_0_shifted_cropped_and_pasted_save_dir)
            img_img_format_1_centered_cropped_and_pasted_PIL_format = self.crop_and_paste_image(img_img_format_1_centered,
                                                                                               self.lower_lengths[1],
                                                                                               self.upper_lengths[1])
            img_img_format_1_centered_cropped_and_pasted_PIL_format.save(img_1_centered_cropped_and_pasted_save_dir)
            logging.info(f"Successfully saved both processed images 0 and 1 for image {org_img_file_name}")
            #---------------------------------

            #---LOAD-IMAGES-FOR-INFERENCE---
            img_img_format_0_shifted_cropped_and_pasted_PIL_format = Image.open(
                img_0_shifted_cropped_and_pasted_save_dir).convert("RGB")
            img_img_format_1_centered_cropped_and_pasted_PIL_format = Image.open(
                img_1_centered_cropped_and_pasted_save_dir).convert("RGB")

            #Location to take pixels from in original image
            #FEED-IMAGE-TO-MODEL
            logging.info(f"Feeding image 0 into model")
            self.model_predictor.run_on_opencv_image(img_img_format_0_shifted_cropped_and_pasted_PIL_format)
            tensor_0_current_img = copy.deepcopy(featuremap_current)
            self.save_featuremap_to_disk(tensor_0_current_img, tensor_0_shifted_cropped_and_pasted_save_dir)
            logging.info(f"Saved image 0 into general save folder")

            logging.info(f"Feeding image 1 into model")
            self.model_predictor.run_on_opencv_image(img_img_format_1_centered_cropped_and_pasted_PIL_format)
            tensor_1_current_img = copy.deepcopy(featuremap_current)
            self.save_featuremap_to_disk(tensor_1_current_img, tensor_1_centered_cropped_and_pasted_save_dir)
            logging.info(f"Saved image 1 into general save folder")
            #-------------------------------

            if imageAndPredictionProcessor._DEBUGGING:
                self.utils_helper.display_multi_image_collage_and_save(((img_img_format_0_shifted, f"Image org 0"),
                                                               (img_img_format_1_centered, f"Image org 1"),
                                                               (img_img_format_0_shifted_cropped_and_pasted_PIL_format
                                                                , f"Image cropped 0"),
                                                               (img_img_format_1_centered_cropped_and_pasted_PIL_format
                                                                , f"Image cropped 1"),
                                                               (tensor_0_current_img.numpy()[3, :, :]
                                                                , f"Img 0 tensor 3"),
                                                               (tensor_1_current_img.numpy()[3, :, :]
                                                                , f"Img 1 tensor 3"),),
                                                              [3, 2], img_collage_plot_save_path)

            logging.info(f"Image {org_img_file_name} successfully processed! {i+1}/{total_img_num}")


        logging.info("Dataset shifting complete!")


    def crop_and_paste_image(self, original_image, lower_length, upper_length):
        '''
        :param original_image: the original image in PIL format
        :param lower_and_upper_crop_coordinates_list: e.g [0.05, 0.30]
        :return: PIL format of the cropped and pasted image, ready to be saved
        '''
        img_np_format = np.asarray(original_image)
        org_img_height = img_np_format.shape[0]
        org_img_width = img_np_format.shape[1]

        # DO THE CROPPING AND PASTING
        img_np_format_all_zeros = np.zeros_like(img_np_format)
        img_np_format_cropped_and_pasted = img_np_format_all_zeros

        # Location to take pixels from in original image
        _crop_img_lower_length_index = math.floor(org_img_width * lower_length)
        _crop_img_upper_length_index = math.floor(org_img_width * upper_length)
        _cropped_piece_length = int(_crop_img_upper_length_index - _crop_img_lower_length_index)

        # This is the upper pat of the image
        _crop_img_lower_height_index = math.floor(org_img_height * self.default_vertical_cropping_percentage)
        # This is the lower part of the image
        _crop_img_upper_height_index = math.floor(org_img_height * (1-self.default_vertical_cropping_percentage))

        # Location to place pixels in clean image
        _paste_img_lower_length_index = math.floor((org_img_width - _cropped_piece_length) / 2)
        _paste_img_upper_length_index = _paste_img_lower_length_index + _cropped_piece_length
        _paste_img_lower_height_index = _crop_img_lower_height_index
        _paste_img_upper_height_index = _crop_img_upper_height_index

        img_np_format_cropped_and_pasted[_paste_img_lower_height_index:_paste_img_upper_height_index,
        _paste_img_lower_length_index:_paste_img_upper_length_index, :] = \
            img_np_format[_crop_img_lower_height_index:_crop_img_upper_height_index,
            _crop_img_lower_length_index:_crop_img_upper_length_index, :]

        img_img_format_shifted = Image.fromarray(img_np_format_cropped_and_pasted)
        return img_img_format_shifted


    def save_featuremap_to_disk(self, data, location_to_save):
        torch.save(data, location_to_save)


    def pad_img_with_zeros(self, img_without_padding):
        img_without_padding_height = img_without_padding.shape[0]
        img_without_padding_width = img_without_padding.shape[1]
        img_np_padded = np.pad(img_without_padding, [(0, img_without_padding_height*2),
                                                     (0, img_without_padding_width*2),
                                                     (0, 0)], mode='constant', constant_values=0)

        if imageAndPredictionProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((img_np_padded, f"Image padded"), ),
                                                          [1, 1])

        return img_np_padded