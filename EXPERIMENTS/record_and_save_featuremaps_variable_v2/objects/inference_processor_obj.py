import json, os
import copy
import numpy as np
import cv2
import glob
import math
import torch

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter

from maskrcnn_benchmark.config import cfg
from EXPERIMENTS.record_and_save_featuremaps_variable_v2.objects.predictor_custom import COCO_predictor

filter_number_list = []
featuremap_current = []

def hook_module_backbone_layer3_last_bn(m, i, o):
    global filter_number_list
    global featuremap_current

    featuremap_current = []
    featuremap_current = copy.deepcopy(o[0, :, :, :]).cpu().squeeze()


class inferenceProcessor:
    _DEBUGGING = True
    _VALID_IMG_EXT = ['jpg']

    def __init__(self, org_dataset_folder, new_dataset_folder, utils_helper, model_config_path):
        self.org_dataset_folder = org_dataset_folder
        self.new_dataset_folder = new_dataset_folder
        self.utils_helper = utils_helper
        self.model_config_path = model_config_path


    def load_model(self):
        cfg.merge_from_file(self.model_config_path)
        cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
        self.model_predictor = COCO_predictor(cfg=cfg, custom_config_file=self.model_config_path, \
                                         weight_file_dir = cfg.OUTPUT_DIR, \
                                         use_conf_threshold=False, max_num_pred=5, min_image_size=60, masks_per_dim=3)
        self.model = self.model_predictor.model
        print("Model loaded! \n")
        print(self.model)


    def attach_hooks_to_model(self):
        self._module_backbone_layer3_last_bn = self.model._modules.get('backbone')._modules.get('body')._modules.get('layer3').\
            _modules.get('22')._modules.get('bn3')
        self.hook_module = self._module_backbone_layer3_last_bn.register_forward_hook(hook_module_backbone_layer3_last_bn)


    def read_all_images_in_org_dataset(self):
        self.all_org_img_paths = []
        if not inferenceProcessor._DEBUGGING:
            [self.all_org_img_paths.extend(glob.glob(os.path.join(self.org_dataset_folder, '*.' + e))) for e in inferenceProcessor._VALID_IMG_EXT]
        else:
            self.all_org_img_paths = ["/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/dataset_coco_2017_cropped_n_centered/" \
                                     "Variable_shifted_h_0.5_v_1.0_shifted_h_0.05_v_0.25/000000524280.jpg"]

        self.all_org_img_paths = []
        [self.all_org_img_paths.extend(glob.glob(os.path.join(self.org_dataset_folder, '*.' + e))) for e in
         inferenceProcessor._VALID_IMG_EXT]


    def process_all_images(self):
        global featuremap_current
        total_img_num = len(self.all_org_img_paths)
        for i, org_img_path in enumerate(self.all_org_img_paths):
            org_img_file_name, org_img_ext = self.utils_helper.extract_filename_and_ext(org_img_path)
            prediction_tensor_dir_path = os.path.join(self.new_dataset_folder, org_img_file_name)
            self.utils_helper.check_dir_and_make_if_na(prediction_tensor_dir_path)
            prediction_tensor_full_path = os.path.join(prediction_tensor_dir_path, "prediction_tensor.pth")

            img_img_format = Image.open(org_img_path).convert("RGB")
            img_np_format = np.asarray(img_img_format)
            org_img_height = img_np_format.shape[0]
            org_img_width = img_np_format.shape[1]
            print(f"Successfully loaded image {org_img_file_name}!")

            #Location to take pixels from in original image
            #FEED-IMAGE-TO-MODEL
            print(f"Feeding image into model")
            self.model_predictor.run_on_opencv_image(img_img_format)

            if inferenceProcessor._DEBUGGING:
                self.utils_helper.display_multi_image_collage(((img_img_format, f"Image fed in"),
                                                               (featuremap_current.numpy()[3, :, :], f"Tensor layer 3"),),
                                                              [1, 2])
            self.save_tensor_torch_format(featuremap_current, prediction_tensor_full_path)

            print(f"Image {org_img_file_name} successfully processed! {i+1}/{total_img_num}")

        print("Dataset shifting complete!")


    def save_tensor_torch_format(self, data, path_to_save):
        torch.save(data, path_to_save)


    def pad_img_with_zeros(self, img_without_padding):
        img_without_padding_height = img_without_padding.shape[0]
        img_without_padding_width = img_without_padding.shape[1]
        img_np_padded = np.pad(img_without_padding, [(0, img_without_padding_height*2),
                                                     (0, img_without_padding_width*2),
                                                     (0, 0)], mode='constant', constant_values=0)

        if inferenceProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((img_np_padded, f"Image padded"), ),
                                                          [1, 1])

        return img_np_padded