import json, os
import copy
import numpy as np
import cv2
import glob
import math

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter

class imageProcessor:
    _DEBUGGING = False
    _VALID_IMG_EXT = ['jpg', 'png']

    def __init__(self, org_dataset_folder, new_dataset_folder, utils_helper,
                 lower_length, upper_length):
        self.org_dataset_folder = org_dataset_folder
        self.new_dataset_folder = new_dataset_folder
        self.utils_helper = utils_helper
        self.lower_length = lower_length
        self.upper_length = upper_length

    def read_all_images_in_org_dataset(self):
        self.all_org_img_paths = []
        [self.all_org_img_paths.extend(glob.glob(os.path.join(self.org_dataset_folder, '*.' + e))) for e in imageProcessor._VALID_IMG_EXT]

    def process_all_images(self):
        total_img_num = len(self.all_org_img_paths)
        for i, org_img_path in enumerate(self.all_org_img_paths):
            org_img_file_name, org_img_ext = self.utils_helper.extract_filename_and_ext(org_img_path)
            new_img_path = os.path.join(self.new_dataset_folder,
                                        org_img_file_name + "." + org_img_ext)

            img_img_format = Image.open(org_img_path).convert("RGB")
            img_np_format = np.asarray(img_img_format)
            org_img_height = img_np_format.shape[0]
            org_img_width = img_np_format.shape[1]
            print(f"Successfully loaded image {org_img_file_name}!")

            #DO THE CROPPING AND PASTING
            img_np_format_all_zeros = np.zeros_like(img_np_format)
            img_np_format_cropped_and_pasted = img_np_format_all_zeros

            #Location to take pixels from in original image
            _crop_img_lower_length_index = math.floor(org_img_width*self.lower_length)
            _crop_img_upper_length_index = math.floor(org_img_width*self.upper_length)
            _cropped_piece_length = int(_crop_img_upper_length_index - _crop_img_lower_length_index)
            #Location to place pixels in clean image
            _paste_img_lower_length_index = math.floor((org_img_width - _cropped_piece_length)/2)
            _paste_img_upper_length_index = _paste_img_lower_length_index + _cropped_piece_length

            img_np_format_cropped_and_pasted[:, _paste_img_lower_length_index:_paste_img_upper_length_index, :] =\
                img_np_format[:, _crop_img_lower_length_index:_crop_img_upper_length_index, :]


            if imageProcessor._DEBUGGING:
                self.utils_helper.display_multi_image_collage(((img_np_format_cropped_and_pasted, f"Image cleaned"),
                                                               (img_np_format, f"Image original"),),
                                                              [1, 2])

            img_img_format_shifted = Image.fromarray(img_np_format_cropped_and_pasted)
            img_img_format_shifted.save(new_img_path)
            print(f"Image {org_img_file_name} successfully processed! {i+1}/{total_img_num}")

        print("Dataset shifting complete!")

    def pad_img_with_zeros(self, img_without_padding):
        img_without_padding_height = img_without_padding.shape[0]
        img_without_padding_width = img_without_padding.shape[1]
        img_np_padded = np.pad(img_without_padding, [(0, img_without_padding_height*2),
                                                     (0, img_without_padding_width*2),
                                                     (0, 0)], mode='constant', constant_values=0)

        if imageProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((img_np_padded, f"Image padded"), ),
                                                          [1, 1])

        return img_np_padded