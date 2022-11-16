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
                 horizontal_shift, vertical_shift,):
        self.org_dataset_folder = org_dataset_folder
        self.new_dataset_folder = new_dataset_folder
        self.utils_helper = utils_helper
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift

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

            img_np_format_padded = self.pad_img_with_zeros(img_np_format)
            #Here we shift the image
            #Those are temporary variables used just to make shorter abreviations for the complex slicing
            #expression that needs to be performed
            _v_0 = math.floor(self.vertical_shift*org_img_height)
            _h_0 = math.floor(self.horizontal_shift*org_img_width)
            _v_1 = _v_0 + org_img_height
            _h_1 = _h_0 + org_img_width
            img_np_format_padded[_v_0:_v_1, _h_0:_h_1, :] = img_np_format_padded[0:org_img_height,0:org_img_width,:]
            if imageProcessor._DEBUGGING:
                self.utils_helper.display_multi_image_collage(((img_np_format_padded, f"Image padded & shifted"),),
                                                              [1, 1])

            #img_img_format_padded = Image.fromarray(img_np_format_padded)
            img_img_format_shifted = Image.fromarray(img_np_format_padded[org_img_height:org_img_height*2,org_img_width:org_img_width*2,:])
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