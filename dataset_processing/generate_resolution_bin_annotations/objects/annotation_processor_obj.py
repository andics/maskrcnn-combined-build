import json, os
import copy
import numpy as np
import cv2
import math

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter

class annotationProcessor:
    _DEBUGGING = False
    _VERBOSE = True
    _WRITE_ALL_RLE_FORMAT = True
    _USE_COMPRESSED_FORMAT = False

    def __init__(self, original_annotations_path,
                 filter_threshold_array,
                 experiment_name,
                 new_annotations_file_path,
                 middle_boundary,
                 utils_helper, logger):

        self.original_annotations_path = original_annotations_path
        self.filter_threshold_array = filter_threshold_array
        self.experiment_name = experiment_name
        self.new_annotations_file_path = new_annotations_file_path
        self.middle_boundary = middle_boundary
        self.utils_helper = utils_helper
        self.logger = logger

    def read_annotations(self):
        with open(self.original_annotations_path) as json_file:
            self.org_annotations_data = json.load(json_file)

        self.new_annotations_data = copy.deepcopy(self.org_annotations_data)
        self.coco = COCO(self.original_annotations_path)
        self.logger.log(f"Loaded JSON annotation data from: {self.original_annotations_path}")

    def filter_annotations_w_wrong_area_ratio(self):
        #Filters the annotations which have a high-res-area/total-area ratio not compatible with the
        #current bin
        #E.g. If bin (self.filter_threshold_array) is [0.0, 0.1] => all segmentations with more than 10%
        #of their area outside the high-resolution region will be filtered

        _anns_len = len(self.new_annotations_data["annotations"])
        self.ann_indices_to_keep = []
        for i, annotation in enumerate(self.new_annotations_data["annotations"]):
            #---DEBUGGING---
            if annotationProcessor._DEBUGGING:
                pass
            #---------------
            self._bin_check_this_annotation(annotation, i)
            self.logger.log(f"Processed {i+1}/{_anns_len} annotations")

        self.new_annotations_data["annotations"] = [self.new_annotations_data["annotations"][index] for index in
                                                    self.ann_indices_to_keep]


    def write_new_annotations_to_disk(self):
        self.utils_helper.write_data_to_json(self.new_annotations_file_path, self.new_annotations_data)


    def _calculate_high_res_bbox(self, image_array):
        '''
        :param image_array: a numpy array representing the image
        :return: a list of size 4 representing [high_res_bbox_top_corner_y, high_res_bbox_top_corner_x,
         high_res_bbox_bottom_corner_y, high_res_bbox_bottom_corner_x]
        '''
        #The bounding box of each border is inclusive of the pixels at it.
        #This means that areas stepping on the border itself will be counted as "inside"
        img_width = image_array.shape[1]
        img_height = image_array.shape[0]
        org_img_bbox_repr = [0, 0, img_height, img_width]

        #Calculate the image center, considering that Python indexing has an origin [0, 0]
        [center_y, center_x] = [math.floor(img_height/2) - 1, math.floor(img_width/2) - 1]
        margin_to_combine_with_center = int(self.middle_boundary/2)
        assert margin_to_combine_with_center == self.middle_boundary/2

        #Logic behind this statement is that if a 100x100 region is desired, one must add 49 and 50
        #along each diagonal dicection, starting from the center
        return [center_y - (margin_to_combine_with_center - 1), center_x - (margin_to_combine_with_center - 1),
                center_y + (margin_to_combine_with_center) + 1, center_x + (margin_to_combine_with_center) + 1]


    def _bin_check_this_annotation(self, annotation, index):
        '''
        :param annotation: Expects an annotation dictionary in the
        native format with which the JSON file stores each annotation (segmentation, ...)
        :return: None. This is because Dictionaries are mutable so we can modify the segmentation in-place
        '''

        #---DEBUGGING---
        if annotationProcessor._VERBOSE:
            self.logger.log(f"Working on annotation with ID {index}: "
                            f" | Area {annotation['area']}"
                            f" | Image ID {annotation['image_id']}"
                            f" | Annotation ID {annotation['id']}"
                            f" | Label {self.coco.loadCats(annotation['category_id'])}|")

        annotation_coco_format = self.coco.loadAnns(annotation["id"])[0]
        #current_image_binary_mask: stores images as numpy array [height, width]
        current_image_binary_mask = self.coco.annToMask(annotation_coco_format)
        current_image_binary_mask_img = Image.fromarray(current_image_binary_mask)

        #Calculate the area of the current segmentation manually
        current_image_binary_mask_calculated_area = np.count_nonzero(current_image_binary_mask)
        assert current_image_binary_mask_calculated_area == current_image_binary_mask.sum()

        self.logger.log(f"Calculated segmentation area: {current_image_binary_mask_calculated_area}")

        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask_img, f"Image ID {annotation['image_id']}"), ),
                                                          [1, 1])
            pass
        _high_res_border_bbox = self._calculate_high_res_bbox(current_image_binary_mask)
        #Crop the bbox region

        current_image_bunary_mask_inside_hr_bin = np.zeros_like(current_image_binary_mask)
        current_image_bunary_mask_inside_hr_bin[_high_res_border_bbox[0]:_high_res_border_bbox[2],
          _high_res_border_bbox[1]:_high_res_border_bbox[3]] = current_image_binary_mask[
                                                          _high_res_border_bbox[0]:_high_res_border_bbox[2],
                                                          _high_res_border_bbox[1]:_high_res_border_bbox[3]]

        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_bunary_mask_inside_hr_bin, f"Inside high-res"
                                                                                                     f" Image ID {annotation['image_id']}"), ),
                                                          [1, 1])
            pass

        #Calculate the area of the current segmentation inside high-res manually
        current_image_binary_mask_calculated_area_inside_hr = np.count_nonzero(current_image_bunary_mask_inside_hr_bin)
        assert current_image_binary_mask_calculated_area_inside_hr == current_image_bunary_mask_inside_hr_bin.sum()

        #Now we proceed to the filtering
        #Calculate hr/total ratio
        high_res_area_fract = current_image_binary_mask_calculated_area_inside_hr/current_image_binary_mask_calculated_area
        if annotationProcessor._VERBOSE:
            self.logger.log(f"Calculated segmentation area inside high-resolution region:"
                            f" {current_image_binary_mask_calculated_area_inside_hr}"
                            f"\n | Ratio: {high_res_area_fract} | ")


        if (high_res_area_fract >= self.filter_threshold_array[0]) and (high_res_area_fract <= self.filter_threshold_array[1]):
            self.ann_indices_to_keep.append(index)
            self.logger.log(f"Annotation {annotation['id']} on image {annotation['image_id']} was kept")
        else:
            self.logger.log(f"Annotation {annotation['id']} on image {annotation['image_id']} was deleted")

        return None
