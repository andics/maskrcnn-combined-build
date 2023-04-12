import json, os
import copy
import numpy as np
import cv2
import logging
import math
import random

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter
from tqdm import tqdm

class miskAnnotationProcessor:
    _DEBUGGING = False

    def __init__(self, bins_lower_th_array, bins_upper_th_array,
                 bins_annotations_paths_array, bins_paths_array,
                 ann_summary_file_name, utils_helper,
                 normalization_factor, large_objects_present,
                 ann_subset_files_name_template,
                 ann_subset_files_summary_name_template):
        self.bins_lower_th_array = bins_lower_th_array
        self.bins_upper_th_array = bins_upper_th_array
        self.bins_annotations_paths_array = bins_annotations_paths_array
        self.bins_paths_array = bins_paths_array
        self.annotation_summary_file_name = ann_summary_file_name
        self.utils_helper = utils_helper
        self.normalization_factor = normalization_factor
        self.large_objects_present = large_objects_present
        self.generated_annotation_subset_files_name_template = ann_subset_files_name_template
        self.generated_annotation_subset_files_summary_name_template =\
            ann_subset_files_summary_name_template

        self.num_filtered_annotations_in_each_bin_array = np.array([])
        self.ann_subset_file_paths_array = []
        self.ann_subset_summary_file_paths_array = []


    def read_all_nums_objects(self):
        #This function reads the number of objects in ann bins and stores in array

        for bin_path in self.bins_paths_array:
            _ann_summary_file = os.path.join(bin_path, self.annotation_summary_file_name)
            assert(os.path.exists(_ann_summary_file))

            with open(_ann_summary_file) as _ann_json_file:
                json_data = json.load(_ann_json_file)

            if not self.large_objects_present:
                self.num_filtered_annotations_in_each_bin_array =\
                    np.append(self.num_filtered_annotations_in_each_bin_array,
                              (int(json_data["small_annotations"]),
                               int(json_data["medium_annotations"])
                               ))
            else:
                self.num_filtered_annotations_in_each_bin_array = \
                    np.append(self.num_filtered_annotations_in_each_bin_array,
                              (int(json_data["small_annotations"]),
                               int(json_data["medium_annotations"]),
                               int(json_data["large_annotations"])
                               ))


    def eval_normalization_factor(self):
        # Use the normalization factor to determine if we should select min, or a fixed number

        assert(self.normalization_factor > 0)

        if self.normalization_factor <= 1:
            self.smallest_num_annotations = self.num_filtered_annotations_in_each_bin_array.min()
            self.target_subsample_number = int(round(self.smallest_num_annotations * \
                                                 self.normalization_factor))
        else:
            self.target_subsample_number = int(self.normalization_factor)


    def generate_new_annotation_files_with_subsamples(self):
        for bin_lower_th, bin_upper_th,\
            bin_path, bin_annotation_file_path in zip(self.bins_lower_th_array, self.bins_upper_th_array,
                                                      self.bins_paths_array, self.bins_annotations_paths_array):
            annotation_subset_file_name = self.generated_annotation_subset_files_name_template % str(self.target_subsample_number)
            annotation_subset_summary_file_name = self.generated_annotation_subset_files_summary_name_template % str(self.target_subsample_number)

            ann_subset_file_path = os.path.join(bin_path, annotation_subset_file_name)
            ann_subset_summary_file_path = os.path.join(bin_path, annotation_subset_summary_file_name)

            self.ann_subset_file_paths_array.append(
                ann_subset_file_path)
            self.ann_subset_summary_file_paths_array.append(
                ann_subset_summary_file_path)

            self.generate_rand_ann_file_subset(bin_lower_th,
                                               bin_upper_th,
                                               bin_annotation_file_path,
                                               ann_subset_file_path,
                                               ann_subset_summary_file_path,
                                               self.target_subsample_number)

    def generate_rand_ann_file_subset(self, bin_lower_th,
                                      bin_upper_th,
                                      bin_annotation_file_path,
                                      ann_subset_file_path,
                                      ann_subset_summary_file_path,
                                      target_subsample_number):
        if os.path.exists(ann_subset_file_path):
            logging.info(f"Normalized annotation file"
                        f" for bin {ann_subset_file_path} already exists!. Skipping...")
            return

        with open(bin_annotation_file_path) as json_file:
            org_annotations_data = json.load(json_file)

        new_annotations_data = copy.deepcopy(org_annotations_data)
        coco = COCO(bin_annotation_file_path)
        logging.debug(f"Loaded JSON annotation data from: {bin_annotation_file_path}")

        original_anns_len = len(new_annotations_data["annotations"])
        total_num_preds_small = 0
        total_num_preds_medium = 0
        total_num_preds_large = 0

        ann_indices_small_objs = []
        ann_indices_medium_objs = []
        ann_indices_large_objs = []
        for i, annotation in tqdm(enumerate(new_annotations_data["annotations"]),
                                  total = len(new_annotations_data["annotations"]),
                                  desc ="Progress for summarizing the new annotations"):

            logging.debug(f"Working on annotation with ID {i}: "
                          f" | Area (default) {annotation['area']}"
                          f" | Image ID {annotation['image_id']}"
                          f" | Annotation ID {annotation['id']}"
                          f" | Label {coco.loadCats(annotation['category_id'])}|")

            annotation_coco_format = coco.loadAnns(annotation["id"])[0]
            # current_image_binary_mask: stores images as numpy array [height, width]
            current_image_binary_mask = coco.annToMask(annotation_coco_format)
            current_image_binary_mask_img = Image.fromarray(current_image_binary_mask)

            # Calculate the area of the current segmentation manually
            current_image_binary_mask_calculated_area = np.count_nonzero(current_image_binary_mask)
            assert current_image_binary_mask_calculated_area == current_image_binary_mask.sum()
            logging.debug(f"Calculated segmentation area: {current_image_binary_mask_calculated_area}")

            if current_image_binary_mask_calculated_area <= 32 ** 2:
                ann_indices_small_objs.append(i)
                total_num_preds_small += 1
            elif current_image_binary_mask_calculated_area <= 96 ** 2:
                ann_indices_medium_objs.append(i)
                total_num_preds_medium += 1
            else:
                ann_indices_large_objs.append(i)
                total_num_preds_large += 1

        #TODO: add a fixed random seed shuffle here, but with legit randomness
        #Then, make the generate random indices function
        #select as little overlapping batches of consecutive
        # indices, of size target sample number


        ann_indices_to_keep_small = self.generate_random_indices(target_subsample_number,
                                                            ann_indices_small_objs)
        ann_indices_to_keep_medium = self.generate_random_indices(target_subsample_number,
                                                            ann_indices_medium_objs)
        ann_indices_to_keep_large = self.generate_random_indices(target_subsample_number if self.large_objects_present else 0,
                                                            ann_indices_large_objs)

        ann_indices_to_keep_all = self.utils_helper.concatenate_lists(ann_indices_to_keep_small,
                                                                        ann_indices_to_keep_medium,
                                                                        ann_indices_to_keep_large)

        new_annotations_data["annotations"] = [new_annotations_data["annotations"][index] for index in
                                                    ann_indices_to_keep_all]

        logging.info(f"  -  Annotations left at the end: {len(new_annotations_data['annotations'])}/{original_anns_len}")
        logging.info(f"  -  Small annotations: {len(ann_indices_to_keep_small)}")
        logging.info(f"  -  Medium annotations: {len(ann_indices_to_keep_medium)}")
        logging.info(f"  -  Large annotations: {len(ann_indices_to_keep_large)}")

        #---WRITE-NEW-ANNOTATION-FILE-TO-DISK---
        self.utils_helper.write_data_to_json(ann_subset_file_path, new_annotations_data)
        logging.info(f"Successfully saved normalised annotations for bin {bin_lower_th}-{bin_upper_th} to disk...")

        #---WRITE-SUMMARY-TO-DISK---
        _dict_to_write = {"org_annotations_number": original_anns_len,
                          "after_filtering_annotations_number": len(new_annotations_data['annotations']),
                          "small_annotations": len(ann_indices_to_keep_small),
                          "medium_annotations": len(ann_indices_to_keep_medium),
                          "large_annotations": len(ann_indices_to_keep_large)}

        self.utils_helper.write_data_to_json(ann_subset_summary_file_path, _dict_to_write)
        logging.info(f"Successfully normalised bin {bin_lower_th}-{bin_upper_th}... Moving to next (if any)")


    def generate_random_indices(self, n, list_to_choose_n_elements_from):
        """Pick n points from list"""
        random_sample = random.sample(list_to_choose_n_elements_from, n)
        return random_sample
