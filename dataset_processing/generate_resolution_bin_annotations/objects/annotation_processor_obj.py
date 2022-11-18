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
    _DEBUGGING = True
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


    def _binary_mask_to_compressed_rle(self, binary_mask):
        #Function to transform a binary mask to compressed RLE, only used for is_crowd True and compress=True
        return mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))


    def _rle_to_binary_mask(self, segmentation, org_annotation):
        '''
        :param segmentation: an RLE-format segmentation (dict: counts, size)
        :return: a numpy array representing a binary mask of the segmentation
        '''
        return self.coco.annToMask({'image_id': org_annotation['image_id'], 'segmentation': segmentation})


    def _binary_mask_to_polygon(self, binary_mask, tolerance=1):
        """Converts a binary mask to COCO polygon representation
        Args:
            binary_mask: a 2D binary numpy array where '1's represent the object
            tolerance: Maximum distance from original points of polygon to approximated
                polygonal chain. If tolerance is 0, the original coordinate array is returned.
        """

        def close_contour(contour):
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))
            return contour
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:
            contour = close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)


    def _binary_mask_to_uncompressed_rle(self, binary_mask):
        #Function to transform a binary mask to uncompressed RLE, only used for is_crowd True
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle


    def _binary_mask_to_polygon_v2(self, bin_mask):
        mask_new, contours = cv2.findContours((bin_mask).astype(np.uint8), cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []

        for contour in contours:
            contour = contour.flatten().tolist()
            # segmentation.append(contour)
            if len(contour) > 4:
                segmentation.append(contour)
        if len(segmentation) == 0:
            return segmentation

        return segmentation


    def _binary_mask_to_polygon_v3(self, bin_mask):

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = mask.frPyObjects(segmentation, bin_mask.shape[0], bin_mask.shape[1])
        RLE = mask.merge(RLEs)
        # RLE = cocomask.encode(np.asfortranarray(mask))
        area = mask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(bin_mask)

        return segmentation, [x, y, w, h], area


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
        margin_to_combine_with_center = self.middle_boundary/2
        assert isinstance(margin_to_combine_with_center, int)

        #Logic behind this statement is that if a 100x100 region is desired, one must add 49 and 50
        #along each diagonal dicection, starting from the center
        return [center_y - (margin_to_combine_with_center - 1), center_x - (margin_to_combine_with_center - 1),
                center_y + (margin_to_combine_with_center), center_x - (margin_to_combine_with_center)]


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
                            f" | Image ID {annotation['id']}"
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
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask_img, f"Image ID {annotation['id']}"), ),
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
            self.utils_helper.display_multi_image_collage(((current_image_bunary_mask_inside_hr_bin, f"Inside high-res Image ID {annotation['id']}"), ),
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


        if high_res_area_fract >= self.filter_threshold_array[0] & high_res_area_fract <= self.filter_threshold_array[1]:
            self.logger.log(f"Annotation {annotation['id']} on image {annotation['image_id']} was deleted")
        else:
            self.ann_indices_to_keep.append(index)
            self.logger.log(f"Annotation {annotation['id']} on image {annotation['image_id']} was kept")

        return None
