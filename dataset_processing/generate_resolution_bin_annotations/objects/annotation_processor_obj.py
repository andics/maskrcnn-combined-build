import json, os
import copy
import numpy as np
import cv2

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter

class annotationProcessor:
    _DEBUGGING = False
    _WRITE_ALL_RLE_FORMAT = True
    _USE_COMPRESSED_FORMAT = False

    def __init__(self, config_file, utils_helper, logger):
        self.config_file = config_file
        self.utils_helper = utils_helper
        self.logger = logger

    def setup_new_annotations_folder_structure(self):

        #Using the same subdir as the one used by the logger to store the new annotations
        self.new_annotations_dir_path = os.path.join(self.config_file["FRAME"]["new_annotation_dir_path"],
                                                      self.config_file["LOGGING"]["logs_subdir"])
        _tmp = self.utils_helper.check_dir_and_make_if_na(self.new_annotations_dir_path)
        self.new_annotations_file_path = os.path.join(self.new_annotations_dir_path,
                                         "instances_val2017" + "." + \
                                         self.utils_helper.extract_filename_and_ext(
                                             self.config_file["FRAME"]["org_annotation_file_path"])[1])

        self.logger.log(f"Finished setting up new annotations folder structure! Create new annotations sub-dir: {_tmp}")

    def read_annotations(self):
        with open(self.config_file["FRAME"]["org_annotation_file_path"]) as json_file:
            self.org_annotations_data = json.load(json_file)

        self.new_annotations_data = copy.deepcopy(self.org_annotations_data)
        self.logger.log(f"Loaded JSON annotation data from: {self.config_file['FRAME']['org_annotation_file_path']}")
        self.coco = COCO(self.config_file["FRAME"]["org_annotation_file_path"])

    def filter_annotations_outside_border(self):
        #FYI: One of the problematic annotations has an index: 36394

        _anns_len = len(self.new_annotations_data["annotations"])
        self.ann_indices_to_keep = []
        for i, annotation in enumerate(self.new_annotations_data["annotations"]):
            #---DEBUGGING---
            if annotationProcessor._DEBUGGING:
                if annotation["id"] != 900100072795:
                    continue
                else:
                    print("Here23")
            #---------------
            self._borderize_this_annotation(annotation, i)
            self.logger.log(f"Processed {i+1}/{_anns_len} annotations")

        self.new_annotations_data["annotations"] = [self.new_annotations_data["annotations"][index] for index in
                                                    self.ann_indices_to_keep]

    def write_new_annotations_to_disk(self):
        self.utils_helper.write_data_to_json(self.new_annotations_file_path, self.new_annotations_data)

    def _binary_mask_to_uncompressed_rle(self, binary_mask):
        #Function to transform a binary mask to uncompressed RLE, only used for is_crowd True
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle

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

    def _calculate_border_bbox(self, image_array):
        '''
        :param image_array: a numpy array representing the image
        :return: a list of size 4 representing [bbox_top_corner_y, bbox_top_corner_x, bbox_bottom_corner_y, bbox_botton_corner_x]
        '''
        #The bounding box of each border is exclusive of the pixels at it.
        #This means that the remaining annotations after filtering will not be "stepping" on the border itself
        img_width = image_array.shape[1]
        img_height = image_array.shape[0]
        org_img_bbox_repr = [0, 0, img_height, img_width]
        border_thickness_ratio = self.config_file["FRAME"]["thickness_ratio"]

        return [round(number) for number in [org_img_bbox_repr[2] * border_thickness_ratio, org_img_bbox_repr[3] * border_thickness_ratio,
          org_img_bbox_repr[2] * (1 - border_thickness_ratio), org_img_bbox_repr[3] * (1 - border_thickness_ratio)]]

    def _borderize_this_annotation(self, annotation, index):
        '''
        :param annotation: Expects an annotation dictionary in the
        native format with which the JSON file stores each annotation (segmentation, ...)
        :return: None. This is because Dictionaries are mutable so we can modify the segmentation in-place
        '''

        #---DEBUGGING---
        if annotationProcessor._DEBUGGING:
            if annotation["image_id"] in [776, 5586, 12670, 20992, 72795]:
                print("Here")

            if annotation["id"] in [900100072795]:
                print("Here")
        #----------------

        annotation_coco_format = self.coco.loadAnns(annotation["id"])[0]
        #current_image_binary_mask: stores images as [height, width]
        current_image_binary_mask = self.coco.annToMask(annotation_coco_format)
        current_image_binary_mask_img = Image.fromarray(current_image_binary_mask)

        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask_img, f"Image ID {annotation['id']}"), ),
                                                          [1, 1])
            pass
        _border_bbox = self._calculate_border_bbox(current_image_binary_mask)
        #Crop the bbox region
        current_image_binary_mask[_border_bbox[0]:_border_bbox[2], _border_bbox[1]:_border_bbox[3]] = 0

        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask, f"After Cropping Image ID {annotation['id']}"), ),
                                                          [1, 1])
            pass

        #Now we start to check the state of the new segmentation
        #Case 1: the remaining segmentation is empty
        if current_image_binary_mask.max() == 0:
            self.logger.log(
                f"Remaining annotation {annotation['id']} on image {annotation['image_id']} was empty and deleted"
                f" (by checking max value)")
            return None

        #Case 2: the remaining segmentation is a point
        if len(np.where(current_image_binary_mask == 1)[0]) == 1:
            self.logger.log(f"Remaining annotation {annotation['id']} on image {annotation['image_id']} is a point")

        encoded_segmentation = self._binary_mask_to_compressed_rle(current_image_binary_mask)
        #encoded_segmentation: stores the segmentation in compressed RLE format (gh^fafv2...)
        #encoded_segmentation: used only to extract features of the segmentation: area, bbox
        new_segm_area = mask.area(encoded_segmentation).tolist()
        #Ensure area is in correct format
        if not (isinstance(new_segm_area, float) | isinstance(new_segm_area, int)): new_segm_area = new_segm_area.tolist()

        bbox_from_rle = mask.toBbox(encoded_segmentation).tolist()
        if annotation["iscrowd"] == 1 or annotationProcessor._WRITE_ALL_RLE_FORMAT:
            #For unknown to mankind reasons, the COCO creators decided to have two different formats for
            #iscrowd segmentations and normal segmentations, hence the need for this if
            if not annotationProcessor._USE_COMPRESSED_FORMAT:
                segmentation = self._binary_mask_to_uncompressed_rle(current_image_binary_mask)
            else:
                segmentation = self._binary_mask_to_compressed_rle(current_image_binary_mask)

            if annotationProcessor._DEBUGGING:
                self.utils_helper.display_multi_image_collage(
                    ((self._rle_to_binary_mask(segmentation, annotation), f"After Converting to RLE {annotation['id']}"),),
                    [1, 1])
                pass
        else:
            segmentation = self._binary_mask_to_polygon(current_image_binary_mask)
            #Check if non-empty after transformation
            if not segmentation:
                self.logger.log(
                    f"Remaining annotation {annotation['id']} on image {annotation['image_id']} was empty and deleted"
                    f" (by checking area)")
                return None

        if new_segm_area == 0:
            self.logger.log(
                f"Remaining annotation {annotation['id']} on image {annotation['image_id']} was empty and deleted"
                f" (by checking area)")
            return None

        self.new_annotations_data["annotations"][index]["segmentation"] = segmentation
        self.new_annotations_data["annotations"][index]["area"] = new_segm_area
        self.new_annotations_data["annotations"][index]["bbox"] = bbox_from_rle
        self.ann_indices_to_keep.append(index)

        return None
