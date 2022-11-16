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

class annotationProcessor:
    _DEBUGGING = False
    _WRITE_ALL_RLE_FORMAT = False
    _USE_COMPRESSED_FORMAT = False

    def __init__(self, org_annotations_location, new_annotations_location, utils_helper,
                 horizontal_shift, vertical_shift):
        self.org_annotations_location = org_annotations_location
        self.new_annotations_location = new_annotations_location
        self.utils_helper = utils_helper
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift

    def read_annotations(self):
        with open(self.org_annotations_location) as json_file:
            self.org_annotations_data = json.load(json_file)

        self.new_annotations_data = copy.deepcopy(self.org_annotations_data)
        print(f"Loaded JSON annotation data from: {self.org_annotations_location}")
        self.coco = COCO(self.org_annotations_location)


    def shift_annotations(self):
        #FYI: One of the problematic annotations has an index: 36394

        _anns_len = len(self.new_annotations_data["annotations"])
        self.ann_indices_to_keep = []
        for i, annotation in enumerate(self.new_annotations_data["annotations"]):
            self._shift_this_annotation(annotation, i)
            print(f"Processed {i+1}/{_anns_len} annotations")

        self.new_annotations_data["annotations"] = [self.new_annotations_data["annotations"][index] for index in
                                                    self.ann_indices_to_keep]

    def _shift_this_annotation(self, annotation, index):
        '''
        :param annotation: Expects an annotation dictionary in the
        native format with which the JSON file stores each annotation (segmentation, ...)
        :return: None. This is because Dictionaries are mutable so we can modify the segmentation in-place
        '''

        #---DEBUGGING---
        '''
        if annotationProcessor._DEBUGGING:
            if annotation["image_id"] in [776, 5586, 12670, 20992, 72795]:
                print("Here")

            if annotation["id"] in [900100072795]:
                print("Here")
        '''
        #----------------

        annotation_coco_format = self.coco.loadAnns(annotation["id"])[0]
        #current_image_binary_mask: stores images as [height, width]
        current_image_binary_mask = self.coco.annToMask(annotation_coco_format)
        current_image_binary_mask_img = Image.fromarray(current_image_binary_mask)

        org_img_height = current_image_binary_mask.shape[0]
        org_img_width = current_image_binary_mask.shape[1]

        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask_img, f"Image ID {annotation['image_id']}"), ),
                                                          [1, 1])
            pass
        #---THE-SHIFTING-PARADIGM---
        current_image_binary_mask_padded = self.pad_img_with_zeros(current_image_binary_mask)
        # Here we shift the image
        # Those are temporary variables used just to make shorter abreviations for the complex slicing
        # expression that needs to be performed
        _v_0 = math.floor(self.vertical_shift * org_img_height)
        _h_0 = math.floor(self.horizontal_shift * org_img_width)
        _v_1 = _v_0 + org_img_height
        _h_1 = _h_0 + org_img_width
        current_image_binary_mask_padded[_v_0:_v_1, _h_0:_h_1] = current_image_binary_mask_padded[0:org_img_height, 0:org_img_width]
        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask_padded, f"Image padded & shifted"),),
                                                          [1, 1])

        current_image_binary_mask_shifted = current_image_binary_mask_padded[org_img_height:org_img_height * 2, org_img_width:org_img_width * 2]
        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask_shifted, f"After Shifting Image ID {annotation['image_id']}"), ),
                                                          [1, 1])
            pass
        #---------------------------

        #Now we start to check the state of the new segmentation
        #Case 1: the remaining segmentation is empty
        if current_image_binary_mask.max() == 0:
            print(
                f"Remaining annotation {annotation['id']} on image {annotation['image_id']} was empty and deleted"
                f" (by checking max value)")
            return None

        #Case 2: the remaining segmentation is a point
        if len(np.where(current_image_binary_mask == 1)[0]) == 1:
            print(f"Remaining annotation {annotation['id']} on image {annotation['image_id']} is a point")

        encoded_segmentation = self._binary_mask_to_compressed_rle(current_image_binary_mask_shifted)
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
                segmentation = self._binary_mask_to_uncompressed_rle(current_image_binary_mask_shifted)
            else:
                segmentation = self._binary_mask_to_compressed_rle(current_image_binary_mask_shifted)

            if annotationProcessor._DEBUGGING:
                self.utils_helper.display_multi_image_collage(
                    ((self._rle_to_binary_mask(segmentation, annotation), f"After Converting to RLE {annotation['id']}"),),
                    [1, 1])
                pass
        else:
            segmentation = self._binary_mask_to_polygon(current_image_binary_mask_shifted)
            #Check if non-empty after transformation
            if not segmentation:
                print(
                    f"Remaining annotation {annotation['id']} on image {annotation['image_id']} was empty and deleted"
                    f" (by checking area)")
                return None

        if new_segm_area == 0:
            print(
                f"Remaining annotation {annotation['id']} on image {annotation['image_id']} was empty and deleted"
                f" (by checking area)")
            return None

        self.new_annotations_data["annotations"][index]["segmentation"] = segmentation
        self.new_annotations_data["annotations"][index]["area"] = new_segm_area
        self.new_annotations_data["annotations"][index]["bbox"] = bbox_from_rle
        self.ann_indices_to_keep.append(index)

        return None

    def write_new_annotations_to_disk(self):
        self.utils_helper.write_data_to_json(self.new_annotations_location, self.new_annotations_data)

    def pad_img_with_zeros(self, img_without_padding):
        '''
        :param img_without_padding: A binary mask representing the particular annotation
        :return: The binary mask of the annotation but shifted
        '''
        img_without_padding_height = img_without_padding.shape[0]
        img_without_padding_width = img_without_padding.shape[1]
        img_np_padded = np.pad(img_without_padding, [(0, img_without_padding_height*2),
                                                     (0, img_without_padding_width*2)], mode='constant', constant_values=0)

        if annotationProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((img_np_padded, f"Annotation binary mask padded"), ),
                                                          [1, 1])

        return img_np_padded

    def _binary_mask_to_compressed_rle(self, binary_mask):
        #Function to transform a binary mask to compressed RLE, only used for is_crowd True and compress=True
        return mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    def _binary_mask_to_uncompressed_rle(self, binary_mask):
        #Function to transform a binary mask to uncompressed RLE, only used for is_crowd True
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle

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

        return polygons