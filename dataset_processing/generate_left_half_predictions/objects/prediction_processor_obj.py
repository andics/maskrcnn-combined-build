import json, os
import copy
import numpy as np
import cv2
import torch
import pycocotools.mask as mask_util
import math

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from itertools import groupby
from skimage import measure
from operator import itemgetter

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

from utils_gen import dataset_utils

class predictionProcessor:
    _DEBUGGING = False
    _WRITE_ALL_RLE_FORMAT = False
    _USE_COMPRESSED_FORMAT = False

    def __init__(self, config_file, utils_helper, logger):
        self.config_file = config_file
        self.utils_helper = utils_helper
        self.logger = logger

    def setup_new_predictions_folder_structure(self):

        #Using the same subdir as the one used by the logger to store the new predictions
        self.new_predictions_dir_path = os.path.join(self.config_file["FRAME"]["new_predictions_dir_path"],
                                                      self.config_file["LOGGING"]["logs_subdir"])
        _tmp = self.utils_helper.check_dir_and_make_if_na(self.new_predictions_dir_path)
        self.new_predictions_file_path = os.path.join(self.new_predictions_dir_path,
                                         self.utils_helper.extract_filename_and_ext(self.config_file["FRAME"]["org_predictions_file_path"])[0] + "." + \
                                         self.utils_helper.extract_filename_and_ext(
                                             self.config_file["FRAME"]["org_predictions_file_path"])[1])

        self.logger.log(f"Finished setting up new predictions folder structure! Create new predictions sub-dir: {_tmp}")
        model_cfg_path = self.config_file["MODEL"]["model_config"]
        cfg.merge_from_file(model_cfg_path)
        cfg.freeze()

    def read_predictions(self):
        self.org_predictions_data = torch.load(self.config_file["FRAME"]["org_predictions_file_path"])
        self.new_predictions_data = copy.deepcopy(self.org_predictions_data)

        self.logger.log(f"Loaded PTH prediction data from: {self.config_file['FRAME']['org_predictions_file_path']}")

    def setup_objects(self):
        dataset_catalogue = dataset_utils.get_dataset_catalog(cfg)
        try:
            self.org_annotation_file = os.path.join(dataset_catalogue.DATA_DIR,
                                                    dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["ann_file"])
            self.org_annotation_images_path = os.path.join(dataset_catalogue.DATA_DIR,
                                                           dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["img_dir"])

            #This part is hardcoded to acocunt for the fact that the dataset images could be located in the coco shared
            #but also in Danny's directory - as is the case for filtered images
            if os.path.exists(self.org_annotation_images_path):
                pass
            else:
                self.org_annotation_images_path = os.path.join("/home/projects/bagon/dannyh/data",
                                                               dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["img_dir"])

            self.org_annotation_images_path = os.path.join("/home/projects/bagon/dannyh/data",
                                                    dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["img_dir"])
            self.logger.log(f"Successfully loaded annotation file for the {cfg.DATASETS.TEST} dataset!")
        except Exception as e:
            self.logger.log(f"Attempted to load annotation file for the {cfg.DATASETS.TEST} dataset but failed:")
            self.logger.log(e.__str__())

        self.coco = COCO(self.org_annotation_file)
        self.coco_dataset = COCODataset(ann_file=self.org_annotation_file,
                                        root=self.org_annotation_images_path,
                                        cfg=cfg, remove_images_without_annotations=False)

    def filter_predictions_outside_border(self):
        #Workflow should be as follows:
        #Check workflow.txt file in this directory

        #Initiate Masker class for projecting masks to fit image size
        #This class is used to transform the Maskrcnn-native 28x28 mask to fit the image size and look like something
        masker = Masker(threshold=0.5, padding=1)

        for ind, img_predictions in enumerate(self.new_predictions_data):
            #Empty cropped predictions will be discarded
            inds_to_keep = []
            img_id = self.coco_dataset.id_to_img_map[ind]
            img_file_path = os.path.join(self.org_annotation_images_path,
                                         self.coco_dataset.coco.imgs[img_id]['file_name'])
            #Load original image in order to generate the border bounding box as well as to visualize
            org_img_np_format = np.array(Image.open(img_file_path))
            org_img_width = self.coco_dataset.coco.imgs[img_id]["width"]
            org_img_height = self.coco_dataset.coco.imgs[img_id]["height"]

            # the resized dimensions of the image as in the model
            rsz_img_width = img_predictions.size[0]
            rsz_img_height = img_predictions.size[1]

            #Resize the predictions to the original image dimensions for cropping inside the given FOV
            rsz_predictions = img_predictions.resize((org_img_width, org_img_height))
            rsz_predictions = rsz_predictions.convert("xywh")
            rsz_pred_masks = rsz_predictions.get_field('mask')
            #Masker is necessary only if masks haven't been already resized.
            if list(rsz_pred_masks.shape[-2:]) != [org_img_height, org_img_width]:
                rsz_pred_masks = masker(rsz_pred_masks.expand(1, -1, -1, -1, -1), rsz_predictions)
                rsz_pred_masks = rsz_pred_masks[0]
            rsz_pred_bboxes = rsz_predictions.bbox

            _border_bbox = self._calculate_border_bbox(org_img_np_format)
            #Cycle through all predictions and borderize them
            for i in range(len(rsz_predictions)):
                #Prediction mask before cropping
                pred_mask = rsz_pred_masks[i]
                pred_mask_binary_array_1d_for_disp = np.swapaxes(np.swapaxes(pred_mask.numpy(), 0, 2), 0, 1)
                #The masks require that the diemsnions of the mask tensor are (1, height, width)
                pred_mask_binary_array = np.repeat(pred_mask_binary_array_1d_for_disp, 3, axis=2)
                pred_mask_binary_form = Image.fromarray(pred_mask_binary_array * 255)
                if predictionProcessor._DEBUGGING:
                    self.utils_helper.display_multi_image_collage(
                        ((pred_mask_binary_form, f"Mask before cropping on image {self.coco_dataset.coco.imgs[img_id]['file_name']}"),),
                        [1, 1])
                pass
                #We crop the mask of the resized prediction
                #The binary mask from which we crop need be numpy (to ensure matrix and have dimensions
                #(m_num_rows, n_num_columns)
                pred_mask_binary_array_1d_for_disp[_border_bbox[0]:_border_bbox[2], _border_bbox[1]:_border_bbox[3]] = 0
                cropped_pred_mask_binary_array = np.repeat(pred_mask_binary_array_1d_for_disp, 3, axis=2)
                cropped_pred_mask_binary_form = Image.fromarray(cropped_pred_mask_binary_array * 255)
                if predictionProcessor._DEBUGGING:
                    self.utils_helper.display_multi_image_collage(
                        ((cropped_pred_mask_binary_form, f"Mask after cropping on image {self.coco_dataset.coco.imgs[img_id]['file_name']}"),),
                        [1, 1])
                self.logger.log(f"Working on prediction {i}, image {ind}")
                pred_mask_binary_array_1d_for_mask = np.swapaxes(np.swapaxes(pred_mask_binary_array_1d_for_disp, 2, 0),
                                                                 1, 2)
                if pred_mask_binary_array_1d_for_disp.any():  # in case the cropped prediction mask is not empty
                    fov_pred_mask = torch.from_numpy(pred_mask_binary_array_1d_for_mask)
                    # create a new bounding box for the cropped prediction mask
                    mask = mask_util.encode(np.array(torch.from_numpy(pred_mask_binary_array_1d_for_disp), order="F"))
                    fov_pred_bbox = mask_util.toBbox(mask).astype(int)
                    if predictionProcessor._DEBUGGING:
                        mask_v2 = self._binary_mask_to_compressed_rle(pred_mask_binary_array_1d_for_disp)
                        fov_pred_bbox_v2 = mask_util.toBbox(mask_v2).tolist()
                        self.logger.log(f"Transformed prediction into RLE format")
                    rsz_pred_masks[i] = fov_pred_mask
                    rsz_pred_bboxes[i] = torch.from_numpy(fov_pred_bbox.astype('float32'))
                    inds_to_keep.append(i)
            #Discard all annotations which were empty after cropping
            rsz_pred_masks = rsz_pred_masks[inds_to_keep, :, :, :]
            rsz_pred_bboxes = rsz_pred_bboxes[inds_to_keep, :]

            img_fov_crop_predictions = copy.deepcopy(rsz_predictions)
            img_fov_crop_predictions.bbox = rsz_pred_bboxes
            #For resizing the cropped masks back to 28x28
            #Img_fov_crop_predictions.extra_fields['mask'] = fov_pred_masks_rsz
            img_fov_crop_predictions.extra_fields['mask'] = rsz_pred_masks
            img_fov_crop_predictions.extra_fields['scores'] = img_fov_crop_predictions.extra_fields['scores'][inds_to_keep]
            img_fov_crop_predictions.extra_fields['labels'] = img_fov_crop_predictions.extra_fields['labels'][inds_to_keep]
            img_fov_crop_predictions = img_fov_crop_predictions.convert("xyxy")
            img_fov_crop_predictions = img_fov_crop_predictions.resize((rsz_img_width, rsz_img_height))
            self.new_predictions_data[ind] = img_fov_crop_predictions
            if ind == 1000:
                self.logger.log("Reached 1K lol")

    def write_new_predictions_to_disk(self):
        torch.save(self.new_predictions_data, self.new_predictions_file_path)

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

    def _rle_to_binary_mask(self, segmentation, org_prediction):
        '''
        :param segmentation: an RLE-format segmentation (dict: counts, size)
        :return: a numpy array representing a binary mask of the segmentation
        '''
        return self.coco.annToMask({'image_id': org_prediction['image_id'], 'segmentation': segmentation})

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
        img_width = image_array.shape[1]
        img_height = image_array.shape[0]
        org_img_bbox_repr = [0, 0, img_height, img_width]
        border_thickness_ratio = self.config_file["FRAME"]["thickness_ratio"]

        return [0, math.floor(img_width/2), img_height, img_width]

    def _borderize_this_prediction(self, prediction, index):
        '''
        :param prediction: Expects an prediction dictionary in the
        native format with which the JSON file stores each prediction (segmentation, ...)
        :return: None. This is because Dictionaries are mutable so we can modify the segmentation in-place
        '''

        #---DEBUGGING---
        if predictionProcessor._DEBUGGING:
            if prediction["image_id"] in [776, 5586, 12670, 20992, 72795]:
                print("Here")

            if prediction["id"] in [900100072795]:
                print("Here")
        #----------------

        prediction_coco_format = self.coco.loadAnns(prediction["id"])[0]
        #current_image_binary_mask: stores images as [height, width]
        current_image_binary_mask = self.coco.annToMask(prediction_coco_format)
        current_image_binary_mask_img = Image.fromarray(current_image_binary_mask)

        if predictionProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask_img, f"Image ID {prediction['id']}"), ),
                                                          [1, 1])
            pass
        _border_bbox = self._calculate_border_bbox(current_image_binary_mask)
        #Crop the bbox region
        current_image_binary_mask[_border_bbox[0]:_border_bbox[2], _border_bbox[1]:_border_bbox[3]] = 0

        if predictionProcessor._DEBUGGING:
            self.utils_helper.display_multi_image_collage(((current_image_binary_mask, f"After Cropping Image ID {prediction['id']}"), ),
                                                          [1, 1])
            pass

        #Now we start to check the state of the new segmentation
        #Case 1: the remaining segmentation is empty
        if current_image_binary_mask.max() == 0:
            self.logger.log(
                f"Remaining prediction {prediction['id']} on image {prediction['image_id']} was empty and deleted"
                f" (by checking max value)")
            return None

        #Case 2: the remaining segmentation is a point
        if len(np.where(current_image_binary_mask == 1)[0]) == 1:
            self.logger.log(f"Remaining prediction {prediction['id']} on image {prediction['image_id']} is a point")

        encoded_segmentation = self._binary_mask_to_compressed_rle(current_image_binary_mask)
        #encoded_segmentation: stores the segmentation in compressed RLE format (gh^fafv2...)
        #encoded_segmentation: used only to extract features of the segmentation: area, bbox
        new_segm_area = mask.area(encoded_segmentation).tolist()
        #Ensure area is in correct format
        if not (isinstance(new_segm_area, float) | isinstance(new_segm_area, int)): new_segm_area = new_segm_area.tolist()

        bbox_from_rle = mask.toBbox(encoded_segmentation).tolist()
        if prediction["iscrowd"] == 1 or predictionProcessor._WRITE_ALL_RLE_FORMAT:
            #For unknown to mankind reasons, the COCO creators decided to have two different formats for
            #iscrowd segmentations and normal segmentations, hence the need for this if
            if not predictionProcessor._USE_COMPRESSED_FORMAT:
                segmentation = self._binary_mask_to_uncompressed_rle(current_image_binary_mask)
            else:
                segmentation = self._binary_mask_to_compressed_rle(current_image_binary_mask)

            if predictionProcessor._DEBUGGING:
                self.utils_helper.display_multi_image_collage(
                    ((self._rle_to_binary_mask(segmentation, prediction), f"After Converting to RLE {prediction['id']}"),),
                    [1, 1])
                pass
        else:
            segmentation = self._binary_mask_to_polygon(current_image_binary_mask)
            #Check if non-empty after transformation
            if not segmentation:
                self.logger.log(
                    f"Remaining prediction {prediction['id']} on image {prediction['image_id']} was empty and deleted"
                    f" (by checking area)")
                return None

        if new_segm_area == 0:
            self.logger.log(
                f"Remaining prediction {prediction['id']} on image {prediction['image_id']} was empty and deleted"
                f" (by checking area)")
            return None

        self.new_predictions_data["predictions"][index]["segmentation"] = segmentation
        self.new_predictions_data["predictions"][index]["area"] = new_segm_area
        self.new_predictions_data["predictions"][index]["bbox"] = bbox_from_rle
        self.ann_indices_to_keep.append(index)

        return None
