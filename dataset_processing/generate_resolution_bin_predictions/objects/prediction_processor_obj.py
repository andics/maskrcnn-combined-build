import json, os
import copy
import numpy as np
import cv2
import math
import torch
import pycocotools.mask as mask_util

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
    _VERBOSE = False

    def __init__(self, org_predictions_path,
                 new_predictions_path,
                 area_threshold_array,
                 middle_boundry,
                 dataset_name,
                 model_cfg_path,
                 logger,
                 utils_helper):
        self.org_predictions_path = org_predictions_path
        self.new_predictions_path = new_predictions_path
        self.area_threshold_array = area_threshold_array
        self.middle_boundry = middle_boundry
        self.dataset_name = dataset_name
        self.model_cfg_path = model_cfg_path
        self.logger = logger
        self.utils_helper = utils_helper


    def setup_objects_and_misk_variables(self):
        #The following fuction is used to setup variables needed for extracting the height/width
        # of each prediction later on. In fact, those variables are the image directory as well as the annotation file

        cfg.merge_from_file(self.model_cfg_path)
        cfg.merge_from_list(["DATASETS.TEST", self.dataset_name])
        cfg.freeze()

        dataset_catalogue = dataset_utils.get_dataset_catalog(cfg)

        try:
            self.org_annotation_file = os.path.join(dataset_catalogue.DATA_DIR,
                                                    dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["ann_file"])
            self.org_annotation_images_path = os.path.join(dataset_catalogue.DATA_DIR,
                                                           dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["img_dir"])
            if not os.path.exists(self.org_annotation_file):
                self.org_annotation_file = dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["ann_file"]
                self.org_annotation_images_path = dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["img_dir"]

            assert os.path.exists(self.org_annotation_file)

            self.logger.log(f"Successfully loaded annotation file for the {cfg.DATASETS.TEST} dataset!")
        except Exception as e:
            self.logger.log(f"Attempted to load annotation file for the {cfg.DATASETS.TEST} dataset but failed:")
            self.logger.log(e.__str__())

        self.coco = COCO(self.org_annotation_file)
        self.coco_dataset = COCODataset(ann_file=self.org_annotation_file,
                                        root=self.org_annotation_images_path,
                                        cfg=cfg, remove_images_without_annotations=False)


    def read_predictions(self):
        self.org_predictions_data = torch.load(self.org_predictions_path)
        self.new_predictions_data = copy.deepcopy(self.org_predictions_data)

        self.logger.log(f"Loaded PTH prediction data from: {self.org_predictions_path}")


    def filter_predictions_w_wrong_area_ratio(self):
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

            _high_res_border_bbox = self._calculate_high_res_bbox(org_img_np_format)
            # Crop the bbox region
            #Cycle through all predictions for a given image and borderize them
            for i in range(len(rsz_predictions)):
                #Prediction mask before cropping
                pred_mask = rsz_pred_masks[i]
                pred_mask_binary_array_1d_for_disp = np.swapaxes(np.swapaxes(pred_mask.numpy(), 0, 2), 0, 1)
                #The masks require that the diemsnions of the mask tensor are (1, height, width)
                pred_mask_binary_array = np.repeat(pred_mask_binary_array_1d_for_disp, 3, axis=2)
                pred_mask_binary_form = Image.fromarray(pred_mask_binary_array * 255)
                if predictionProcessor._DEBUGGING:
                    self.utils_helper.display_multi_image_collage(
                        ((org_img_np_format, f"Original image {self.coco_dataset.coco.imgs[img_id]['file_name']}"),
                         (pred_mask_binary_form, f"Prediction mask before filtering"),),
                        [1, 2])
                pass
                #We crop the mask of the resized prediction
                #The binary mask from which we crop need be numpy (to ensure matrix and have dimensions
                #(m_num_rows, n_num_columns)
                pred_mask_binary_array_1d_for_disp[_high_res_border_bbox[0]:_high_res_border_bbox[2], _high_res_border_bbox[1]:_high_res_border_bbox[3]] = 0
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
        #The bounding box of each border is exclusive of the pixels at it.
        #This means that the remaining predictions after filtering will not be "stepping" on the border itself
        img_width = image_array.shape[1]
        img_height = image_array.shape[0]
        org_img_bbox_repr = [0, 0, img_height, img_width]
        border_thickness_ratio = self.config_file["FRAME"]["thickness_ratio"]

        return [round(number) for number in [org_img_bbox_repr[2] * border_thickness_ratio, org_img_bbox_repr[3] * border_thickness_ratio,
          org_img_bbox_repr[2] * (1 - border_thickness_ratio), org_img_bbox_repr[3] * (1 - border_thickness_ratio)]]


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
