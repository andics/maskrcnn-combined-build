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
    _VERBOSE = True

    def __init__(self, org_predictions_path,
                 annotation_file_path,
                 dataset_images_path,
                 config_path,
                 logger):
        self.org_predictions_path = org_predictions_path
        self.org_annotation_file = annotation_file_path
        self.org_annotation_images_path = dataset_images_path
        self.config_path = config_path
        self.logger = logger


    def setup_coco_image_reader(self):
        #The following fuction is used to setup variables needed for extracting the height/width
        # of each prediction later on. In fact, those variables are the image directory as well as the annotation file
        cfg.merge_from_file(self.config_path)
        cfg.freeze()

        dataset_catalogue = dataset_utils.get_dataset_catalog(cfg)

        try:
            assert os.path.exists(self.org_annotation_file)

            self.logger.log(f"Successfully loaded parent annotation file:\n{self.org_annotation_file}")
        except Exception as e:
            self.logger.log(f"Attempted to load parent annotation file:\n{self.org_annotation_file}\nbut failed:\n")
            self.logger.log(e.__str__())

        self.coco = COCO(self.org_annotation_file)
        self.coco_dataset = COCODataset(ann_file=self.org_annotation_file,
                                        root=self.org_annotation_images_path,
                                        cfg=cfg, remove_images_without_annotations=False)


    def read_predictions(self):
        self.org_predictions_data = torch.load(self.org_predictions_path)
        self.new_predictions_data = copy.deepcopy(self.org_predictions_data)

        self.logger.log(f"Loaded PTH prediction data from: {self.org_predictions_path}")


    def count_predictions(self):
        total_num_preds_grand = 0
        total_num_preds_small = 0
        total_num_preds_medium = 0
        total_num_preds_large = 0

        masker = Masker(threshold=0.5, padding=1)
        #new_predictions_data: contains 5000 BoxLists, full of predictions (per image)
        total_num_images = len(self.new_predictions_data)
        for ind, img_predictions in enumerate(self.new_predictions_data):
            #self.logger.log(f"Working on image {ind}/{total_num_images}")
            #Empty cropped predictions will be discarded
            img_id = self.coco_dataset.id_to_img_map[ind]
            img_file_path = os.path.join(self.org_annotation_images_path,
                                         self.coco_dataset.coco.imgs[img_id]['file_name'])
            #Load original image in order to generate the border bounding box as well as to visualize
            org_img_np_format = np.array(Image.open(img_file_path))
            org_img_width = self.coco_dataset.coco.imgs[img_id]["width"]
            org_img_height = self.coco_dataset.coco.imgs[img_id]["height"]

            #Resize the predictions to the original image dimensions for cropping inside the given FOV
            rsz_predictions = img_predictions.resize((org_img_width, org_img_height))
            rsz_predictions = rsz_predictions.convert("xywh")
            rsz_pred_masks_28_x_28 = rsz_predictions.get_field('mask')
            #Masker is necessary only if masks haven't been already resized.
            if list(rsz_pred_masks_28_x_28.shape[-2:]) != [org_img_height, org_img_width]:
                #This iff actually get called every time we process a new image
                # It is needed in order to filter out the logit scores lower than the Threshold
                #This is why it is possible to get after filtering segmentation empty with
                rsz_pred_masks_img_hight_width = masker(rsz_pred_masks_28_x_28.expand(1, -1, -1, -1, -1), rsz_predictions)
                rsz_pred_masks_img_hight_width = rsz_pred_masks_img_hight_width[0]
            rsz_pred_bboxes = rsz_predictions.bbox

            # Crop the bbox region
            #Cycle through all predictions for a given image and borderize them
            for i in range(len(rsz_predictions)):
                #self.logger.log(f"Working on prediction {i+1}/{len(rsz_predictions)} on image {self.coco_dataset.coco.imgs[img_id]['file_name']}")
                #Prediction mask before cropping
                sing_pred_on_sing_img_mask = rsz_pred_masks_img_hight_width[i, :, :, :]
                #Format [bbox_top_x_corner, bbox_top_y_corner, bbox_width, bbox_height]
                sing_pred_on_sing_img_bbox = rsz_pred_bboxes[i, :]

                pred_mask_binary_np = np.swapaxes(np.swapaxes(sing_pred_on_sing_img_mask.numpy(), 0, 2), 0, 1)[:,:,0]
                pred_mask_binary_image_form = Image.fromarray(pred_mask_binary_np)
                #The masks require that the diemsnions of the mask tensor are (1, height, width), so we convert to normal img format
                #(height, width)
                pred_mask_binary_np_3_channels = pred_mask_binary_np[:, :, None] * np.ones(3, dtype=int)[None, None, :]
                #We calculate the area of the total binary mask (first) as well as the
                #area of the binary mask inside the middle boundry
                current_image_pred_mask_tot_calculated_area = np.count_nonzero(pred_mask_binary_np)
                assert current_image_pred_mask_tot_calculated_area == pred_mask_binary_np.sum()

                if current_image_pred_mask_tot_calculated_area == 0: continue

                total_num_preds_grand += 1
                if current_image_pred_mask_tot_calculated_area <= 32**2:
                    total_num_preds_small += 1
                    continue
                elif current_image_pred_mask_tot_calculated_area <= 96**2:
                    total_num_preds_medium += 1
                    continue
                else:
                    total_num_preds_large += 1
                    continue

        return [total_num_preds_grand, total_num_preds_small, total_num_preds_medium, total_num_preds_large]