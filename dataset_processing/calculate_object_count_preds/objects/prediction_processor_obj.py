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

    def __init__(self, org_predictions_path, logger):
        self.org_predictions_path = org_predictions_path
        self.logger = logger


    def read_predictions(self):
        self.org_predictions_data = torch.load(self.org_predictions_path)
        self.new_predictions_data = copy.deepcopy(self.org_predictions_data)

        self.logger.log(f"Loaded PTH prediction data from: {self.org_predictions_path}")


    def count_predictions(self):
        #Workflow should be as follows:
        #Check workflow.txt file in this directory

        #Initiate Masker class for projecting masks to fit image size
        #This class is used to transform the Maskrcnn-native 28x28 mask to fit the image size and look like something
        #new_predictions_data: contains 5000 BoxLists, full of predictions (per image)

        total_num_images = len(self.new_predictions_data)
        total_num_preds_small = 0
        total_num_preds_medium = 0
        total_num_preds_large = 0
        total_grand = 0

        for ind, img_predictions in enumerate(self.new_predictions_data):
            total_grand += img_predictions.get_field("labels").__len__()
            '''
            if annotation['area'] <= 32**2:
                total_binary_mask_small += current_image_binary_mask_resized
                continue
            elif annotation['area'] <= 96**2:
                total_binary_mask_medium += current_image_binary_mask_resized
                continue
            else:
                total_binary_mask_large += current_image_binary_mask_resized
                continue
            '''

        return total_grand
