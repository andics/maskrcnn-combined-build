# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import os
import numpy as np
from torchvision import transforms as T
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog


class Model_predictor(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self, cfg):

        self.cfg = cfg.clone()
        self.cfg.freeze()

        self.__init_helper_obj__()

        self.load_model()

        #TODO: Infer the original class arguments from the config file and add Demo functions progressively as you need more.
        # This will allow you to understand exactly what is going on


    def __init_const__(self):
        self.expected_image_extensions = (".jpg", ".png")

        self.columns_on_pred_table = 3

        # Measured in pixels
        self.prediction_image_width = 640
        self.prediction_image_height = 1100
        # DPI is redundant, can be any value
        self.dpi = 100

        # NOT including total_num_predictions
        self.max_num_preds = 23
        self.show_bboxes = False

        self.choose_top_n_if_none = True
        # Under how many predictions should the tresholded ones be to take the top N ones instead?
        self.critical_number_to_select_top_n_predictions = 5
        self.top_n_if_none = 5


    def __init_helper_obj__(self):
        self.model_checkpoint_dir = self.cfg.OUTPUT_DIR

        #The dataset's image folder and anootation_file location are
        #recorded as dictionary elements of the get() method of this class
        self.dataset_catalog = DatasetCatalog()

        self.dataset_location_images = self.dataset_catalog.get(self.cfg.MODEL_PREDICTOR.DATASET_NAME).args.root
        self.dataset_location_annotation_file = self.dataset_catalog.get(self.cfg.MODEL_PREDICTOR.DATASET_NAME).args.ann_file
        print("test")


    def gather_images(self):
        self.images_names_w_extensions = []

        for file in os.listdir(self.dataset_location_images):
            if file.lower().endswith(self.expected_image_extensions):
                self.images_names_w_extensions.append(file)

        print("Gathered images for model %s!" % self.model_name)


    def load_model(self):
        self.model_predictor = COCO_predictor(cfg=self.cfg, custom_config_file=self.config_path,\
                                         weight_file_dir=self.model_weight, \
                                         use_conf_threshold=True, confidence_threshold=self.confidence_threshold, \
                                              max_num_pred=Model_single.max_num_preds, min_image_size=60,
                                              masks_per_dim=3, show_bboxes=Model_single.show_bboxes,
                                              top_if_none_critical_number = Model_single.critical_number_to_select_top_n_predictions,
                                              choose_top_n_if_none=Model_single.choose_top_n_if_none,
                                              top_n_if_none=Model_single.top_n_if_none)

        print("Loaded model %s!" % self.model_name)


    def infer_on_model(self):
        #The main function where image predictions will be generated
        self.image_predictions = []




'''
        custom_config_file,
        weight_file_dir,
        default_weight_file_name = "model_final.pth",
        confidence_threshold=0.5,
        use_conf_threshold=True,
        max_num_pred=5,
        show_mask_heatmaps=False,
        masks_per_dim=1,
        min_image_size=60,
        display_total_predictions = True,
        show_bboxes=False,
        choose_top_n_if_none=False,
        top_n_if_none=5,
        top_if_none_critical_number=3
'''