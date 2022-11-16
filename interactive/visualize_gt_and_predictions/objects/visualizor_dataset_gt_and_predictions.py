import sys, os
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as func
from interactive.visualize_gt_and_predictions.utils import util_functions
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

import fiftyone as fo
import time
import torch
import numpy as np
import math
import pickle

class datasetVisualizer:
    _DEBUGGING = False
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

    def __init__(self, annotation_file_path, gt_images_base_path, predictions_file_path,
                 pickle_save_path, port, utils_helper):
        self.annotation_file_path = annotation_file_path
        self.gt_images_base_path = gt_images_base_path
        self.predictions_file_path = predictions_file_path
        self.pickle_save_path = pickle_save_path
        self.port = port
        self.utils_helper = utils_helper
        self.confidence_threshold = 0.5
        #Those are auxilary variables used for cases in which a .pickle file is provided
        self.skip_prediction_processing = False
        self.skip_pickling = False

        self.dataset_type = fo.types.COCODetectionDataset

    def load_dataset(self):
        # Import the dataset
        if self.pickle_save_path is not "None":
            try:
                self.dataset = self.utils_helper.load_pickled_51_session(self.pickle_save_path)
                print("Successfully loaded pickled FiftyOne App session! Proceeding to visualizations ...")
                self.skip_prediction_processing = True
                self.skip_pickling = True
            except:
                print(".pickle file was not found. Proceeding to loading predictions from disk & saving them on disk after...")
                self.dataset = fo.Dataset.from_dir(
                    dataset_type=self.dataset_type,
                    data_path=self.gt_images_base_path,
                    labels_path=self.annotation_file_path,
                    label_types=["segmentations"],
                )
        else:
            print(
                "No save path was provided. Proceeding to load the predictions real-time. Will not save session on disk!")
            self.dataset = fo.Dataset.from_dir(
                dataset_type=self.dataset_type,
                data_path=self.gt_images_base_path,
                labels_path=self.annotation_file_path,
                label_types=["segmentations"],
            )
            self.skip_pickling = True

        print(self.dataset)

    def run_visualization_gt(self):
        print(f"Visualising session with port {self.port}")
        self.session = fo.launch_app(self.dataset, remote=True, port=self.port)

    def load_predictions(self):
        if self.skip_prediction_processing:
            #No use yet: intended for session disk-loading
            print("Skipping prediction loading ...")
            return None

        self.org_predictions_data = torch.load(self.predictions_file_path)

    def add_prediction_fields_to_dataset(self):
        if self.skip_prediction_processing:
            # No use yet: intended for session disk-loading
            print("Skipping prediction processing ...")
            return None

        # Add predictions to samples
        '''
        for item in classes[1:]:
            if item.isnumeric():
                pass:
            else:
                pass
        '''

        #This confidence threshold is used for the masks: only logits with probability > threshold
        #are considered a 1 in the binary mask
        masker = Masker(threshold = self.confidence_threshold, padding = 1)

        counter = 0
        with fo.ProgressBar() as pb:
            for sample, sample_preds in pb(zip(self.dataset, self.org_predictions_data)):
                print(f"Processing image {counter}")
                counter = counter + 1
                # Load image
                #---USED-FOR-LOADING-IMAGES---
                img_file_path = os.path.join(self.gt_images_base_path, sample.filename)
                org_img_img_format = Image.open(img_file_path)

                if datasetVisualizer._DEBUGGING:
                    self.utils_helper.display_multi_image_collage(
                        ((org_img_img_format,
                          f"Image {sample.filename}"),),
                        [1, 1])
                #---USED-FOR-LOADING-PREDS---
                org_img_width = org_img_img_format.width
                org_img_height = org_img_img_format.height

                sample_preds_resized = sample_preds.resize((org_img_width, org_img_height))
                sample_preds_resized = sample_preds_resized.convert("xyxy")

                pred_masks = sample_preds_resized.get_field('mask')

                rsz_pred_masks = masker(pred_masks.expand(1, -1, -1, -1, -1), sample_preds_resized)[0]
                rsz_pred_bboxes = sample_preds_resized.bbox
                labels = sample_preds_resized.get_field("labels")
                scores = sample_preds_resized.get_field("scores")
                #----------------------------
                # Convert detections to FiftyOne format
                detections = []
                for label, score, box, mask in zip(labels,
                                             scores,
                                             rsz_pred_bboxes,
                                             rsz_pred_masks):
                    # Convert to [top-left-x, top-left-y, width, height]
                    # in relative coordinates in [0, 1] x [0, 1]
                    x1, y1, x2, y2 = box
                    rel_box = [x1 / org_img_width, y1 / org_img_height,
                               (x2 - x1) / org_img_width, (y2 - y1) / org_img_height]
                    pred_mask_binary_array_1d_for_disp = np.swapaxes(np.swapaxes(mask.numpy(), 0, 2), 0, 1)
                    pred_mask_binary_inside_bbox = pred_mask_binary_array_1d_for_disp[math.floor(y1):math.ceil(y2),
                                                   math.floor(x1):math.ceil(x2)]

                    detections.append(
                        fo.Detection(
                            label=datasetVisualizer.CATEGORIES[label],
                            bounding_box=rel_box,
                            confidence=score,
                            mask=pred_mask_binary_inside_bbox
                        )
                    )

                # Save predictions to dataset
                sample["predictions"] = fo.Detections(detections=detections)
                sample.save()
        print("Finished loading predictions!")

    def pickle_session(self):
        #This does not work yet! It is a functionality that we may wish to implement in the future
        return None
        if self.skip_pickling:
            print("Skipping session pickling ...")
            return None

        with open(self.pickle_save_path, 'wb') as f:
            pickle.dump(self.dataset, f)

        print("Successfully pickled the session. Ready to use next time!")
        '''
        self.dataset_3 = fo.Dataset.from_dir(
            dataset_type=self.dataset_type,
            data_path="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/data",
            labels_path="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/predictions.json",
            label_types=["segmentations"],
        )

        self.dataset_4 = fo.Dataset.from_dir(
            dataset_type=self.dataset_type,
            data_path="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/data",
            labels_path=["/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/predictions.json",
                         "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/labels/detections.json"],
            label_field=["predictions", "ground_truth"],
            label_types=["segmentations", "segmentations"],
        )

        self.dataset_4 = fo.Dataset.from_dir(
            dataset_type=self.dataset_type,
            data_path="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/data",
            labels_path="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/predictions.json",
            label_field=["predictions"],
            label_types=["segmentations"],
        )

        self.dataset_6 = fo.Dataset.from_dir(
            dataset_type=self.dataset_type,
            data_path="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/fifty_one/test2/data",
            labels_path="/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/variable_pretrained_resnet/baseline_resnet_norm/inference/90000_coco_2017_h_0.5_v_1.0_var/segm.json",
            label_field="predictions",
            label_types=["detections", "segmentations"],
        )
        '''