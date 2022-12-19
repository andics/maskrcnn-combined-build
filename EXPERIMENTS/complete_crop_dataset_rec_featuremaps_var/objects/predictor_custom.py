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


class COCO_predictor(object):
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

    def __init__(
        self,
        cfg,
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
    ):

        self.cfg = cfg.clone()
        self.cfg.freeze()
        self.model = build_detection_model(self.cfg)
        self.device = torch.device(self.cfg.MODEL.DEVICE)
        self.model.to(self.cfg.MODEL.DEVICE)
        self.min_image_size = min_image_size
        self.transforms = build_transforms(self.cfg, False)
        self.display_total_predictions = display_total_predictions

        # Weight loading
        checkpointer = DetectronCheckpointer(self.cfg, self.model, save_dir=weight_file_dir)
        if os.path.isfile(weight_file_dir):
            _ = checkpointer.load(weight_file_dir)
        else:
            _ = checkpointer.load(os.path.join(weight_file_dir, default_weight_file_name))

        self.model.eval()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.use_conf_threshold = use_conf_threshold
        self.max_num_pred = max_num_pred
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim
        self.show_bboxes = show_bboxes
        self.choose_top_n_if_none = choose_top_n_if_none
        self.top_n_if_none = top_n_if_none
        self.top_if_none_critical_number = top_if_none_critical_number


    def run_on_opencv_image(self, image_model_acceptable_format):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        #image - image tensor-like format
        #Convert to tensor
        image = image_model_acceptable_format.convert("RGB")
        image = np.array(image)

        predictions = self.compute_prediction(image_model_acceptable_format, image)
        total_predictions = len(predictions)
        print(predictions)

        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)

        if self.show_bboxes:
            result = self.overlay_boxes(result, top_predictions)

        '''
        Problematic code. Used just for visual. so not vital:
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        result, prediction_dictionary = self.generate_class_names(result, top_predictions)


        if self.display_total_predictions:
            prediction_dictionary.append("Total : " + str(total_predictions))
        '''

        return None


    def compute_prediction(self, image_model_acceptable_format, image_tensor_like):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        # convert to an ImageList, padded so that it is divisible by
        image_model_acceptable_format = self.transforms(image_model_acceptable_format).to(self.cfg.MODEL.DEVICE).unsqueeze(0)

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_model_acceptable_format)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original_model image size
        height, width = image_tensor_like.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions_original = predictions
        scores = predictions.get_field("scores")

        if self.use_conf_threshold:
            keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
            predictions = predictions[keep]
            scores = predictions.get_field("scores")

        _, idx = scores.sort(0, descending=True)
        if len(idx) > self.max_num_pred:
            idx = idx[0:self.max_num_pred]
           # print('Selected top ', str(self.max_num_pred), ' predictions! \n')

        if len(idx) < self.top_if_none_critical_number and self.choose_top_n_if_none:
            scores = predictions_original.get_field("scores")
            _, idx = scores.sort(0, descending=True)
            print("NOT enough thresholded predictions => choosing TOP N instead!")
            if len(idx) < self.top_n_if_none:
                idx = idx
            else:
                idx = idx[0:self.top_n_if_none]
            predictions = predictions_original

        return predictions[idx]



    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def generate_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        text_scale = .6

        prediction_numerator = 0
        prediction_dictionary = []
        for box, score, label in zip(boxes, scores, labels):
            [x_top, y_top, x_bottom, y_bottom] = [box[0], box[1], box[2], box[3]]
            x, y = (x_top + x_bottom)/2, (y_top + y_bottom)/2
            #print('Inside prediction loop! \n')

            prediction_text = template.format(label, score)
            prediction_dictionary.append(str(prediction_numerator) + ' - ' + prediction_text)
            str_for_img = str(prediction_numerator)

            cv2.putText(
                image, str_for_img, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1
            )
            prediction_numerator = prediction_numerator + 1

        return image, prediction_dictionary
