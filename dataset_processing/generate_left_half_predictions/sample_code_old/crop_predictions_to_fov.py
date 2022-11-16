# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys
sys.path.remove('/workspace/object_detection')
sys.path.append('/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp')

from pycocotools.coco import COCO
from maskrcnn_benchmark.data.datasets.coco import COCODataset

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import json
import copy
import pycocotools.mask as mask_util
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
import torch.nn.functional as F
import argparse

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

def main():

    parser = argparse.ArgumentParser(description="MaskRCNN crop predictions to FOV")
    parser.add_argument(
        '-a',
        "--prediction_file",
        default="/home/labs/waic/shared/coco/predictions/instances_val2017.json",
        metavar="FILE",
        help="path to ground truth predictions file",
        type=str,
    )
    parser.add_argument(
        '-i',
        "--image_dir",
        default="/home/labs/waic/shared/coco/val2017",
        help="path to images folder",
        type=str,
    )
    parser.add_argument(
        '-s',
        "--save_dir",
        default="",
        metavar="FILE",
        help="Path to output save folder",
        type=str,
    )
    parser.add_argument(
        '-p',
        "--prediction_dir",
        help="Path to predictions file folder location",
        default="",
        type=str,
    )
    args = parser.parse_args()
    num_of_channels = 2 # number of channel FOVs
    annotations_file_path = args.prediction_file
    fov_crop_info_dir = "/home/labs/waic/dannyh/data/coco_filt/predictions/fov_crop_info/val2017"
    #images_location_base_dir = "/home/labs/waic/dannyh/data/coco_filt/val2017/Stacked_multi_channel"
    images_location_base_dir = args.image_dir
    original_images_location_base_dir = "/home/labs/waic/shared/coco/val2017"
    # save_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_baseline_val/"
    # predictions_dir = "/home/labs/waic/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/original_resnet101/inference/coco_2017_baseline_val"
    # save_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_constant_val/"
    # predictions_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_constant_val"
    # save_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_variable_val/"
    # predictions_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_variable_val"
    # save_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_multi_ch_val/"
    # predictions_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_multi_ch_val"
    # save_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_multi_ch_2d_val/"
    # predictions_dir = "/home/labs/waic/dannyh/work/exp_data/variable_resolution/inference/coco_2017_multi_ch_2d_val"
    save_dir = args.save_dir
    predictions_dir = args.prediction_dir

    coco = COCO(annotations_file_path)
    coco_dataset = COCODataset(annotations_file_path, images_location_base_dir, remove_images_without_predictions=False)

    print("Total number of images: ", len(coco.getImgIds()))

    # load model predictions from stored prediction file
    loaded_predictions = torch.load(os.path.join(predictions_dir,"predictions.pth"))
    print("Loaded predictions from file: {}".format(os.path.join(predictions_dir,"predictions.pth")))
    # initiate Masker class for projecting masks to fit image size
    masker = Masker(threshold=0.5, padding=1)

    for ch in range(num_of_channels):
        print("Cropping predictions to fit CH"+str(ch + 1)+" FOV:")
        fov_predictions = copy.deepcopy(loaded_predictions)
        for ind, img_predictions in enumerate(fov_predictions):
            # empty cropped predictions will be discarded
            inds_to_keep = []
            img_id = coco_dataset.id_to_img_map[ind]
            img_file_path = os.path.join(images_location_base_dir, coco_dataset.coco.imgs[img_id]['file_name'])
            # load original image in grayscale for visualization
            org_img = Image.open(img_file_path).convert('LA')
            org_img_width = coco_dataset.coco.imgs[img_id]["width"]
            org_img_height = coco_dataset.coco.imgs[img_id]["height"]
            # the resized dimensions of the image as in the model
            rsz_img_width = img_predictions.size[0]
            rsz_img_height = img_predictions.size[1]
            # resize the predictions to the original image dimensions for cropping inside the given FOV
            rsz_predictions = img_predictions.resize((org_img_width, org_img_height))
            rsz_predictions = rsz_predictions.convert("xywh")
            rsz_pred_masks = rsz_predictions.get_field('mask')
            # Masker is necessary only if masks haven't been already resized.
            if list(rsz_pred_masks.shape[-2:]) != [org_img_height, org_img_width]:
                rsz_pred_masks = masker(rsz_pred_masks.expand(1, -1, -1, -1, -1), rsz_predictions)
                rsz_pred_masks = rsz_pred_masks[0]
            rsz_pred_bboxes = rsz_predictions.bbox

            # loading FOV crop details for this image from stored JSON file
            # ch_img_id = str(ch + 1) + '{:010d}'.format(img_id)
            ch_img_id = str(ch + 1) + '{:010d}'.format(img_id)[3:]
            crop_info_json_file = open(os.path.join(fov_crop_info_dir, ch_img_id+".json"))
            fov_crop_info = json.load(crop_info_json_file)
            crop_info_json_file.close()
            crop_bbox = fov_crop_info['bbox'] # format: [left, top, right, bottom] (xyxy)

            if ((crop_bbox[2] - crop_bbox[0]) < org_img_width) | ((crop_bbox[3] - crop_bbox[1]) < org_img_height):
                # # for resizing the cropped masks back to 28x28
                # fov_pred_masks_rsz = torch.zeros([len(rsz_predictions),1,28,28])
                for i in range(len(rsz_predictions)):
                    # prediction mask before cropping
                    pred_mask = rsz_pred_masks[i]
                    #  pad the cropped prediction mask to fit the original image dimensions
                    new_rle_mask = np.zeros(pred_mask.shape[-2:], dtype="uint8")
                    crop_slice = np.s_[ crop_bbox[1]:crop_bbox[3] ,crop_bbox[0]:crop_bbox[2] ]
                    new_rle_mask[crop_slice] = pred_mask.numpy().squeeze()[crop_slice]
                    if new_rle_mask.any(): # in case the cropped prediction mask is not empty
                        fov_pred_mask = torch.from_numpy(new_rle_mask)
                        # create a new bounding box for the cropped prediction mask
                        mask = mask_util.encode(np.array(fov_pred_mask, order="F"))
                        fov_pred_bbox = mask_util.toBbox(mask).astype(int)
                        # # for resizing the cropped masks back to 28x28
                        # mask_bbox_slice = np.s_[ fov_pred_bbox[1]:fov_pred_bbox[1]+fov_pred_bbox[3] ,fov_pred_bbox[0]:fov_pred_bbox[0]+fov_pred_bbox[2] ]
                        # new_rle_mask_in_bbox = new_rle_mask[mask_bbox_slice]
                        # mask_in_bbox_rsz = F.interpolate(torch.from_numpy(new_rle_mask_in_bbox.astype('float32')).expand((1, 1, -1, -1)), size=(28, 28), mode='bilinear', align_corners=False)
                        # mask_in_bbox_rsz = mask_in_bbox_rsz[0]
                        # fov_pred_masks_rsz[i] = mask_in_bbox_rsz
                        rsz_pred_masks[i] = fov_pred_mask
                        rsz_pred_bboxes[i] = torch.from_numpy(fov_pred_bbox.astype('float32'))
                        inds_to_keep.append(i)
            # keep only non-empty cropped predictions
            rsz_pred_masks = rsz_pred_masks[inds_to_keep,:,:,:]
            rsz_pred_bboxes = rsz_pred_bboxes[inds_to_keep, :]
            # # for resizing the cropped masks back to 28x28
            # fov_pred_masks_rsz = fov_pred_masks_rsz[inds_to_keep,:,:,:]
            img_fov_crop_predictions = copy.deepcopy(rsz_predictions)
            img_fov_crop_predictions.bbox = rsz_pred_bboxes
            # # for resizing the cropped masks back to 28x28
            # img_fov_crop_predictions.extra_fields['mask'] = fov_pred_masks_rsz
            img_fov_crop_predictions.extra_fields['mask'] = rsz_pred_masks
            img_fov_crop_predictions.extra_fields['scores'] = img_fov_crop_predictions.extra_fields['scores'][inds_to_keep]
            img_fov_crop_predictions.extra_fields['labels'] = img_fov_crop_predictions.extra_fields['labels'][inds_to_keep]
            img_fov_crop_predictions = img_fov_crop_predictions.convert("xyxy")
            img_fov_crop_predictions = img_fov_crop_predictions.resize((rsz_img_width, rsz_img_height))
            fov_predictions[ind] = img_fov_crop_predictions

            # # visualize cropped predictions inside given FOV
            # pred_masks_to_show = copy.deepcopy(img_fov_crop_predictions.get_field('mask'))
            # if list(pred_masks_to_show.shape[-2:]) != [org_img_height, org_img_width]:
            #     pred_masks_to_show = masker(pred_masks_to_show.expand(1, -1, -1, -1, -1), img_fov_crop_predictions)
            #     pred_masks_to_show = pred_masks_to_show[0]
            # plt.figure(0)
            # plt.imshow(org_img)
            # ax = plt.gca()
            # for k in range(len(pred_masks_to_show)):
            #     new_mask_img_to_plt = np.ones((pred_masks_to_show[k].shape[1], pred_masks_to_show[k].shape[2], 3))
            #     color_mask = np.random.random((1, 3)).tolist()[0]
            #     for i in range(3):
            #         new_mask_img_to_plt[:, :, i] = color_mask[i]
            #     ax.imshow(np.dstack((new_mask_img_to_plt, pred_masks_to_show[k].numpy().squeeze() * 0.5)))
            # print("Processed image path: ", img_file_path)
            if (ind+1) % 500 == 0:
                print('Processed {} images'.format(ind+1))
        # store cropped predictions to file
        torch.save(fov_predictions, os.path.join(save_dir, "fov_ch" + '{:d}'.format(ch+1) + "_predictions.pth"))

if __name__ == "__main__":
    main()

