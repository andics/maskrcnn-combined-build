# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        #ROOT: The folder containing the images
        self.ids = sorted(self.ids)

        self.img_ext = ".jpg"

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        ann_root = os.path.dirname(ann_file)

        if ('train' in ann_file):
            fov_crop_info_file = os.path.join(ann_root, "train2017_fov_crop_multi_ch.json")
            self.folder_to_save_new_dataset = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/dataset_stacked_single_val_and_train_2017/stacked_single_from_combined_train"
        else:
            fov_crop_info_file = os.path.join(ann_root, "val2017_fov_crop_multi_ch.json")
            self.folder_to_save_new_dataset = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/dataset_stacked_single_val_and_train_2017/stacked_single_from_combined_val"

        json_file = open(fov_crop_info_file)
        crop_info = json.load(json_file)
        print("Loaded custom FOV data for cropped images!")

        json_file.close()
        self.fov_crop_info = crop_info['fov_crop_info']
        self.num_of_channels = 3

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        #It is important to make the transform function assigned to variable _transform.
        #This is due to the superclass having the same variable as an attribute, which
        #causes __getitem__ to apply an unknown horizontal flip before the complete image
        #is reconstructed
        self._transforms = transforms


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        #img: the CH3 image

        #print("What is returned item?: ", type(img))

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
		
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        new_dataset_img_name = str(img_id) + self.img_ext
        new_dataset_img_save_path = os.path.join(self.folder_to_save_new_dataset, new_dataset_img_name)

        if not os.path.exists(new_dataset_img_save_path):

            combined_ch_img = Image.new('RGB', img.size, (0, 0, 0,))

            for k in reversed(range(self.num_of_channels)):
                ch_img_id = (k+1)*(10**10) + (img_id - (round(img_id/(10**10)) * (10**10)) )
                # load resolution channel image from file
                ch_img_file = os.path.join(self.root, str(ch_img_id)+'.jpg')
                ch_img = Image.open(ch_img_file)
                # create an aligned channel image by positioning the channel's image inside the FOV

                for crop_info in self.fov_crop_info:
                    image_id = crop_info['id']
                    if image_id == ch_img_id:
                        ch_crop_bbox = crop_info['bbox']
                        break

                combined_ch_img.paste(ch_img, ch_crop_bbox)
                #print("One iteration of image pasting loop!")

            #print("Transformed image {img_name} with shape {img_shape}!".format(img_name=img_id, img_shape=list(img.size)))
            img = combined_ch_img
            img.save(new_dataset_img_save_path)
            #print("Created stacked image!")

        else:
            combined_ch_img = Image.open(new_dataset_img_save_path)
            img = combined_ch_img
            #print("Loaded stacked image!")


        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx


    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
