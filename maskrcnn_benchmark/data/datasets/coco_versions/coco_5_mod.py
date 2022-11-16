# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

# DH: multi-channel
import os
import json
from PIL import Image
import numpy as np
import random

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)

        #root: image dir - coco/train2017
        #ann_file - self explanatory

        self.ids = sorted(self.ids)

        # DH: multi-channel
        # number of resolution channels in the input
        self.num_of_channels = 3
        # channels' FOV crop info loaded from JSON file stored with in the annotations folders
        ann_root = os.path.dirname(ann_file)
        self.ann_root = ann_root
        # identify training/testing mode
        if ('train' in ann_file):
            fov_crop_info_file = os.path.join(ann_root, "train2017_fov_crop_multi_ch.json")
            self.img_dir = os.path.join(os.path.dirname(ann_root), 'train2017_multi')
            # 50% probability for horizontal flip of the input in training mode
            self.flip_prob = 0.5
        else:
            fov_crop_info_file = os.path.join(ann_root, "val2017_fov_crop_multi_ch.json")
            self.img_dir = os.path.join(os.path.dirname(ann_root), 'val2017_multi')
            # no horizontal flip of the input in testing mode
            self.flip_prob = 0
        json_file = open(fov_crop_info_file)
        crop_info = json.load(json_file)
        print("Loaded custom FOV data for stacked images!")

        json_file.close()
        self.fov_crop_info = crop_info['fov_crop_info']

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        # DH: multi-channel
        # modified transforms
        self.img_transforms = transforms['img_transform']
        self.transform_noflip = transforms['transform_noflip']
        self.transform_hflip = transforms['transform_hflip']

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        #print("Working with image ID BEFORE pre-processing: ", img_id)

        #Quick fix for evaluating and training mode
        #if not ("train" in self.img_dir):
        #    img_id = int(str(img_id)[1:])

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
        # if len(target)==0:
        # print('#################################################')
        # print('Image ID:')
        # print(anno[0]['image_id'])
        # print(self.coco.loadImgs(anno[0]['image_id'])[0]['file_name'])
        # print('Original boxes:')
        # print(boxes)
        # print('Target converted boxes:')
        # print(target.bbox)
        # print('Image size:')
        # timg = Image.open(os.path.join(self.root, self.coco.loadImgs(anno[0]['image_id'])[0]['file_name'])).convert('RGB')
        # print(timg.size)
        # print(img.size)

        # DH: multi-channel
        # random horizontal flip of the input
        flip_fl = random.random() < self.flip_prob
        for k in range(self.num_of_channels):
            #ch_img_id = (k+1)*(10**10) + (img_id - (round(img_id/(10**10)) * (10**10)) )

            #print("Generating multi-channel tensor. Working with image ID: ", img_id)

            ch_img_id = (k + 1) * (10 ** 10) + img_id
            # load resolution channel image from file
            ch_img_file = os.path.join(self.img_dir, str(ch_img_id) + '.jpg')
            ch_img = Image.open(ch_img_file)

            #print("Opening image: ", ch_img_file)

            # create an aligned channel image by positioning the channel's image inside the FOV
            aligned_ch_img = Image.new('RGB', img.size, 0)
            for crop_info in self.fov_crop_info:
                image_id = crop_info['id']
                if image_id == ch_img_id:
                    ch_crop_bbox = crop_info['bbox']
                    break
            aligned_ch_img.paste(ch_img, ch_crop_bbox)
            # apply the transforms on every channel separately
            if flip_fl:
                if self.transform_hflip is not None:
                    aligned_ch_img, ch_crop_bbox = self.transform_hflip(aligned_ch_img, BoxList(
                        torch.as_tensor(ch_crop_bbox).reshape(-1, 4), aligned_ch_img.size, mode='xyxy'))
            else:
                if self.transform_noflip is not None:
                    aligned_ch_img, ch_crop_bbox = self.transform_noflip(aligned_ch_img, BoxList(
                        torch.as_tensor(ch_crop_bbox).reshape(-1, 4), aligned_ch_img.size, mode='xyxy'))
            # convert transformed channel image back to Numpy image
            aligned_ch_img = np.asarray(aligned_ch_img.permute([1, 2, 0]))
            ch_crop_bbox = np.maximum([0, 0, 0, 0], np.round(np.asarray(ch_crop_bbox.bbox)))
            ch_crop_bbox = ch_crop_bbox[0].astype(int).tolist()
            ch_fov_slice = np.s_[ch_crop_bbox[1]:ch_crop_bbox[3], ch_crop_bbox[0]:ch_crop_bbox[2]]
            tmp_img = aligned_ch_img
            # recreate the aligned image with zero padding outside the channel's FOV
            aligned_ch_img = np.zeros(aligned_ch_img.shape)
            aligned_ch_img[ch_fov_slice] = tmp_img[ch_fov_slice]
            # updating the multi-channel image by concatenating the current channel's image to the previous ones
            if k == 0:
                multi_ch_img = aligned_ch_img
            else:
                multi_ch_img = np.concatenate((multi_ch_img, aligned_ch_img), axis=2)

        # converting the combined multi-channel image to Tensor
        # print("Turning multi-channel image into a tensor for evaluation!")
        # print("Shape of NumPy array: ", np.shape(multi_ch_img))
        multi_ch_img = torch.tensor(multi_ch_img, dtype=torch.float32).permute([2, 0, 1])

        #Used to add extra channels to the multi stacked - used to test if the input overlapping is a problem for the NN
        #----------------------------------------------------------------------------------------------------------
        #Adding the information for the mixed modules to the tensor
        #Order: ch1_ch2 - 0:6, ch2_ch3 - 6:12, ch1, ch2, ch3
        #multi_ch_img = torch.cat((multi_ch_img[0:6, :, :], multi_ch_img[3:9, :, :], multi_ch_img[:, :, :]), dim=0)
        #----------------------------------------------------------------------------------------------------------

        # apply the transforms to the annotation labels (target)
        if flip_fl:
            if self.transform_hflip is not None:
                img, target = self.transform_hflip(img, target)
        else:
            if self.transform_noflip is not None:
                img, target = self.transform_noflip(img, target)

        return multi_ch_img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

