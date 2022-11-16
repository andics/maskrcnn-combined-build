# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from utils_gen import dataset_utils


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        #The superclass is initialized without being given transforms argument.
        #This is because if it is given a transforms argument, it applies in onto the image and target in the __getitem__
        #function. In turn, this intereferes with the transform function applied here after the __getitem__
        # (the now transformed image & target doesn't match the expected by 'transforms' DataType)
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

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
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field in the config
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # MODIFICATION
        # Also filter out tiny objects
        area_threshold = 1
        anno_org = anno
        anno = [obj for obj in anno if obj["area"] >= area_threshold]
        if len(anno_org) != len(anno):
            print(f"Filtered {len(anno_org)-len(anno)} objects with area smaller than {area_threshold}")
        # This modification is used to account for the shifted & cropped annotations all being in RLE format.
        # When COCO does the evaluation, it likes the annotations to be in Polygon format
        #print(f"Getting annotations for image {self.id_to_img_map[idx]}")
        if len(anno) != 0:
            if isinstance(anno[0]["segmentation"], dict):
                #print(f"Detected RLE-formatted annotations. Transforming...")
                #We have RLE format
                for annotation in anno:
                    #First convert to binary mask
                    try:
                        _ann_mask = self.coco.annToMask({'image_id': annotation['image_id'], 'segmentation': annotation['segmentation']})
                        poly_segm = dataset_utils.polygonFromMaskV2(_ann_mask)
                        annotation["segmentation"] = poly_segm
                    except Exception:
                        print("We r here")
        # ------------

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

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data