# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
# from PIL import Image
# import os


from maskrcnn_benchmark.utils.registry import Registry
from maskrcnn_benchmark.data.datasets.coco_versions import coco
from maskrcnn_benchmark.data.datasets.coco_versions import coco_single_isolated_norm
from maskrcnn_benchmark.data.datasets.coco_versions import coco_5_mod
from maskrcnn_benchmark.data.datasets.coco_versions import coco_multi_stacked
from maskrcnn_benchmark.data.datasets.coco_versions import coco_multi_stacked_single

class COCODataset():
    def __new__(cls, ann_file, root, remove_images_without_annotations, cfg, transforms=None):
        assert cfg.DATALOADER.COCO_VERSION in _DATALOADER_COCO_VERSIONS, \
            "DATALOADER.COCO_VERSION: {} are not registered in registry".format(
                cfg.DATALOADER.COCO_VERSION
            )
        print("DATALOADER.COCO_VERSION: using {} COCODataset for model".format(
            cfg.DATALOADER.COCO_VERSION))

        return _DATALOADER_COCO_VERSIONS[cfg.DATALOADER.COCO_VERSION](ann_file, root, remove_images_without_annotations, transforms)


_DATALOADER_COCO_VERSIONS = Registry({
    "dataloader_COCO_version_baseline": coco.COCODataset,
    "dataloader_COCO_version_equiconst": coco.COCODataset,
    "dataloader_COCO_version_variable": coco.COCODataset,
    "dataloader_COCO_version_multi_mixed": coco.COCODataset,
    "dataloader_COCO_version_single": coco.COCODataset,
    "dataloader_COCO_version_single_isolated_norm": coco_single_isolated_norm.COCODataset,
    "dataloader_COCO_version_multi_stacked_single": coco_multi_stacked_single.COCODataset,
    "dataloader_COCO_version_multi_stacked": coco_multi_stacked.COCODataset,
    "dataloader_COCO_version_5_mod": coco_5_mod.COCODataset,
})
