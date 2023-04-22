# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    """
    A class used to localize different datasets and their corresponding annotation files
    DATA_DIR: points to the base folder, which is prepended to all img_dirs & ann_files, unless they start with /
            Needs to be changed to /home/projects/bagon/dannyh/data/coco_filt/ occasionally
    """
    DATA_DIR = "/home/projects/bagon/shared"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2017_variable_train": {
            "img_dir": "coco_filt/train2017/Variable",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_variable_val": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2017_constant_train": {
            "img_dir": "coco_filt/train2017/Constant",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_constant_val": {
            "img_dir": "coco_filt/val2017/Constant",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2017_varch2_train": {
            "img_dir": "coco_filt/train2017/CH2",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_varch2_val": {
            "img_dir": "coco_filt/val2017/CH2",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2017_multi_ch1_train": {
            "img_dir": "coco_filt/train2017/CH1",
            "ann_file": "coco_filt/annotations/instances_ch1_train2017.json"
        },
        "coco_2017_multi_ch1_val": {
            "img_dir": "coco_filt/val2017/CH1",
            "ann_file": "coco_filt/annotations/instances_ch1_val2017.json"
        },
        "coco_2017_multi_ch1_padded_train": {
            "img_dir": "coco_filt/train2017/CH1.no_filt_padded",
            "ann_file": "coco_filt/annotations/instances_ch1_padded_train2017.json"
        },
        "coco_2017_multi_ch1_padded_val": {
            "img_dir": "coco_filt/val2017/CH1.no_filt_padded",
            "ann_file": "coco_filt/annotations/instances_ch1_padded_val2017.json"
        },
        "coco_2017_multi_ch2_padded_train": {
            "img_dir": "coco_filt/train2017/CH2.filt_padded",
            "ann_file": "coco_filt/annotations/instances_ch2_padded_train2017.json"
        },
        "coco_2017_multi_ch2_padded_val": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/CH2.filt_padded",
            "ann_file": "/home/projects/bagon/dannyh/data/coco_filt/annotations/instances_ch2_padded_val2017.json"
        },
        "coco_2017_multi_ch2_train": {
            "img_dir": "coco_filt/train2017/CH2",
            "ann_file": "coco_filt/annotations/instances_ch2_train2017.json"
        },
        "coco_2017_multi_ch2_val": {
            "img_dir": "coco_filt/val2017/CH2",
            "ann_file": "coco_filt/annotations/instances_ch2_val2017.json"
        },
        "coco_2017_ch3_train": {
            "img_dir": "coco_filt/train2017/CH3",
            "ann_file": "coco_filt/annotations/instances_ch3_train2017.json"
        },
        "coco_2017_ch3_val": {
            "img_dir": "coco_filt/val2017/CH3",
            "ann_file": "coco_filt/annotations/instances_ch3_val2017.json"
        },
        "coco_2017_mixed_ch_train": {
            "img_dir": "coco_filt/train2017_multi",
            "ann_file": "coco_filt/annotations/instances_multi_train2017.json"
        },
        "coco_2017_mixed_ch_val": {
            "img_dir": "coco_filt/val2017_multi",
            "ann_file": "coco_filt/annotations/instances_multi_val2017.json"
        },
        # Used to test models on borderized predictions: any model
        "coco_2017_val_0.1_border": {
            "img_dir": "coco/val2017",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/annotations/gen_ann_frame_0.1_thickness/instances_val2017.json"
        },
        # -----------------------------------------------
        "coco_2017_val_0.49_border": {
            "img_dir": "coco/val2017",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/annotations/gen_ann_frame_0.49_thickness/instances_val2017.json"
        },
        # Used to train/test multi_stacked_5_modules model & multi_stacked

        "coco_2017_multi_ch_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco_filt/annotations/instances_train2017.json"
        },
        "coco_2017_multi_ch_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco_filt/annotations/instances_val2017.json"
        },

        # ------------------------------------------------

        # Used to train multi-stacked-single
        "coco_2017_multi_stacked_single_train": {
            "img_dir": "coco_filt/train2017_multi",
            "ann_file": "coco_filt/annotations/instances_ch3_train2017.json"
        },
        "coco_2017_multi_stacked_single_val": {
            "img_dir": "coco_filt/val2017_multi",
            "ann_file": "coco_filt/annotations/instances_ch3_val2017.json"
        },
        "coco_2017_multi_stacked_single_premade_val": {
            "img_dir": "coco_filt/val2017/Stacked_multi_channel",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/annotations/multi_stacked_single/instances_multi_stacked_val2017.json"
        },
        # -----------------------------------

        # Used to test the center-shifted datasets for equiconst and variable: top-left corner
        "coco_2017_h_0.5_v_0.5_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable_shifted_h_0.5_v_0.5",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_0.5.json"
        },
        "coco_2017_h_0.5_v_0.5_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant_shifted_h_0.5_v_0.5",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_0.5.json"
        },
        # Used to test the center-shifted datasets for equiconst and variable: left image half
        "coco_2017_h_0.5_v_1.0_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable_shifted_h_0.5_v_1.0",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_1.0.json"
        },
        "coco_2017_h_0.5_v_1.0_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant_shifted_h_0.5_v_1.0",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_1.0.json"
        },
        # Used to test the center-shifted datasets for equiconst and variable: bottom-right corner
        "coco_2017_h_1.5_v_1.5_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable_shifted_h_1.5_v_1.5",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_1.5_v_1.5.json"
        },
        "coco_2017_h_1.5_v_1.5_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant_shifted_h_1.5_v_1.5",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations_polygon/instances_val2017_shifted_h_1.5_v_1.5.json"
        },
        # ------------------------------------

        # Used to test the center-shifted datasets for equiconst and variable WITHOUT FRAGMENTS (the white spots at the border): left image half
        "coco_2017_h_0.5_v_1.0_var_nf": {
            "img_dir": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/filtered_no_fragments/Variable_shifted_h_0.5_v_1.0",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_1.0.json"
        },
        "coco_2017_h_0.5_v_1.0_equiconst_nf": {
            "img_dir": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/filtered_no_fragments/Constant_shifted_h_0.5_v_1.0",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_1.0.json"
        },
        # ------------------------------------
        # Used to test the center-shifted datasets for equiconst and variable WITHOUT FRAGMENTS (the white spots at the border) and WITHOUT hallucinations: left image half
        "coco_2017_h_0.5_v_1.0_var_nf_trimmed_predictions": {
            "img_dir": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/filtered_no_fragments/Variable_shifted_h_0.5_v_1.0",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_1.0.json"
        },
        "coco_2017_h_0.5_v_1.0_equiconst_nf_trimmed_predictions": {
          "img_dir": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/filtered_no_fragments/Constant_shifted_h_0.5_v_1.0",
          "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_1.0.json"
        },
        #-------------------------------------

        #Used to test the Variable resolution and the equiconstant on objects which span
        #the high and low resolution regions in particular area-ratios (bins)
        #-------------------------------------
        "coco_2017_res_bin_0.0_0.1_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.0_0.1_instances_val2017.json"
        },
        "coco_2017_res_bin_0.1_0.2_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.1_0.2_instances_val2017.json"
        },
        "coco_2017_res_bin_0.2_0.3_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.2_0.3_instances_val2017.json"
        },
        "coco_2017_res_bin_0.3_0.4_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.3_0.4_instances_val2017.json"
        },
        "coco_2017_res_bin_0.4_0.5_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.4_0.5_instances_val2017.json"
        },
        "coco_2017_res_bin_0.5_0.6_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.5_0.6_instances_val2017.json"
        },
        "coco_2017_res_bin_0.6_0.7_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.6_0.7_instances_val2017.json"
        },
        "coco_2017_res_bin_0.7_0.8_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.7_0.8_instances_val2017.json"
        },
        "coco_2017_res_bin_0.8_0.9_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.8_0.9_instances_val2017.json"
        },
        "coco_2017_res_bin_0.9_1.0_var": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.9_1.0_instances_val2017.json"
        },
        #---EQUICONST---
        "coco_2017_res_bin_0.0_0.1_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.0_0.1_instances_val2017.json"
        },
        "coco_2017_res_bin_0.1_0.2_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.1_0.2_instances_val2017.json"
        },
        "coco_2017_res_bin_0.2_0.3_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.2_0.3_instances_val2017.json"
        },
        "coco_2017_res_bin_0.3_0.4_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.3_0.4_instances_val2017.json"
        },
        "coco_2017_res_bin_0.4_0.5_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.4_0.5_instances_val2017.json"
        },
        "coco_2017_res_bin_0.5_0.6_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.5_0.6_instances_val2017.json"
        },
        "coco_2017_res_bin_0.6_0.7_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.6_0.7_instances_val2017.json"
        },
        "coco_2017_res_bin_0.7_0.8_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.7_0.8_instances_val2017.json"
        },
        "coco_2017_res_bin_0.8_0.9_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.8_0.9_instances_val2017.json"
        },
        "coco_2017_res_bin_0.9_1.0_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.9_1.0_instances_val2017.json"
        },

        #---TEST-DATASET-FOR-LOCAL-WORK---
        "test_coco_2017_res_bin_0.0_0.1_var": {
            "img_dir": "W:/bagon/dannyh/data/coco_filt/val2017/Variable",
            "ann_file": "Q:/Projects/Variable_resolution/Programming/maskrcnn-combined-build/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.0_0.1_instances_val2017.json"
        },
        "test_coco_2017_res_bin_0.0_0.1_equiconst": {
            "img_dir": "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
            "ann_file": "Q:/Projects/Variable_resolution/Programming/maskrcnn-combined-build/dataset_processing/generate_resolution_bin_annotations/filtered_annotations/0.0_0.1_instances_val2017.json"
        },
        #-------------------------------------

        "coco_2017_multi_debug_train": {
            "img_dir": "coco_filt/train2017_multi",
            "ann_file": "coco_filt/annotations/instances_multi_debug_train2017.json"
        },
        "coco_2017_multi_debug_val": {
            "img_dir": "coco_filt/val2017_multi",
            "ann_file": "coco_filt/annotations/instances_multi_debug_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "voc_2007_train": {
            "coco_dir": "data/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "data/VOC2007/JPEGImages",
            "ann_file": "data/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "data/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "data/VOC2007/JPEGImages",
            "ann_file": "data/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "data/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "data/VOC2007/JPEGImages",
            "ann_file": "data/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "data/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "data/VOC2012/JPEGImages",
            "ann_file": "data/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "data/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "data/VOC2012/JPEGImages",
            "ann_file": "data/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "data/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "data/images",
            "ann_file": "data/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "data/images",
            "ann_file": "data/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "data/images",
            "ann_file": "data/annotations/instancesonly_filtered_gtFine_test.json"
        }
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
