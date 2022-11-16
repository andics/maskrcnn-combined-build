import sys
sys.path.remove('/workspace/object_detection')
sys.path.append('/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp')

import os
import json

from .model_class import Model_single

def generate_n_image_predictions(universal_plot_save_path, universal_preserve_original_dir_content):
    universal_config_path = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/configs/baseline_model/e2e_mask_rcnn_R_101_FPN_1x_equal.yaml"
    universal_expected_number_of_images = 5000

    models_weights_paths = (
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/original_resnet101/last_checkpoint",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/constant_resnet101",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/variable_resnet101",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/ch3_resnet101_ch3_equal/last_checkpoint",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/ch2_resnet101_ch2_equal/last_checkpoint",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/ch1_resnet101_ch1_equal/last_checkpoint",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/multi_resnet101_all_ch1_new/last_checkpoint",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/multi_resnet101_all_ch1_new/last_checkpoint",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp/trained_models/multi_resnet101_all_ch1_new/last_checkpoint",
    )

    model_names = (
        "Baseline_resnet101",
        "Constant_blurry_resnet101",
        "Variable_res_resnet101",
        "Single_CH3_resnet101",
        "Single_CH2_resnet101",
        "Single_CH1_resnet101",
        "MultiMixed_CH3_resnet101",
        "MultiMixed_CH2_resnet101",
        "MultiMixed_CH1_resnet101",
    )

    images_locations = (
        "/home/projects/bagon/shared/coco/val2017",
        "/home/projects/bagon/dannyh/data/coco_filt/val2017/Constant",
        "/home/projects/bagon/dannyh/data/coco_filt/val2017/Variable",
        "/home/projects/bagon/dannyh/data/coco_filt/val2017/CH3",
        "/home/projects/bagon/dannyh/data/coco_filt/val2017/CH2",
        "/home/projects/bagon/dannyh/data/coco_filt/val2017/CH1",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/comparative_visualizations/val2017_multi/CH3",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/comparative_visualizations/val2017_multi/CH2",
        "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/comparative_visualizations/val2017_multi/CH1",
    )

    generated_image_paths = []

    for current_model_weight, current_model_name, current_img_location in zip(models_weights_paths, model_names, images_locations):

        current_model = Model_single(config_path=universal_config_path,
                     model_weight = current_model_weight,
                     model_name = current_model_name,
                     images_location = current_img_location,
                     preserve_original_dir_content = universal_preserve_original_dir_content,
                     base_plot_save_dir = universal_plot_save_path,
                     confidence_threshold = 0.75)
        current_model.run()
        #assert len(current_model.images_saved_paths) == universal_expected_number_of_images, "Incorrect num of images!"

