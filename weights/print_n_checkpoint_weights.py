from utils_gen import model_utils as utils

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from demo.predictor_custom import COCO_predictor

import numpy as np


def main():
    folder_to_scan = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/trained_models/equiconst_pretrained_resnet/baseline/last_checkpoint"

    all_model_files, all_model_file_names = utils.find_all_files_with_ext(folder_to_scan, ".pth")

    model_recorded_weights_at_points = []

    for i in range(len(all_model_files)):
        file = all_model_files[i]
        file_name = all_model_file_names[i]

        current_weight = extract_model_weight_at_point(file)
        print(file_name, " - ", current_weight)

        model_recorded_weights_at_points.append(current_weight)

    #print(*model_recorded_weights_at_points, sep=" || ")



def extract_model_weight_at_point(model_path):

    universal_config_file = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/configs/R-101-FPN/equiconst_pretrained_resnet/equiconst_pretrained_resnet_baseline.yaml"
    config_file = universal_config_file
    cfg.merge_from_file(config_file)

    utils.setup_env_variables()
    model_loaded = COCO_predictor(cfg = cfg, custom_config_file = universal_config_file, \
                                     weight_file_dir = model_path, \
                                     use_conf_threshold = False, max_num_pred = 5, min_image_size = 60, masks_per_dim = 3)

    weight = model_loaded.model.backbone.body._modules.get('stem')._modules.get('bn1').weight.data.cpu().numpy().flat[0]

    #weight = model_loaded.model.backbone.body.layer3._modules['0'].conv1.weight.data.cpu().numpy().flat[0]
    #print(model_loaded.model.backbone.body._modules.get('stem')._modules.get('conv1')._parameters.get('weight').data.shape)

    return weight


def convert_none_to_str(data):
    if isinstance(data, list):
        data[:] = [convert_none_to_str(i) for i in data]
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = convert_none_to_str(v)
    return 'None' if data is None else data


if __name__=="__main__":
    main()