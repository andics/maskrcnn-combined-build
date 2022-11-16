import sys, os
from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0])
    print(path_main)
    sys.path.remove('/workspace/object_detection')
    sys.path.append(path_main)
    os.chdir(path_main)
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

import interactive.model_utils as model_utils
import fiftyone as fo
import time

full_path_to_annotation_file = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/annotations/instances_val2017_shifted_h_0.5_v_0.5.json"

#Original annotations
#full_path_to_annotation_file = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/annotations/original/instances_val2017.json"

full_path_to_val_dataset = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/datasets_coco_2017_center_shifted/val2017_shifted_h_0.5_v_0.5"
dataset_type = fo.types.COCODetectionDataset

model_config_path = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/configs/R-101-FPN/variable_pretrained_resnet/variable_pretrained_resnet_baseline_resnet_norm.yaml"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=dataset_type,
    data_path=full_path_to_val_dataset,
    labels_path=full_path_to_annotation_file,
    label_types=["detections", "segmentations"],
)
print(dataset)

#Load the variable resolution MaskRCNN
model = model_utils.load_model(model_config_path)
print("Model loaded & ready to go!")
session = fo.launch_app(dataset, remote=True, port=6001)

#Put a breakpoint here so you can modify the view
time.sleep(7200)