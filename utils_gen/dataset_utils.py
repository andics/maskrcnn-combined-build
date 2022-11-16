from maskrcnn_benchmark.utils.imports import import_file
import pycocotools.mask as mask
import cv2
import os
import numpy as np
from skimage import measure

def get_dataset_catalog(cfg):
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    return paths_catalog.DatasetCatalog


def get_dataset_info_from_cfg(cfg):
    dataset_catalogue = get_dataset_catalog(cfg)
    try:
        org_annotation_file = os.path.join(dataset_catalogue.DATA_DIR,
                                                dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["ann_file"])
        org_annotation_images_path = os.path.join(dataset_catalogue.DATA_DIR,
                                                       dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["img_dir"])

        # This part is hardcoded to acocunt for the fact that the dataset images could be located in the coco shared
        # but also in Danny's directory - as is the case for filtered images
        if os.path.exists(org_annotation_images_path):
            pass
        else:
            org_annotation_file = dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["ann_file"]
            org_annotation_images_path = dataset_catalogue.DATASETS[cfg.DATASETS.TEST[0]]["img_dir"]

            print(f"Successfully found path to annotation & image dir files for the {cfg.DATASETS.TEST} dataset!")
    except Exception as e:
        print(f"Attempted to load annotation & image dir files for the {cfg.DATASETS.TEST} dataset but failed:")
        print(e.__str__())

    return org_annotation_file, org_annotation_images_path


def rle_to_polygons(rle_encoding):
    maskedArr = mask.decode(rle_encoding)
    area = float((maskedArr > 0.0).sum())

    return polygonFromMask(maskedArr)

def polygonFromMask(maskedArr):
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError
    return segmentation

def polygonFromMaskV2(binary_mask, tolerance=1):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """

    def close_contour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons