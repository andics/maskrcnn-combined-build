import numpy as np
import os
import json
import argparse
import sys
import matplotlib.pyplot as plt
import geopandas as gpd

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
import pycocotools.mask as maskUtils
from PIL import Image
from matplotlib import pyplot as plt
from itertools import groupby
import copy
from shapely import geometry, affinity
from dataset_processing.generate_border_annotations.utils.util_functions import Utilities_helper
#/home/projects/bagon/andreyg/Projects/Variable_Resolution/Datasets/border_based_evaluation/val2017_edited

def create_rle_mask(mask_image):
    width, height = mask_image.size

    rle_mask = {}
    for x in range(width):
        for y in range(height):
            pixel = mask_image.getpixel((x,y))
            # If the pixel is not black...
            if pixel != 0:
                pixel_str = str(pixel)
                if rle_mask.get(pixel_str) is None:
                    rle_mask[pixel_str] = Image.new('1', (width , height ))

                # Set the pixel value to 1 (default is 0), accounting for padding
                rle_mask[pixel_str].putpixel((x, y), 1)

    return rle_mask

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def main():
    parser = argparse.ArgumentParser(description="Annotations update")
    parser.add_argument(
        "--type",
        default="ch1",
        help="Type of filter to apply image to",
    )
    parser.add_argument(
        "--main-working-dir",
        default="val",
        help="Main directory with resources",
    )
    parser.add_argument(
        "--data",
        default="val",
        help="Type of filter to apply image to",
    )
    parser.add_argument(
        "--fov",
        default="ch",
        help="Type of filter to apply image to",
    )
    args = parser.parse_args()
    debugging = True
    utils_helper = Utilities_helper()

    type_strs = ['ch1', 'ch2', 'ch3', 'border_0.1']
    if args.type not in type_strs:
        pass
    prefix_idx = type_strs.index(args.type)
    if prefix_idx > 2:
        prefix_idx = 2
    print(args.type)

    data_strs = ['val', 'train']
    if args.data not in data_strs:
        pass

    fov_strs = ['ch', 'full']
    if args.fov not in fov_strs:
        pass

    main_working_dir = args.main_working_dir

    is_train = bool(data_strs.index(args.data))
    print(args.data)

    is_full_fov = bool(fov_strs.index(args.fov))
    print(args.fov)

    print("Reading in original annotations file")
    if not is_train:
        org_annot_file_path = os.path.join(main_working_dir, "instances_val2017.json")
    else:
        org_annot_file_path = os.path.join(main_working_dir, "instances_train2017.json")
    json_file = open(org_annot_file_path)
    org_metadata = json.load(json_file)
    json_file.close()
    org_segmentations = {}
    for annot in org_metadata['annotations']:
        image_id = annot['image_id']
        if image_id not in org_segmentations:
            org_segmentations[image_id] = []
        org_segmentations[image_id].append(annot)
    org_images_info = {}
    for image_info in org_metadata['images']:
        image_id = image_info['id']
        if image_id not in org_images_info:
            org_images_info[image_id] = []
        org_images_info[image_id].append(image_info)
    print("Finished reading original annotations file")

    new_annotations = []
    new_images_info = []
    img_cnt = 0
#    for image_id in list(org_segmentations.keys()):
    for image_id in list(org_images_info.keys()):
        new_image_name = str(prefix_idx + 1) + '{:010d}'.format(image_id)
        # print("Processing annotations for image: " + new_image_name)
        if not is_train:
            crop_info_json = os.path.join(main_working_dir, "val2017_edited", args.type.upper(), new_image_name+".json")
        else:
            crop_info_json = os.path.join(main_working_dir, "train2017", args.type.upper(), new_image_name+".json")
        #Pre-set data----------
        json_file = open(crop_info_json)
        crop_info = json.load(json_file)
        json_file.close()
        crop_bbox = crop_info['bbox']
        new_image_id = crop_info['id']
        new_segmentation = []
        non_crowd_seg_count = 0
        #----------------------
        if image_id in org_segmentations.keys():
            segmentation = copy.deepcopy(org_segmentations[image_id])
        else:
            segmentation = []
        org_img_width = org_images_info[image_id][0]['width']
        org_img_height = org_images_info[image_id][0]['height']
        if (crop_bbox[2] - crop_bbox[0] == org_img_width) & (
                crop_bbox[3] - crop_bbox[1] == org_img_height):
            new_segmentation = segmentation
            non_crowd_seg_count = 1
            for i in range(len(new_segmentation)):
                new_segmentation[i]['image_id'] = new_image_id
                new_segmentation[i]['id'] = int(str(prefix_idx + 1) + '{:010d}'.format(new_segmentation[i]['id']))
        else:
            for i in range(len(segmentation)):
                segm = segmentation[i]
                if segm['iscrowd'] != 0:
                    if type(segm['segmentation']['counts']) == list:
                        org_rle = maskUtils.frPyObjects(segm['segmentation'], org_img_height, org_img_width)
                    else:
                        org_rle = [segm['segmentation']]
                    org_rle_mask = maskUtils.decode(org_rle)

                    if not is_full_fov:
                        #  cropped FOV without padding
                        org_mask_img = Image.fromarray(org_rle_mask)
                        new_mask_img = org_mask_img.crop(crop_bbox)
                    else:
                        #  cropped FOV with padding to fit the full original FOV
                        new_rle_mask = np.zeros(org_rle_mask.shape, dtype="uint8")
                        crop_slice = np.s_[ crop_bbox[1]:crop_bbox[3] ,crop_bbox[0]:crop_bbox[2]  ]
                        org_rle_mask[crop_slice] = 0
                        new_mask_img = Image.fromarray(org_rle_mask)
                    new_mask = create_rle_mask(new_mask_img)

                    if bool(new_mask):
                        new_rle = maskUtils.encode(np.asfortranarray(np.array(new_mask['1'], dtype=np.uint8)))
                        new_area = float(maskUtils.area(new_rle))
                        if new_area>0:
                            new_segm = {
                                'segmentation': binary_mask_to_rle(np.array(new_mask['1'], dtype=np.uint8)),
                                'iscrowd': segm['iscrowd'],
                                'image_id': new_image_id,
                                'category_id': segm['category_id'],
                                'id': int(str(prefix_idx + 1) + '{:010d}'.format(segm['id'])),
                                'bbox': [round(float(t),2) for t in maskUtils.toBbox(new_rle)],
                                'area': new_area
                            }
                            new_segmentation.append(new_segm)
                else:
                    poly_segmentations = []
                    x_coord = segm['segmentation'][0][0::2]
                    y_coord = segm['segmentation'][0][1::2]
                    org_segm_poly = geometry.Polygon(zip(x_coord,y_coord))
                    #---Why simplify?
                    org_segm_poly = org_segm_poly.simplify(1.0, preserve_topology=False)
                    full_img_poly = geometry.Polygon(shell=[(0, 0), (0, org_img_height),
                                                     (org_img_width, org_img_height), (org_img_width, 0)],
                                                     holes=[[(crop_bbox[0], crop_bbox[1]), (crop_bbox[0], crop_bbox[3]),
                                                      (crop_bbox[2], crop_bbox[1]), (crop_bbox[2], crop_bbox[3])]])
                    if debugging:
                        utils_helper.plot_polygon(full_img_poly)
                    new_segm_poly = org_segm_poly.intersection(full_img_poly)
                    new_segm_poly = new_segm_poly.simplify(1.0, preserve_topology=False)
                    if not is_full_fov:
                        #  cropped FOV without padding
                        new_segm_poly = affinity.translate(new_segm_poly, -crop_bbox[0], -crop_bbox[1], 0)
                    if (new_segm_poly.is_empty == False) & (new_segm_poly.geom_type == 'GeometryCollection'):
                        new_segm_poly = [g for g in new_segm_poly if (g.geom_type == 'Polygon') | (g.geom_type == 'MultiPolygon')]
                        if len(new_segm_poly)>0:
                            new_segm_poly = new_segm_poly[0]
                        else:
                            new_segm_poly = geometry.Point([0,0])
                    if (new_segm_poly.is_empty==False) & ((new_segm_poly.geom_type=='Polygon') | (new_segm_poly.geom_type=='MultiPolygon')):
                        new_area = new_segm_poly.area
                        x, y, max_x, max_y = new_segm_poly.bounds
                        width = max_x - x; height = max_y - y
                        if (height <= 1) | (width <= 1):
                            print('bbox width or height less than 2 pixels ({},{})'.format(width, height))
                            continue
                        new_bbox = (x, y, width, height)
                        if new_segm_poly.geom_type == 'MultiPolygon':
                            for poly in new_segm_poly:
                                poly_segmentations.append([round(t,2) for t in np.array(poly.exterior.coords).ravel().tolist()])
                        else:
                            poly_segmentations.append([round(t,2) for t in np.array(new_segm_poly.exterior.coords).ravel().tolist()])
                        new_segm = {
                            'segmentation': poly_segmentations,
                            'iscrowd': segm['iscrowd'],
                            'image_id': new_image_id,
                            'category_id': segm['category_id'],
                            'id': int(str(prefix_idx + 1) + '{:010d}'.format(segm['id'])),
                            'bbox': [round(t,2) for t in new_bbox],
                            'area': new_area
                        }
                        new_segmentation.append(new_segm)
                        non_crowd_seg_count += 1

                    else:
                        if new_segm_poly.is_empty:
                            print('Cropped annotation is empty (imageID: {}, objID: {})'.format(new_image_id, segm['id']))
                            q=True
                        else:
                            print('Cropped annotation is non-empty (imageID: {}, objID: {})'.format(new_image_id, segm['id']))
                            if new_segm_poly.geom_type=='Point':
                                print('Cropped annotation is a point (imageID: {}, objID: {})'.format(new_image_id,segm['id']))


        if is_full_fov | ((len(new_segmentation)>0) & (non_crowd_seg_count>0)):
            for segm in new_segmentation:
                new_annotations.append(segm)
            # update and add image info only if there are annotations for this image
            new_img_info = copy.deepcopy(org_images_info[image_id][0])
            new_img_info['id'] = new_image_id
            new_img_info['file_name'] = crop_info['file_name']
            if not is_full_fov:
                # cropped FOV without padding
                new_img_info['width'] = int(crop_bbox[2] - crop_bbox[0])
                new_img_info['height'] = int(crop_bbox[3] - crop_bbox[1])
            else:
                # cropped FOV with padding to fit the full original FOV
                new_img_info['width'] = org_img_width
                new_img_info['height'] = org_img_height
            new_images_info.append(new_img_info)
        img_cnt += 1
        if img_cnt%1000==0:
            print('Processed {} images'.format(img_cnt))
    new_metadata = {
        'info' : copy.deepcopy(org_metadata['info']),
        'licenses': copy.deepcopy(org_metadata['licenses']),
        'images': new_images_info,
        'annotations': new_annotations,
        'categories': copy.deepcopy(org_metadata['categories'])
    }

    if not is_train:
#        new_annot_file_path = os.path.join("/home/dannyh/mraid11/Data/Work/Images", "mscoco_filt", "annotations", "instances_" + args.type + "_val2017.json")
        new_annot_file_path = os.path.join(main_working_dir, "instances_" + args.type + "_padded_val2017.json")

    else:
        new_annot_file_path = os.path.join(main_working_dir, "instances_" + args.type + "_padded_train2017.json")
    print("Writing new annotations to file " + new_annot_file_path)
    json_file = open(new_annot_file_path, 'w')
    json.dump(new_metadata, json_file)
    json_file.close()

if __name__ == "__main__":
    main()