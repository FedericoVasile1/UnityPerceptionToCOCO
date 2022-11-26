# Dataset structure as saved by Unity Perception package: 
#      Perception_dataset_base_folder   
#      |- Dataset*
#      |   |- captures_000.json
#      |   |- captures_001.json
#      |   |- captures_*.json
#      |   |- annotation_definitions.json
#      |   |- metric_definitions.json
#      |   |- sensors.json
#      |
#      |- RGB*
#      |   |- rgb_2.png
#      |   |- rgb_3.png
#      |   |- rgb_*.png
#      |
#      |- InstanceSegmentation*
#      |   |- segmentation_2.png
#      |   |- segmentation_3.png
#      |   |- segmentation_*.png

import argparse
import glob
import os
import json

import cv2
import numpy as np
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon


def get_folders(input_dataset_path):
    dataset_folder = glob.glob(os.path.join(input_dataset_path, 'Dataset*'))
    if len(dataset_folder) != 1:
        raise Exception('No Dataset* folder found or multiple found')
    rgb_folder = glob.glob(os.path.join(input_dataset_path, 'RGB*'))
    if len(rgb_folder) != 1:
        raise Exception('No RGB* folder found or multiple found')
    inst_seg_folder = glob.glob(os.path.join(input_dataset_path, 
                                             'InstanceSegmentation*'))
    if len(inst_seg_folder) != 1:
        raise Exception('No InstanceSegmentation* folder found or multiple '
                        'found')
    return dataset_folder[0], rgb_folder[0], inst_seg_folder[0]


def create_coco_json():
    coco_json = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }
    return coco_json


def check_annotations(dataset_folder):
    with open(os.path.join(dataset_folder, 'annotation_definitions.json')) as f:
        annotations_definition = json.load(f)['annotation_definitions']
    flag = False
    for ann in annotations_definition:
        if ann is not None and ann['id'] == 'instance segmentation':
            flag = True
            break
    if not flag:
        raise Exception('No annotation with id \'instance segmentation\' found')

    with open(os.path.join(dataset_folder, 'metric_definitions.json')) as f:
        metric_definition = json.load(f)['metric_definitions']
    flag = False
    for ann in metric_definition:
        if ann is not None and ann['id'] == 'RenderedObjectInfo':
            flag = True
            break
    if not flag:
        raise Exception('No annotation with id \'RenderedObjectInfo\' found')


def get_ids(dataset_folder):
    with open(os.path.join(dataset_folder, 'metric_definitions.json')) as f:
        metric_definitions = json.load(f)['metric_definitions']

    category_id_to_category_name = {}
    for met_infos in metric_definitions:
        if met_infos is not None and met_infos['id'] == 'RenderedObjectInfo':
            for cat in met_infos['spec']:
                category_id_to_category_name[int(cat['label_id'])] = cat['label_name']
            break
    
    instance_id_to_category_id = {}
    instance_color_to_instance_id = {}
    metrics_files_list = glob.glob(
        os.path.join(dataset_folder, 'metrics_*.json')
    )
    # metrics_files_list.sort()
    for i in range(len(metrics_files_list)):    
        # I'm not sure the metrics_files_list.sort() works as I want it in  
        # any case, therefore construct the correct filename by myself 
        metr_filename = 'metrics_' + str(i).zfill(3) + '.json'
        appo = os.path.basename(metrics_files_list[i])
        metr_filename = metrics_files_list[i].replace(appo, metr_filename)
        with open(metr_filename, 'r') as f:
            metrics = json.load(f)['metrics']

        # Iterate over each frame
        for item in metrics:
            if item['metric_definition'] != 'RenderedObjectInfo':
                continue 
            for v in item['values']:
                instance_id_to_category_id[v['instance_id']] = v['label_id']
                color = (
                    v['instance_color']['r'], 
                    v['instance_color']['g'], 
                    v['instance_color']['b']
                )
                instance_color_to_instance_id[str(color)] = v['instance_id']

    return category_id_to_category_name, instance_id_to_category_id, instance_color_to_instance_id


def fill_coco_categories(coco_json, category_id_to_category_name):
    categories = []
    
    for id, name in category_id_to_category_name.items():
        cat = {}
        cat['supercategory'] = ''
        cat['id'] = id
        cat['name'] = name

        # Fastest way to add elem to list
        categories += cat,

    coco_json['categories'] = categories

    return coco_json


def fill_coco_images(dataset_folder, coco_json):
    images = []

    width, height = None, None
    captures_files_list = glob.glob(
        os.path.join(dataset_folder, 'captures_*.json')
    )
    # captures_files_list.sort()
    for i in range(len(captures_files_list)):    
        # I'm not sure the captures.sort() works as I want it in any case, 
        # therefore construct the correct filename by myself 
        cap_filename = 'captures_' + str(i).zfill(3) + '.json'
        appo = os.path.basename(captures_files_list[i])
        cap_filename = captures_files_list[i].replace(appo, cap_filename)
        with open(cap_filename, 'r') as f:
            captures = json.load(f)['captures']

        # Iterate over each frame
        for _, cap in enumerate(captures): 
            img = {}

            img['license'] = -1
            img['file_name'] = os.path.basename(cap['filename'])
            img['coco_url'] = ''
            if width is None and height is None:
                height, width, _ = cv2.imread(
                    os.path.join(os.path.dirname(dataset_folder), cap['filename'])
                ).shape
            # Assuming that all images in the dataset have the same size
            img['height'] = height
            img['width'] = width
            img['date_captured'] = ''
            img['flickr_url'] = ''
            img['id'] = int(
                os.path.basename(cap['filename']).split('.')[0].split('_')[1]
            )

            # Fastest way to add elem to list
            images += img,

    coco_json['images'] = images
    return coco_json


def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    # Taken from https://github.com/chrise96/image-to-coco-json-converter

    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation


def create_sub_masks(mask_image, width, height):
    # Taken from https://github.com/chrise96/image-to-coco-json-converter

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, mask_path, img_path):
    # Taken from https://github.com/chrise96/image-to-coco-json-converter

    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        if poly.geom_type == 'MultiPolygon':
            # see https://github.com/chrise96/image-to-coco-json-converter/issues/6
            poly = poly.convex_hull
            print('[WARNING]: the mask having path {} (associated to the '
                  'image having path {}) has been converted from '
                  'MultiPolygon to Polygon through convex_hull. It is '
                  'suggested to visualize back the obtained segmentation '
                  'onto the image to check whether it is reasonable.'
                  .format(mask_path, img_path))

        polygons.append(poly)

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
    
    return polygons, segmentations


def fill_coco_annotations(dataset_folder, coco_json, 
                          instance_id_to_category_id, 
                          instance_color_to_instance_id):
    annotations = []
    ann_counter = 0

    captures_files_list = glob.glob(
        os.path.join(dataset_folder, 'captures_*.json')
    )
    # captures_files_list.sort()
    for i in range(len(captures_files_list)):    
        # I'm not sure the captures.sort() works as I want it in any case, 
        # therefore construct the correct filename by myself 
        cap_filename = 'captures_' + str(i).zfill(3) + '.json'
        appo = os.path.basename(captures_files_list[i])
        cap_filename = captures_files_list[i].replace(appo, cap_filename)
        with open(cap_filename, 'r') as f:
            captures = json.load(f)['captures']

        print('[INFO]: processing {}'.format(cap_filename))
        # Iterate over each frame
        for cap in captures:
            image_id = int(
                os.path.basename(cap['filename']).split('.')[0].split('_')[1]
            )

            for elem in cap['annotations']:
                if elem['id'] != 'instance segmentation':
                    continue
            
                img_path = os.path.join(
                    os.path.dirname(dataset_folder), cap['filename']
                )
                mask_path = os.path.join(
                    os.path.dirname(dataset_folder), elem['filename']
                )
                mask = Image.open(mask_path).convert('RGB')
                w, h = mask.size

                sub_masks = create_sub_masks(mask, w, h)
                black_color = '(0, 0, 0)'
                if black_color in sub_masks:
                    del sub_masks['(0, 0, 0)']      # delete background 
                for color, sub_mask in sub_masks.items():
                    category_id = instance_id_to_category_id[
                        instance_color_to_instance_id[color]
                    ]
                    polygons, _ = create_sub_mask_annotation(
                        sub_mask, mask_path, img_path
                    )
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [
                            np.array(polygons[i].exterior.coords).ravel().tolist()
                        ]
                        annotation = create_annotation_format(
                            polygons[i], segmentation, image_id, 
                            category_id, ann_counter
                        )
                        
                        # Fastest way to add elem to list
                        annotations += annotation,
                        ann_counter += 1

    coco_json['annotations'] = annotations
    return coco_json


def convert(input_dataset_path, output_coco_file):
    dataset_folder, _, _ = get_folders(input_dataset_path)

    check_annotations(dataset_folder)

    coco_json = create_coco_json()

    category_id_to_category_name, \
        instance_id_to_category_id, \
        instance_color_to_instance_id = get_ids(dataset_folder)
    coco_json =  fill_coco_categories(coco_json, category_id_to_category_name)

    coco_json = fill_coco_images(dataset_folder, coco_json)

    coco_json = fill_coco_annotations(dataset_folder, coco_json,
                                      instance_id_to_category_id,
                                      instance_color_to_instance_id)

    with open(output_coco_file, 'w') as outfile:
        json.dump(coco_json, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_path', type=str, required=True)
    parser.add_argument('--output_coco_file', type=str, required=True)
    args = parser.parse_args()

    convert(args.input_dataset_path, args.output_coco_file)
