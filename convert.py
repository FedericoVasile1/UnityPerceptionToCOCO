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
import multiprocessing

import cv2
import numpy as np
from pycocotools.mask import encode as mask_to_rle


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

    flag = False
    for ann in annotations_definition:
        if ann is not None and ann['id'] == 'bounding box':
            flag = True
            break
    if not flag:
        raise Exception('No annotation with id \'bounding box\' found')

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

    # get the category id to name mapping from the RenderedObjectInfo annotator
    category_id_to_category_name = {}
    for met_infos in metric_definitions:
        if met_infos is not None and met_infos['id'] == 'RenderedObjectInfo':
            for cat in met_infos['spec']:
                category_id_to_category_name[int(cat['label_id'])] = cat['label_name']
            break

    # check that the category id to category name mapping is coherent with 
    # the one defined in the bounding box annotator and in the 
    # instance segmentation annotator
    with open(os.path.join(dataset_folder, 'annotation_definitions.json')) as f:
        annotation_definitions = json.load(f)['annotation_definitions']
    appo = {}
    for ann_infos in annotation_definitions:
        if ann_infos is not None and ann_infos['id'] == 'bounding box':
            for cat in ann_infos['spec']:
                appo[int(cat['label_id'])] = cat['label_name']
            break
    if category_id_to_category_name != appo:
        raise Exception('The category id to category name mapping defined '
                        'by the RenderedObjectInfo annotator is different '
                        'from the one defined by the bounding box annotator')
    appo = {}
    for ann_infos in annotation_definitions:
        if ann_infos is not None and ann_infos['id'] == 'instance segmentation':
            for cat in ann_infos['spec']:
                appo[int(cat['label_id'])] = cat['label_name']
            break
    if category_id_to_category_name != appo:
        raise Exception('The category id to category name mapping defined '
                        'by the RenderedObjectInfo annotator is different '
                        'from the one defined by the instance segmentation '
                        'annotator')
    
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
                # TODO: in case the images of the dataset have all the same 
                #       dimesion, comment this line and hard-code the values
                #       to speed up processing 
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


def create_annotation_format(segmentation, area, image_id, bbox, 
                             category_id, annotation_id):
    # Taken from https://github.com/chrise96/image-to-coco-json-converter
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


def process_capture_file(
    capture_filename, dataset_folder, instance_id_to_category_id, 
    instance_color_to_instance_id
):
    with open(capture_filename, 'r') as f:
        captures = json.load(f)['captures']
    print('[INFO]: processing {}'.format(capture_filename))

    annotations_list = []
    # Iterate over each frame
    for cap in captures:
        image_id = int(
            os.path.basename(cap['filename']).split('.')[0].split('_')[1]
        )

        for elem_instseg in cap['annotations']:
            if elem_instseg['id'] != 'instance segmentation':
                continue
        
            mask_path = os.path.join(
                os.path.dirname(dataset_folder), elem_instseg['filename']
            )

            mask = cv2.imread(mask_path)
            dims = np.nonzero(mask)
            if dims[0].size == 0 and dims[1].size == 0 and dims[2].size == 0:
                # mask is all black, i.e., no annotations
                continue
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            assert mask.shape[-1] == 3, mask_path  # RGB
            assert mask.ndim == 3, mask_path       # H, W, C
            assert mask.max() > 1, mask_path # be sure that we are in [0, 255] instead of [0, 1] range

            # TODO: take all the possible colors available in the dataset
            #       from the metadata, then perform the product between 
            #       the mask image and all the colors, keep only the 
            #       non-zero binary images
            unique_colors = np.unique(
                mask.reshape(-1, mask.shape[2]), axis=0
            )

            for u_q in unique_colors:
                if (u_q == [0, 0, 0]).all():    # TODO according to the todo above
                    continue
                
                bin_mask = (mask == u_q).prod(axis=-1).astype('uint8')
                segmentation = mask_to_rle(np.asfortranarray(bin_mask))
                segmentation['counts'] = segmentation['counts'].decode()

                category_id = instance_id_to_category_id[
                    instance_color_to_instance_id[str(tuple(u_q.tolist()))]
                ]

                ##  We need to get also the bounding box annotation 

                # 1. Get the current instance_id given the color
                cur_instance_id = None
                for single_inst in elem_instseg['values']:
                    color = [
                        single_inst['color']['r'], 
                        single_inst['color']['g'], 
                        single_inst['color']['b']
                    ]
                    if (u_q == color).all():
                        cur_instance_id = single_inst['instance_id']
                        break
                if cur_instance_id is None:
                    raise Exception(
                        'Instance color not found. {} {}'
                        .format(u_q, elem_instseg['values'])
                    )

                # 2. Search for the instance_id in the bounding box field and 
                #    get the bounding box values
                x, y, width, height = None, None, None, None
                for elem_bbox in cap['annotations']:
                    if elem_bbox['id'] != 'bounding box':
                        continue

                    for bbox_instances in elem_bbox['values']:
                        if bbox_instances['instance_id'] == cur_instance_id:
                            x, y, width, height = (
                                bbox_instances['x'],
                                bbox_instances['y'],
                                bbox_instances['width'],
                                bbox_instances['height'],
                            ) 
                    if (x, y, width, height) == (None, None, None, None):
                        raise Exception(
                            'No correspondence found between the instance_id '
                            'given by the instance segmentation annotator and '
                            'the one given by the bounding box annotator. {} {}'
                            .format(cur_instance_id, elem_bbox['values'])
                        )

                # For the moment, set every annotation to zero (we cannot 
                # obtain a sequantial number here since this functional
                # is executed in parallel by many processes), we will fix it 
                # later when outside of the multiprocessing
                fake_ann_counter = 0     

                bbox = [x, y, width, height]
                area = int(width * height)
                annotation = create_annotation_format(
                    segmentation, area, image_id, bbox, category_id, 
                    fake_ann_counter
                )

                annotations_list += annotation,

    return annotations_list


def callback_process_capture_file(annotations_list):
    global coco_annotations_field
    coco_annotations_field += annotations_list


def fill_coco_annotations(dataset_folder, coco_json, 
                          instance_id_to_category_id, 
                          instance_color_to_instance_id):

    ## 1. Read mask images and convert coco annotation

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Sorry for the "global" man..  =)
    global coco_annotations_field
    coco_annotations_field = []

    captures_files_list = glob.glob(
        os.path.join(dataset_folder, 'captures_*.json')
    )
    for c_f_l in captures_files_list:
        pool.apply_async(
            process_capture_file, 
            args=(
                c_f_l, dataset_folder, instance_id_to_category_id, 
                instance_color_to_instance_id
            ),
            callback=callback_process_capture_file
        )
    pool.close()
    pool.join()

    ## 2. Assign sequential id to annotations (since process_capture_file 
    ## had assigned zero to each)
    for idx, ann in enumerate(coco_annotations_field):
        ann['id'] = idx

    coco_json['annotations'] = coco_annotations_field
    del coco_annotations_field

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
