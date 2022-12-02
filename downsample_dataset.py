'''
CUSTOM SCRIPT FOR MY DATASET
'''


import json
import os
import argparse

import numpy as np


def main(coco_filename):    
    with open(coco_filename, 'r') as f:
        coco_file = json.load(f)

    NUM_FRAMES_IN_VIDEO = 90

    NUM_FRAMES_TO_EXTRACT_FROM_VIDEO = 6
    STEP_TO_EXTRACT_FRAMES = 10

    FIRST_FRAME_RANGE = 0
    
    start_frame = None
    ids_to_consider = []
    new_img_metadatas = []
    for idx, img_metadata in enumerate(coco_file['images']):
        if idx % NUM_FRAMES_IN_VIDEO == 0:
            start_frame = np.random.randint(
                FIRST_FRAME_RANGE, STEP_TO_EXTRACT_FRAMES
            )
            idxs = [start_frame+STEP_TO_EXTRACT_FRAMES*i 
                    for i in range(NUM_FRAMES_TO_EXTRACT_FROM_VIDEO)]
        
        if idx % NUM_FRAMES_IN_VIDEO not in idxs:
            continue

        new_img_metadatas.append(img_metadata)
        ids_to_consider.append(img_metadata['id'])

    coco_file['images'] = new_img_metadatas

    new_ann_metadatas = []
    for ann_metadata in coco_file['annotations']:
        if ann_metadata['image_id'] not in ids_to_consider:
            continue

        new_ann_metadatas.append(ann_metadata)

    coco_file['annotations'] = new_ann_metadatas

    json_filename = os.path.basename(coco_filename)
    new_json_filename = os.path.splitext(json_filename)[0] + \
        '_downsampled' + os.path.splitext(json_filename)[1]
    full_path = coco_filename.replace(json_filename, new_json_filename)
    with open(full_path, 'w') as f:
        json.dump(coco_file, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', type=str, required=True)
    args = parser.parse_args()

    main(args.coco_file)
