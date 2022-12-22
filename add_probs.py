import argparse
import json
import os

import numpy as np
from pycocotools.coco import COCO


def main(coco_file, num_frames_in_video, frame_threshold, cat_to_delete):
    coco = COCO(coco_file)
    
    new_anns = [] 
    ids = list(coco.imgs.keys())
    ids.sort()
    for idx, img_infos in enumerate(coco.loadImgs(ids=ids)):
        assert idx % num_frames_in_video == img_infos['step']

        ann_id = coco.getAnnIds(imgIds=img_infos['id'])
        anns = coco.loadAnns(ann_id)

        elems_to_remove = []
        for a in anns:
            cat_name = coco.loadCats(a['category_id'])[0]['name']
            if cat_name == cat_to_delete:
                elems_to_remove.append(a)
        for e_t_r in elems_to_remove:
            anns.remove(e_t_r)

        if len(anns) == 0:
            #print('img with no annotations, skipping {} (image id: {})'
            #      .format(img_infos['file_name'], img_infos['id']))
            continue

        if img_infos['step'] == 0:
            # initialize the probabilities
            start_prob = 1 / len(anns)
            video_target_probs = np.linspace(start_prob, 1, frame_threshold)
            video_nontarget_probs = np.linspace(start_prob, 0, frame_threshold)
            num_anns = len(anns)

        if img_infos['step'] < frame_threshold:
            if len(anns) < num_anns:
                # one or more instances are out of the view, redirect their
                # probabilities to the target instance
                diff = num_anns - len(anns)
                video_target_probs[img_infos['step']:] += \
                    video_nontarget_probs[img_infos['step']:] * diff
                num_anns = len(anns)

            for ann in anns:
                if ann['is_video_target']:
                    ann['prob'] = video_target_probs[img_infos['step']]
                else:
                    ann['prob'] = video_nontarget_probs[img_infos['step']]
            
        else:
            for ann in anns:
                if ann['is_video_target']:
                    ann['prob'] = video_target_probs[-1]
                else:
                    ann['prob'] = video_nontarget_probs[-1]
        
        new_anns += anns

    with open(coco_file, 'r') as f:
        coco = json.load(f)
    coco['annotations'] = new_anns
    coco_filename = os.path.basename(coco_file)
    new_coco_filename = coco_filename.split('.')[0] + '_probs.' + \
        coco_filename.split('.')[1]
    path = coco_file.replace(coco_filename, new_coco_filename)
    with open(path, 'w') as f:
        json.dump(coco, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', type=str, required=True)
    parser.add_argument('--num_frames_in_video', type=int, default=90)
    parser.add_argument('--frame_threshold', type=int, default=60)
    parser.add_argument('--cat_to_remove', type=str, default='object_part')
    args = parser.parse_args()

    main(args.coco_file, args.num_frames_in_video, args.frame_threshold, 
         args.cat_to_remove)
