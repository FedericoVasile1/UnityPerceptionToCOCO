import argparse
import copy
import os
import random as rndm

import cv2

import numpy as np
from pycocotools.coco import COCO, maskUtils


def main(images_dir, coco_file, num_imgs_to_show, random):
    coco = COCO(coco_file)

    ids = list(coco.imgs.keys())
    if random:
        ids = rndm.sample(ids, num_imgs_to_show)
    for img_infos in coco.loadImgs(ids=ids):
        img = cv2.imread(os.path.join(images_dir, img_infos['file_name']))
        ann_id = coco.getAnnIds(imgIds=img_infos['id'])
        anns = coco.loadAnns(ann_id)

        for a in anns:
            # 1. MASK
            if type(a['segmentation']['counts']) == list:
                rle = maskUtils.frPyObjects(
                    [a['segmentation']], img_infos['height'], img_infos['width']
                )
            else:
                rle = [a['segmentation']]
            bin_mask = maskUtils.decode(rle) 
            # bin_mask.shape (height, width, 1), binary mask of 0s and 1s
            if a['iscrowd'] == 1:
                color_mask = np.array([2.0, 166.0, 101.0]) 
            if a['iscrowd'] == 0:
                color_mask = np.random.randint(0, 256, (3,))
            R = copy.deepcopy(bin_mask)
            G = copy.deepcopy(bin_mask)
            B = copy.deepcopy(bin_mask)
            temp = bin_mask == 1
            R[temp] = color_mask[0]
            G[temp] = color_mask[1]
            B[temp] = color_mask[2]
            mask_overlay = np.concatenate((B, G, R), axis=-1)
            cv2.addWeighted(mask_overlay, 0.7, img, 1.0, 0, img)

            # 2. BOUNDING BOX
            [bbox_x, bbox_y, bbox_w, bbox_h] = a['bbox']
            upper_left = (int(bbox_x), int(bbox_y))
            bottom_right = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))
            cv2.rectangle(
                img, upper_left, bottom_right, 
                tuple(color_mask[::-1].tolist()), 2
            )
            # class label
            cat_name = coco.loadCats(a['category_id'])[0]['name']
            # probability label
            if 'prob' in a:
                prob = ', prob:{:.2f}'.format(a['prob'])
            else:
                prob = ''
            cv2.putText(
                img, cat_name+prob, upper_left, cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, tuple(color_mask[::-1].tolist()), 1, cv2.LINE_4
            )

        cv2.imshow('img', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--coco_file', type=str, required=True)
    parser.add_argument('--num_imgs_to_show', type=int, default=1)
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()

    main(args.images_dir, args.coco_file, args.num_imgs_to_show, args.random)
