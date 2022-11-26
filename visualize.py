import argparse
import os
import random

import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def main(images_dir, coco_file, num_imgs_to_show, show_bbox, show_text_labels):
    coco = COCO(coco_file)
    
    rand_ids = list(coco.imgs.keys())
    rand_ids = random.sample(rand_ids, num_imgs_to_show)
    for img_infos in coco.loadImgs(ids=rand_ids):
        img = plt.imread(os.path.join(images_dir, img_infos['file_name']))

        _, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)

        ann_id = coco.getAnnIds(imgIds=img_infos['id'])
        anns = coco.loadAnns(ann_id)
        coco.showAnns(anns, draw_bbox=show_bbox)
        if show_text_labels:
            for a in anns:
                cat_name = coco.loadCats(a['category_id'])[0]['name']
                ax.text(a['bbox'][0], a['bbox'][1], cat_name, fontsize='x-small',
                        bbox={'facecolor': 'none', 'pad': 0})
            
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--coco_file', type=str, required=True)
    parser.add_argument('--num_imgs_to_show', type=int, default=1)
    parser.add_argument('--show_bbox', action='store_true')
    parser.add_argument('--show_text_labels', action='store_true')
    args = parser.parse_args()

    main(args.images_dir, args.coco_file, args.num_imgs_to_show, 
         args.show_bbox, args.show_text_labels)
