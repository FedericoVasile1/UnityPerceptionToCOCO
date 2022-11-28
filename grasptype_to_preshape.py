'''
THIS IS JUST A SCRIPT FOR MY OWN LABELS, TO USE AFTER RUNNING convert.py
'''
import argparse
import os
import json


if __name__ == '__main__':
    grasp_type_to_preshape = {
        'large_diameter': 'power_no3',
        'sphere_4fingers': 'power_no3',
        'tripod': 'pinch_3',
        'medium_wrap': 'power_no3',
        'prismatic_4fingers': 'pinch_no3',
        'power_sphere': 'power_no3',
        'adducted_thumb': 'lateral_no3',
        'prismatic_2fingers': 'pinch_3',
        'power_disk': 'power_no3',
        'small_diameter': 'power_no3',
        'object_part': 'object_part'
    }

    preshape_to_id = {
        'power_no3': 0,
        'pinch_3': 1,
        'lateral_no3': 2,
        'pinch_no3': 3,
        'object_part': 4
    }

    grasp_type_id_to_preshape_id = {}

    grasp_type_id_to_grasp_type = {}

    category_to_remove = 'object_part'

    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', required=True)
    args = parser.parse_args()

    with open(args.coco_file, 'r') as f:
        coco = json.load(f)

    new_cats = []
    cats = coco['categories']
    for c in cats:  
        grasp_type_id_to_grasp_type[c['id']] = c['name']

        grasp_type_id_to_preshape_id[c['id']] = \
            preshape_to_id[grasp_type_to_preshape[c['name']]]
        
        flag = False
        for elem in new_cats:
            if elem['name'] == grasp_type_to_preshape[c['name']]:
                flag = True
                break
        if not flag:
            new_c = {
                'supercategory': '',
                'id': preshape_to_id[grasp_type_to_preshape[c['name']]],
                'name': grasp_type_to_preshape[c['name']]
            }
            new_cats.append(new_c)

    idsx_to_del = []
    for i, n_c in enumerate(new_cats):
        if n_c['name'] in category_to_remove:
            idsx_to_del.append(i)
    for i_t_d in idsx_to_del:
        del new_cats[i_t_d]

    coco['categories'] = new_cats

    new_anns = []
    for ann in coco['annotations']:
        if grasp_type_id_to_grasp_type[ann['category_id']] == category_to_remove:
            continue

        ann['category_id'] = grasp_type_id_to_preshape_id[ann['category_id']]
        new_anns.append(ann)

    coco['annotations'] = new_anns

    json_filename = os.path.basename(args.coco_file)
    new_json_filename = args.coco_file.replace(
        json_filename, 'preshape_'+json_filename
    )
    with open(new_json_filename, 'w') as outfile:
        json.dump(coco, outfile)
