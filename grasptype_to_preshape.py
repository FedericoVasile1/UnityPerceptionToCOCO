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

    CATEGORIES_TO_REMOVE = ['object_part']

    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', required=True)
    args = parser.parse_args()

    with open(args.coco_file, 'r') as f:
        coco = json.load(f)

    new_cats = []
    cats = coco['categories']
    for c in cats:  
        grasp_type = c['name']
        grasp_type_id_to_grasp_type[c['id']] = grasp_type

        grasp_type_no_obj_side = grasp_type.split('[')[0]
        grasp_type_id_to_preshape_id[c['id']] = \
            preshape_to_id[grasp_type_to_preshape[grasp_type_no_obj_side]]
        
        flag = False
        for elem in new_cats:
            if elem['name'] == grasp_type_to_preshape[grasp_type_no_obj_side]:
                flag = True
                break
        if not flag:
            new_c = {
                'supercategory': '',
                'id': preshape_to_id[grasp_type_to_preshape[grasp_type_no_obj_side]],
                'name': grasp_type_to_preshape[grasp_type_no_obj_side]
            }
            new_cats.append(new_c)

    idsx_to_del = []
    for i, n_c in enumerate(new_cats):
        if n_c['name'] in CATEGORIES_TO_REMOVE:
            idsx_to_del.append(i)
    new_cats = [v for idx, v in enumerate(new_cats) if idx not in idsx_to_del]

    coco['categories'] = new_cats

    new_anns = []
    for ann in coco['annotations']:
        if grasp_type_id_to_grasp_type[ann['category_id']] in CATEGORIES_TO_REMOVE:
            continue

        ann['category_id'] = grasp_type_id_to_preshape_id[ann['category_id']]
        new_anns.append(ann)

    coco['annotations'] = new_anns

    json_filename = os.path.basename(args.coco_file)
    if 'grasptype' in json_filename:
        new_json_filename = json_filename.replace('grasptype', 'preshape')
    else:
        raise Exception('Expected to find \'grasptype\' in the json file '
                        'name (e.g., coco_train_grasptype.json), '
                        'but not found: {}'.format(json_filename))
    full_path = args.coco_file.replace(
        json_filename, new_json_filename
    )
    with open(full_path, 'w') as outfile:
        json.dump(coco, outfile)
