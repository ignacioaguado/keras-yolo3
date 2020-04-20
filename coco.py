import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import json

def parse_coco_annotation(coco_file_path, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        with open(coco_file_path, 'r') as coco_file:
            coco_json = json.load(coco_file)

        categories = coco_json['categories']

        tag_id_to_name = {cat['id']: cat['name'] for cat in categories}
        img_dict = {}
        for img in coco_json['images']:
            img_dict[img['id']] = {
                    'filename': os.path.join(img_dir, img['filename']),
                    'width': img['width'],
                    'height': img['height'],
                    'object': []
                }

        for ann in coco_json['annotations']:
            img = img_dict[ann['image_id']]

            obj = {}
            cat_name = tag_id_to_name[ann['category_id']]
            obj['name'] = cat_name

            if ann['category_id'] in seen_labels:
                seen_labels[cat_name] += 1
            else:
                seen_labels[cat_name] = 1

            obj['xmin'] = int(round(float(ann['bbox'][0])))
            obj['ymin'] = int(round(float(ann['bbox'][1])))
            obj['xmax'] = int(round(float(ann['bbox'][2])))
            obj['ymax'] = int(round(float(ann['bbox'][3])))

            img['object'] += [obj]

        all_insts = [img_data for img_data in img_dict.values()]
        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}

        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels