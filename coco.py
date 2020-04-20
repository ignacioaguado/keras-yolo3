import numpy as np
import os
from collections import deque
import pickle
import json
from random import shuffle

def parse_coco_annotation(coco_file_path, img_dir, cache_name, labels=[], split_len=None, keep_original_dist=False):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, final_seen_labels, split_len = cache['all_insts'], cache['seen_labels'], cache['split_len']
    else:
        all_insts = []
        initial_seen_labels = {}
        final_seen_labels = {}

        with open(coco_file_path, 'r') as coco_file:
            coco_json = json.load(coco_file)

        categories = coco_json['categories']

        tag_id_to_code = {cat['id']: cat['name'] for cat in categories}
        img_dict = {}
        for img in coco_json['images']:
            img_dict[img['id']] = {
                    'filename': os.path.join(img_dir, img['file_name']),
                    'width': img['width'],
                    'height': img['height'],
                    'object': []
                }

        annot_by_tag_code = {tag_code: [] for tag_code in tag_id_to_code.values()}

        for ann in coco_json['annotations']:
            obj = {}
            obj['image_id'] = ann['image_id']
            cat_name = tag_id_to_code[ann['category_id']]
            obj['name'] = cat_name


            if cat_name in initial_seen_labels:
                initial_seen_labels[cat_name] += 1
            else:
                initial_seen_labels[cat_name] = 1
                final_seen_labels[cat_name] = 0

            obj['xmin'] = int(round(float(ann['bbox'][0])))
            obj['ymin'] = int(round(float(ann['bbox'][1])))
            obj['xmax'] = int(round(float(ann['bbox'][2])))
            obj['ymax'] = int(round(float(ann['bbox'][3])))
            
            annot_by_tag_code[tag_id_to_code[ann['category_id']]].append(obj)

        total_annotations = sum(list(initial_seen_labels.values()))

        split_len = split_len or total_annotations
        
        # Adapted from Tagging abc_data_generators.py
        deque_dict = {}
        for tag_code, tag_annotations in annot_by_tag_code.items():
            shuffle(tag_annotations)
            real_batch_size = min(total_annotations, split_len) if split_len else total_annotations
            deque_dict[tag_code] = {
                'queue': deque(tag_annotations),
                'ratio': int(np.ceil(len(tag_annotations) / total_annotations * real_batch_size)) if keep_original_dist else 1
            }

        continue_looping = True
        annotations_to_use = []

        while continue_looping:
            for tag_code, queue_info in deque_dict.items():
                if queue_info['queue']:
                    for i in range(queue_info['ratio']):
                        annotations_to_use.append(queue_info['queue'].popleft())
                        final_seen_labels[tag_code] += 1

                        if not queue_info['queue']:
                            break

                if len(annotations_to_use) >= split_len \
                        or not any([queue_info['queue'] for tag_code, queue_info in deque_dict.items()]):
                    continue_looping = False
                    break

        img_ids_to_use = []
        for ann in annotations_to_use:
            img_dict[ann['image_id']]['object'].append(ann)
            img_ids_to_use.append(ann['image_id'])

        all_insts = []
        for img_id in img_ids_to_use:
            all_insts.append(img_dict[img_id])

        cache = {'all_insts': all_insts, 'seen_labels': final_seen_labels, 'split_len': split_len}

        print(f'Total images: {len(all_insts)}')
        print(f'Total annotations: {len(annotations_to_use)}')
        print(f'Tag distribution: ')
        for tag_code, items in final_seen_labels.items():
            print(f'{tag_code}: {items}')

        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, final_seen_labels