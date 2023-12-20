import json
import os
from PIL import Image

import sys
sys.path.append('/home/airlab-jmw/pyws/Exploration-of-Potential')

from WoodScape.scripts.parsers.detection.helpers import generate_2d_boxes
from WoodScape.scripts.parsers.detection.annotation_detection_parser import AnnotationDetectionParser
from WoodScape.scripts.parsers.detection.filter_params import FilterParams

def convert_woodscape_to_coco(woodscape_dir, class_mapping, img_dir, output_file):
    # Prepare COCO data structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Load class mapping 
    with open(class_mapping, 'r') as f:
        class_mapping = json.load(f)
    
    classes_to_extract = [
        "vehicles",
        "person",
        "bicycle",
        "traffic_light",
        "traffic_sign"
    ]
    
    
    box_2d_info = json.loads(open('WoodScape/scripts/configs/box_2d_mapping_5_classes.json').read())
    class_names = box_2d_info["classes_to_extract"]
    class_colors = box_2d_info["class_colors"]
    class_ids = box_2d_info["class_indexes"]
    class_obj_thresh = 400
    depiction_ann = box_2d_info["filter_depict_objects"]
    glass_cover_ann = box_2d_info["filter_glass_cover_objects"]
    occluded_ann = box_2d_info["filter_occluded_objects"]
    debug = box_2d_info["debug"]
    filter_ann_params = FilterParams(class_names=class_names)
    # filter_ann_params = None
    # Iterate over Woodscape files
    anno_id = 0
    img_id = 0
    for filename in sorted(os.listdir(woodscape_dir)):
        print(filename)
        if filename.endswith('.json'):
            with open(os.path.join(woodscape_dir, filename), 'r') as f:
                woodscape_data = json.load(f)
                # Extract key (image file name)
                image_filename = list(woodscape_data.keys())[0]
                # Access annotation data
                woodscape_annotation = woodscape_data[image_filename]['annotation']

                modified_woodscape_annotation = []
                for anno in woodscape_annotation:
                    if 'tags' in anno and anno['tags'][0] in class_mapping:
                        anno['tags'][0] = class_mapping[anno['tags'][0]]
                        if anno['tags'][0] not in classes_to_extract:
                            continue
                        else:
                            modified_woodscape_annotation.append(anno)

                # Define categories
                # categories = [class_mapping[anno['tags'][0]] for anno in woodscape_annotation]
                for category in classes_to_extract:
                    if category not in [c['name'] for c in coco_data["categories"]]:
                        coco_data["categories"].append({
                            "id": len(coco_data["categories"]) + 1, 
                            "name": category
                        })
                categories_dict = {c['name']: c['id'] for c in coco_data["categories"]}

                # Add image info
                image_path = os.path.join(img_dir, image_filename[:5] +'.png')
                with Image.open(image_path) as img:
                    width, height = img.size
                coco_data["images"].append({
                    "file_name": image_filename[:5] + '.png',
                    "id": img_id,
                    "width": width,
                    "height": height
                })

                # Read corresponding bbox file
                image = Image.open(image_path)
                json_ann_name = os.path.join(woodscape_dir, filename)
                parser = AnnotationDetectionParser(json_ann_name, (width, height), filter_ann_params)
                boxes = generate_2d_boxes(parser, class_names, class_colors, class_ids, class_obj_thresh,
                                                        image_rgb=image, debug=False)

                
                # Read corresponding bbox file
                # with open(os.path.join(bbox_dir, image_filename.replace('.json', '.txt')), 'r') as f:
                #     bbox_data = f.readlines()

                # Convert annotations
                for anno, box in zip(modified_woodscape_annotation, boxes):
                    segmentation = [coord for point in anno['segmentation'] for coord in point]
                    bbox = [box[1], box[2], box[3] - box[1], box[4] - box[2]]
                    area = bbox[2] * bbox[3]
                    if area < class_obj_thresh:
                        continue
                    
                    coco_data["annotations"].append({
                        "id": anno_id,
                        "image_id": img_id,
                        "category_id": categories_dict[anno['tags'][0]],
                        "segmentation": [segmentation],
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    anno_id += 1

                img_id += 1
   
    # Save COCO format data
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)


# Usage:
if __name__ == '__main__':
    woodscape_dir = '/media/airlab-jmw/DATA/Dataset/wood_valid/json'  # Input directory containing Woodscape json files
    # bbox_dir = '/media/airlab-jmw/DATA/Dataset/test2'  # Input directory containing bbox txt files
    output_file = 'instances_val2017.json'  # Output file
    img_dir = '/media/airlab-jmw/DATA/Dataset/wood_valid/images' 
    class_mapping = 'WoodScape/scripts/mappers/class_names.json'
    class_obj_thresh = 400
    
    convert_woodscape_to_coco(woodscape_dir, class_mapping, img_dir, output_file)