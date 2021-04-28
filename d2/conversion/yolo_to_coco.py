"""Script for converting annotations from YOLO to COCO format.

Usage: python -m conversion.yolo_to_coco conversion/paramaters.yaml.
Outputs: data splits in COCO json format.
"""
import json
import pathlib
import os
import logging

from typing import Tuple, List, Set

import fire
import tqdm
import yaml

import numpy as np
import pandas as pd


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def create_image_annotation(file_name, width, height, image_id):
    images = {"file_name": file_name, "height": height, "width": width, "id": image_id}
    return images


def create_annotation_coco_format(
    category_id, min_x, min_y, width, height, image_id, annotation_id
):
    bbox = (min_x, min_y, width, height)
    area = width * height

    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        "segmentation": [],
    }

    return annotation


def convert_annotation(
    bboxes: np.ndarray, size: Tuple[int, int], cat_mapping: dict
) -> List[Tuple[int, int, int, int, int]]:
    img_width, img_height = size

    def convert_bbox(bbox: np.ndarray):
        ann_class, x, y, width, height = bbox
        ann_class = cat_mapping[int(ann_class)]
        width = int(width * img_width)
        height = int(height * img_height)
        x = int(x * img_width) - width // 2
        y = int(y * img_height) - height // 2
        return ann_class, x, y, width, height

    return [convert_bbox(box) for box in bboxes]


def get_images_list(base_path, filepath) -> Set[str]:
    """Returns list of images identifiers -> camera_id/frame_id."""
    with open(os.path.join(base_path, filepath)) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    images_list = set()

    for image_path in content:
        filename = os.path.join(
            os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path)
        )
        images_list.add(filename)

    return images_list


def load_config(path: str) -> dict:
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def convert(config_path: str):
    config = load_config(config_path)
    annotation_idx = 0
    image_idx = 0

    base_path = config["BASE_PATH"]
    labels_path = os.path.join(base_path, config["ANNOTATIONS_PATH"])

    image_width, image_height = config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"]
    cat_mapping = config["MAPPING"]
    categories = config["CATEGORIES"]
    splits = config["DATA_SPLITS"]

    for split in tqdm.tqdm(splits):
        image_annotations = []
        coco_annotations = []
        image_list = get_images_list(base_path, splits[split])

        for idx, image_identifier in tqdm.tqdm(enumerate(image_list), leave=False):
            image_path = os.path.join(
                base_path, config["IMAGES_PATH"], image_identifier
            )
            if not os.path.isfile(image_path):
                logging.warning("{} not found".format(image_path))
                continue

            image_idx += 1
            image_annotations.append(
                create_image_annotation(
                    image_identifier, image_width, image_height, idx
                )
            )

            label_path = os.path.join(
                labels_path,
                os.path.basename(os.path.dirname(image_identifier)),
                pathlib.Path(image_identifier).with_suffix(".txt").name,
            )

            if is_non_zero_file(label_path):
                annotation = pd.read_csv(label_path, sep=" ", header=None).values
                annotations = convert_annotation(
                    annotation, (image_width, image_height), cat_mapping
                )
                for annotation in annotations:
                    coco_ann = create_annotation_coco_format(
                        *annotation, idx, annotation_idx
                    )
                    annotation_idx += 1
                    coco_annotations.append(coco_ann)

        coco_dataset = {
            "images": image_annotations,
            "annotations": coco_annotations,
            "categories": categories,
        }

        with open("{}_coco_dataset.json".format(split.lower()), "w") as outfile:
            json.dump(coco_dataset, outfile)


if __name__ == "__main__":
    fire.Fire(convert)
