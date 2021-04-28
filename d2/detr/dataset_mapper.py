# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch

import albumentations as albu

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BoxMode

__all__ = ["DetrDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def build_augmentations(border_mode=0):
    return albu.Compose([
        albu.Rotate(limit=10, p=0.5),
        albu.RandomScale(scale_limit=0.1, p=0.5),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            albu.RandomGamma(),
        ], p=0.5),
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.4),
        albu.OneOf(
            [
                albu.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=.4),
                albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=.4)
            ],
            p=.2
        ),
        albu.OneOf([
            albu.GaussNoise(),
            albu.ISONoise(intensity=(0.1, 0.3), p=0.6)
        ], p=0.5),
        albu.OneOf([
            albu.RandomShadow(),
            albu.RandomSunFlare(src_radius=200),
            albu.RandomSnow(),
            albu.RandomRain(),
            albu.RandomFog(fog_coef_upper=0.5)
        ], p=0.1)
    ], p=1,
        bbox_params=albu.BboxParams(format='coco', label_fields=['category_id'], min_visibility=0.25))


class DetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        self.aug = build_augmentations()
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            boxes = [ann['bbox'] for ann in dataset_dict['annotations']]
            labels = [ann['category_id'] for ann in dataset_dict['annotations']]
            boxes = np.array(boxes, dtype=np.float32)

            if len(boxes) > 0:
                h, w, _ = image.shape
                boxes[:, :] = boxes[:, :].clip(min=[0, 0, 0, 0], max=[w, h, w, h])

            augm_annotation = self.aug(image=image, bboxes=boxes, category_id=labels)

            image = augm_annotation['image']
            h, w, _ = image.shape

            augm_boxes = np.array(augm_annotation['bboxes'], dtype=np.float32)
            # sometimes bbox annotations go beyond image
            if len(augm_boxes) > 0:
                augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0, 0, 0, 0], max=[w, h, w, h])
            augm_labels = np.array(augm_annotation['category_id'])
            dataset_dict['annotations'] = [
                {
                    'iscrowd': 0,
                    'bbox': augm_boxes[i].tolist(),
                    'category_id': augm_labels[i],
                    'bbox_mode': BoxMode.XYWH_ABS,
                }
                for i in range(len(augm_boxes))
            ]

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
