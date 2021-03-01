Detectron2 wrapper for DETR
=======

We provide a Detectron2 wrapper for DETR, thus providing a way to better integrate it in the existing detection ecosystem. It can be used for example to easily leverage datasets or backbones provided in Detectron2.

This wrapper currently supports only box detection, and is intended to be as close as possible to the original implementation, and we checked that it indeed match the results. Some notable facts and caveats:
- The data augmentation matches DETR's original data augmentation. This required patching the RandomCrop augmentation from Detectron2, so you'll need a version from the master branch from June 24th 2020 or more recent.
- To match DETR's original backbone initialization, we use the weights of a ResNet50 trained on imagenet using torchvision. This network uses a different pixel mean and std than most of the backbones available in Detectron2 by default, so extra care must be taken when switching to another one. Note that no other torchvision models are available in Detectron2 as of now, though it may change in the future.
- The gradient clipping mode is "full_model", which is not the default in Detectron2.

# Usage

To install Detectron2, please follow the [official installation instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## Evaluating a model

For convenience, we provide a conversion script to convert models trained by the main DETR training loop into the format of this wrapper. To download and convert the main Resnet50 model, simply do:

```
python converter.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_model converted_model.pth
```

You can then evaluate it using:
```
python train_net.py --eval-only --config configs/detr_256_6_6_torchvision.yaml  MODEL.WEIGHTS "converted_model.pth"
```


## Training

To train DETR on a single node with 8 gpus, simply use:
```
python train_net.py --config configs/detr_256_6_6_torchvision.yaml --num-gpus 8
```

To fine-tune DETR for instance segmentation on a single node with 8 gpus, simply use:
```
python train_net.py --config configs/detr_segm_256_6_6_torchvision.yaml --num-gpus 8 MODEL.DETR.FROZEN_WEIGHTS <model_path>
```

## Vertex

Evaluation results are listed in [EVALUATION.MD](EVALUATION.md)

To train DETR follow these steps:
- Pull and convert pre-trained COCO weights:
```
python converter.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_model converted_model.pth
```
- Run the training with 2 gpus:
```
python train_net.py --config=configs/vertex_coco_finetune.yaml \
--num-gpus=2 \
--images_path=/opt/kuna/data/clean_vertexFull100k_S2018/vertexFull100k_S2018/JPEGImages
```

To evaluate a trained model with 2 gpus, define a path to model checkpoint as `MODEL_WEIGHTS` and run:

```
python train_net.py --config=configs/vertex_coco_finetune.yaml \
--num-gpus=2 --images_path=/opt/kuna/data/clean_vertexFull100k_S2018/vertexFull100k_S2018/JPEGImages \
--eval-only MODEL.WEIGHTS $MODEL_WEIGHTS \
DATASETS.TEST '("test_kuna_coco", )' \
DATASETS.PROPOSAL_FILES_TEST '("/opt/kuna/data/clean_vertexFull100k_S2018/vertexFull100k_S2018/vertex_coco/test_coco_dataset.json",)'
```

Example of running inference shown in [inference.ipynb](inference.ipynb)
