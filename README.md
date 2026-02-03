# SCABR: Semantic Correlation-Aware Background Removal for Few-Shot Learning

## Dependencies
* python 3.9.21
* torch 2.7.0
* sklearn 1.6.1, pillow11.2.1, numpy2.0.2
* GPU (RTX3090) + CUDA11.0 CuDNN
  
## Overview
We propose a framework termed Semantic Correlation-Aware Background Removal (SCABR) for few-shot image classification to alleviate distribution bias
caused by background variations. The entire process is illustrated in below.
The novel dataset is processed via saliency detection to separate corresponding
foreground and background images. In order to group images by class labels,
we generate pseudo-labels for query samples via K-means clustering. It is followed by computing background dependency scores. Specifically, we quantify
a class’s dependency on its background by computing the feature similarity of
background regions across intra-class samples. Then the selective background
removal based on the ranking of scores is performed. The ratio parameter α is
set to 0.2. Finally, we leverage a large language model to generate descriptions
as semantic prompts for adaptively tuning the visual feature extraction, and extract representative features through the action of the background suppression
module.
<img width="1260" height="678" alt="绘图_改进" src="https://github.com/user-attachments/assets/9af3e71d-a61b-41b3-8713-9af34cb7d25c" />

## Datasets
The dataset can be downloaded from the following links:
* [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CUB](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)

## Preparation Before Running
Place the foreground_images and background_images in the `filelist` directory. The acquisition method can be referenced from the  [VST]([https://github.com/nnizhang/VST]() implementation.

Ensure that datasets are located in the `filelist` directory. 

#### Dataset Structure:
```
--SCABR
    |--filelist
        |--miniImageNet
            |--train
            |--train_fore
            |--train_back
            |--val
            |--val_fore
            |--val_back
            |--test
            |--test_fore
            |--test_back

```
## Process Images
python cluster.py --n_cluster 50 --input_folder input_folder_path --output_root output_root_path

python multiply_files.py

python image_similarity_select.py --root_folder root_folder_path --folder1 folder1_path --folder2 folder2_path --folder5 folder5_path

## Evaluate SCABR method
python train_vit.py --dataset miniImageNet --exp pre-train --rand_aug --repeat_aug --epochs 800

python train_vit_sp.py   --dataset miniImageNet  --exp sp   --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth

python train_vit_sp.py   --dataset miniImageNet --test_classifier fc  --exp test --test --episodes 2000 --resume checkpoint/miniImageNet/visformer-t/sp/checkpoint_epoch_best.pth 

python train_vit_sp.py  --dataset miniImageNet   --exp sp_5shot --shot 5 --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth

python train_vit_sp.py  --dataset miniImageNet --aug_support 10  --exp test --shot 5   --test --episodes 2000 --resume checkpoint/miniImageNet/visformer-t/sp_5shot/checkpoint_epoch_best.pth

