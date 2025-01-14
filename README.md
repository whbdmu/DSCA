# Domain Adaptive Person Search

## Introduction

This is the official implementation for our paper Unsupervised Domain Adaptive Person Search via Dual Self-Calibration (DSCA) in AAAI2025. The code is based on the official code of [DAPS](https://github.com/caposerenity/DAPS) and [SPCL](https://github.com/yxgeee/SpCL).

Performance :
We retrained the model and got better ReID performance reported in our paper.

|  Source   |  Target   | mAP  | Top-1 |                             CKPT                             |
| :-------: | :-------: | :--: | :---: | :----------------------------------------------------------: |
|    PRW    | CUHK-SYSU | 80.4 | 82.2  | [ckpt](https://drive.google.com/file/d/1JjZKcbcqDeirjhGjvhJ-TIoN0ciUt1AD/view?usp=sharing) |
| CUHK-SYSU |    PRW    | 40.2 | 81.8  | [ckpt](https://drive.google.com/file/d/1hhx3LWthEikiN4swn2q_-Gg_58v-91Gu/view?usp=sharing) |

![framework](doc/framework.png)

## Installation

run `python setup.py develop` to enable SPCL

Install Nvidia [Apex](https://github.com/NVIDIA/apex)

Run `pip install -r requirements.txt` in the root directory of the project.


## Data Preparation

1. Download [CUHK-SYSU](https://drive.google.com/open?id=1z3LsFrJTUeEX3-XjSEJMOBrslxD2T5af) and [PRW](https://goo.gl/2SNesA) datasets, and unzip them.
2. Modify `configs/prw_da.yaml` and `configs/cuhk_sysu_da.yaml` to change the dataset store place (Line 1,5,6) to your own path.

## Testing

1. Following the link in the above table, download our pretrained model to anywhere you like

2. Evaluate its performance by specifing the paths of checkpoint and corresponding configuration file.

PRW as the target domain:

```
python train.py --cfg configs/cuhk_sysu_da.yaml --eval --ckpt $MODEL_PATH
```

CUHK-SYSU as the target domain:

```
python train.py --cfg configs/prw_da.yaml --eval --ckpt $MODEL_PATH
```

## Training

PRW as the target domain:

```
python train.py --cfg configs/cuhk_sysu_da.yaml
```

CUHK-SYSU as the target domain:

```
python train.py --cfg configs/prw_da.yaml
```
