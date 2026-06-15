#!/bin/bash

export DETECTRON2_DATASETS=/data1/data
export CUDA_VISIBLE_DEVICES=2

cd /data1/user/Dense-Object-level-Mapping/Mask2Former

/data1/user/miniconda3/envs/dinov3/bin/python train_net.py \
    --num-gpus 1 \
    --resume \
    --config-file configs/cityscapes/semantic-segmentation/maskformer2_dinov3_vitl_bs16_90k.yaml \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 0.0001 \
    OUTPUT_DIR ./output/dinov3_vitl_cityscapes_m2f
