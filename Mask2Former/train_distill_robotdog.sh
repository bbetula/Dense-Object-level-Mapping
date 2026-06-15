#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

cd /data1/user/Dense-Object-level-Mapping/Mask2Former

/data1/user/miniconda3/envs/dinov3/bin/python train_net.py \
    --num-gpus 1 \
    --resume \
    --config-file configs/robotdog/maskformer2_dinov3_vitl_robotdog_ade20k.yaml \
    OUTPUT_DIR ./output/dinov3_vitl_robotdog_ade20k_distill
