#!/usr/bin/env bash
set -e

eval "$(conda shell.bash hook)"

conda activate dinov3
python segment_images_batch_v2.py
