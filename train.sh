#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_dir dataset/train \
    --val_dir dataset/valid \
    --save_dir runs/phasr
