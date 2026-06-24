#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --weights checkpoints/model_best.pth \
    --data dataset/test \
    --save_dir output
