#!/bin/bash

# First, allocate resources (run this once from login node):
# salloc -N 1 --gres=gpu:8 --mem=500G -c 64 -t 48:00:00

# Then run training with:
srun --gres=gpu:8 python train.py
