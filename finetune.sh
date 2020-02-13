#!/bin/bash

source /home/ycgu/anaconda3/bin/activate pt10py36

GPU_IDS=0,1,2,3
NUM_THREADS=2
CUDA_VISIBLE_DEVICES=${GPU_IDS} OMP_NUM_THREADS=${NUM_THREADS} python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 finetune_temporal_distribute.py
