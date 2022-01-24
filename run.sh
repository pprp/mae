#!/bin/bash
OMP_NUM_THREADS=1

# module load anaconda 
# source activate torch17 
# export PYTHONUNBUFFERED=1

IMAGENET_DIR="/home/pdluser/dataset/imagenet-mini/ILSVRC2012"

LOG_DIR="./logs/pretrained"

# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node 2 main_pretrain.py \
#     --batch_size 64 \
#     --model mae_vit_base_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 100 \
#     --warmup_epochs 10 \
#     --blr 1.5e-4 \
#     --weight_decay 0.05 \
#     --data_path ${IMAGENET_DIR} \
#     --output_dir ${LOG_DIR} \
#     --num_workers 20 

CUDA_VISIBLE_DEVICES=0 python -W ignore -m torch.distributed.launch --nproc_per_node 1 main_pretrain.py \
    --batch_size 32 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${LOG_DIR} \
    --num_workers 20 