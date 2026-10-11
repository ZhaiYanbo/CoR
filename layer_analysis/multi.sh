#!/bin/bash

# 单卡运行 - JSONL 格式 C4 数据集

# MODEL="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/models/Qwen3-30B-A3B"
# HARD_TOKENS="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/hard_samples_all.jsonl"  # 您的 JSONL 格式困难样本文件
# C4_JSONL="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/c4-train.00000-of-01024.jsonl"  # 修改为你的本地路径


# # 或者 8 卡并行
# torchrun --nproc_per_node=8 run_layer_sensitivity.py \
#     --model $MODEL \
#     --hard_tokens $HARD_TOKENS \
#     --dataset_path $C4_JSONL \
#     --dataset_text_field text \
#     --use_distributed \
#     --generate_topk \
#     --output_dir ./sensitivity_results_8gpu \
#     --batch_size 8

MODEL="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/models/Qwen3-30B-A3B"
HARD_TOKENS="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/hard_samples_all.jsonl"  # 您的 JSONL 格式困难样本文件
EASY_TOKENS="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/easy_samples_all.jsonl"
C4_JSONL="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/c4-train.00000-of-01024.jsonl"  # 修改为你的本地路径

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

torchrun \
  --nproc_per_node=${GPUS_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  run_layer_sensitivity_sampled.py \
    --model "$MODEL" \
    --hard_tokens "$HARD_TOKENS" \
    --easy_tokens "$EASY_TOKENS" \
    --dataset_path "$C4_JSONL" \
    --comprehensive \
    --max_samples 1000 \
    --use_distributed \
    --output_dir ./sensitivity_results_1k_hard_easy_comprehensive
