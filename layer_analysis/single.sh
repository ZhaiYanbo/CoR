#!/bin/bash

# 单卡运行 - JSONL 格式 C4 数据集

MODEL="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/models/Qwen3-30B-A3B"
HARD_TOKENS="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/hard_samples_all.jsonl"  # 您的 JSONL 格式困难样本文件
EASY_TOKENS="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/easy_samples_all.jsonl"
C4_JSONL="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/c4-train.00000-of-01024.jsonl"  # 修改为你的本地路径
 
# 单卡运行
# python run_layer_sensitivity.py \
#     --model $MODEL \
#     --hard_tokens $HARD_TOKENS \
#     --dataset_path $C4_JSONL \
#     --dataset_text_field text \
#     --perturbation_scale 0.01 \
#     --generate_topk \
#     --min_topk 2 \
#     --max_topk 8 \
#     --output_dir ./sensitivity_results

# 单卡采样10%
# python run_layer_sensitivity_sampled.py \
#     --model $MODEL \
#     --hard_tokens $HARD_TOKENS \
#     --dataset_path $C4_JSONL \
#     --dataset_text_field text \
#     --sample_ratio 0.1 \
#     --perturbation_scale 0.01 \
#     --generate_topk \
#     --min_topk 2 \
#     --max_topk 8 \
#     --output_dir ./sensitivity_results_10pct

# 用1000个样本去测
# 只处理 1000 个样本
# python run_layer_sensitivity_sampled.py \
#     --model $MODEL \
#     --hard_tokens $HARD_TOKENS \
#     --dataset_path $C4_JSONL \
#     --max_samples 1000 \
#     --generate_topk \
#     --output_dir ./sensitivity_results_1k

# 用1000个样本去测
# 只处理 1000 个样本
# 难样本简单样本都处理
python run_layer_sensitivity_sampled.py \
    --model $MODEL \
    --hard_tokens $HARD_TOKENS \
    --easy_tokens $EASY_TOKENS \
    --dataset_path $C4_JSONL \
    --comprehensive \
    --max_samples 1000 \
    --generate_topk \
    --min_topk 8 \
    --max_topk 16 \
    --output_dir ./sensitivity_results_1k_hard_easy_comprehensive
