#!/bin/bash

# ==============================================================================
# lm-evaluation-harness 离线评估脚本
# ==============================================================================

# --- 1. 设置环境变量 ---
# 强制 Hugging Face libraries (transformers, datasets) 在离线模式下运行，
# 并指定本地数据集的缓存路径。
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# 注意: HF_HOME 和 HF_DATASETS_CACHE 都指向您的数据集根目录
# export HF_HOME="/gemini-1/platform/public/huwt/evaldata3"
# export HF_DATASETS_CACHE="/gemini-1/platform/public/huwt/evaldata3"
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

# 设置代理
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

echo "环境变量设置完毕:"
echo "HF_DATASETS_OFFLINE: $HF_DATASETS_OFFLINE"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "----------------------------------------------------------------"

# --- 2. 配置评估参数 ---

# 指定您的本地模型路径
MODEL_PATH="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/models/Qwen3-30B-A3B"

# 设置 layer_k.json 和 cei_scores.json 文件路径
# LAYER_K="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/lm-evaluation-harness/myconfig/expert_allocation12.json"
LAYER_K="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/lm-evaluation-harness/myconfig/expert_allocation_budget_8_10.json"
CEI_SCORES="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/lm-evaluation-harness/myconfig/expert_cei_scores.json"

# 指定要评估的任务列表，用逗号分隔。
# 这里的任务名应与您的数据集文件夹名保持一致。
# 注意：对于 mmlu，您的文件夹名为 mmlu_all_subjects，因此任务名也应使用它。
# 如果 lm-eval-harness 无法识别此任务名，可能需要创建自定义任务配置。

# TASK_LIST="truthfulqa"
# TASK_LIST="truthfulqa,triviaqa,gsm8k,mmlu,ai2_arc"
TASK_LIST="factor_expert,factor_news,factor_wiki"


# 指定评估使用的设备 (例如: cuda:0, cpu)
DEVICE="cuda:0"

# 批处理大小。'auto' 会让框架自动寻找最优值。如果遇到显存不足(OOM)，请手动设置为一个较小的值，如 1 或 2。
BATCH_SIZE=8

# 评估结果输出目录  
OUTPUT_PATH="./eval_results"

echo "评估参数配置:"
echo "模型路径: $MODEL_PATH"
echo "专家配置: $EXPERT_CONFIG"
echo "任务列表: $TASK_LIST"
echo "设备: $DEVICE"
echo "批处理大小: $BATCH_SIZE"
echo "输出路径: $OUTPUT_PATH"
echo "----------------------------------------------------------------"



# --- 3. 执行评估命令 ---
echo "开始执行 lm-evaluation-harness 评估..."
lm_eval --model custom_our \
    --model_args pretrained=${MODEL_PATH},layer_k_config_file=${LAYER_K},cei_scores_file=${CEI_SCORES},extra_mass=0.1,cei_temp=5.0,debug_hdaav=True,debug_print_layers=20,debug_max_calls_per_layer=2,trust_remote_code=True \
    --tasks ${TASK_LIST} \
    --device ${DEVICE} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} 

echo "----------------------------------------------------------------"
echo "评估完成！结果已保存至 ${OUTPUT_PATH} 目录。"