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
export EXPERT_CONFIG="/mlx/users/zhaiyanbo.188/hu/config/expert_all8.json"
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

# 设置代理
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

echo "环境变量设置完毕:"
echo "HF_DATASETS_OFFLINE: $HF_DATASETS_OFFLINE"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "EXPERT_CONFIG: $EXPERT_CONFIG"
echo "----------------------------------------------------------------"

# --- 2. 配置评估参数 ---

# 指定您的本地模型路径
MODEL_PATH="/mlx/users/zhaiyanbo.188/Qwen3-30B-A3B"

# 指定要评估的任务列表，用逗号分隔。
# 这里的任务名应与您的数据集文件夹名保持一致。
# 注意：对于 mmlu，您的文件夹名为 mmlu_all_subjects，因此任务名也应使用它。
# 如果 lm-eval-harness 无法识别此任务名，可能需要创建自定义任务配置。
TASK_LIST="truthfulqa"

# 指定评估使用的设备 (例如: cuda:0, cpu)
DEVICE="cuda:0"

# 批处理大小。'auto' 会让框架自动寻找最优值。如果遇到显存不足(OOM)，请手动设置为一个较小的值，如 1 或 2。
BATCH_SIZE=16

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

lm_eval --model custom_moe \
    --model_args pretrained=${MODEL_PATH},expert_config_file=${EXPERT_CONFIG},trust_remote_code=True \
    --tasks ${TASK_LIST} \
    --device ${DEVICE} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH}


echo "----------------------------------------------------------------"
echo "评估完成！结果已保存至 ${OUTPUT_PATH} 目录。"