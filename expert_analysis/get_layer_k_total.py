"""
    专家总量恒定的层敏感性专家分配
"""
import json
import os
import math

def calculate_expert_allocation_budget(input_path, output_path, target_avg_k, k_min, k_max, method='robust'):
    """
    Args:
        target_avg_k (int/float): 期望的平均 topk（如 8）。
        k_min (int): 单层最少专家数。
        k_max (int): 单层最多专家数。
    """
    
    # 1. 读取文件
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        sensitivity_data = json.load(f)

    num_layers = len(sensitivity_data)
    target_total = int(target_avg_k * num_layers)  # 总预算，例如 8 * 47 = 376
    print(f"目标总专家数: {target_total} (平均 {target_avg_k}, 层数 {num_layers})")

    # 提取分数
    scores = [item['ratio_score'] for item in sensitivity_data]
    if not scores: return

    # 2. 归一化敏感度分数 [0, 1]
    if method == 'rank':
        # 基于排名的归一化
        score_ranks = sorted(range(num_layers), key=lambda i: sensitivity_data[i]['ratio_score'])
        normalized_scores = [0] * num_layers
        for rank, idx in enumerate(score_ranks):
            normalized_scores[idx] = rank / (num_layers - 1) if num_layers > 1 else 1
    else:
        # 基于数值的归一化 (Standard / Robust)
        if method == 'robust':
            sorted_scores = sorted(scores)
            p_low = sorted_scores[int(num_layers * 0.05)]
            p_high = sorted_scores[int(num_layers * 0.95)]
            effective_min, effective_max = p_low, p_high
        else:
            effective_min, effective_max = min(scores), max(scores)
        
        normalized_scores = []
        for s in scores:
            s_clipped = max(effective_min, min(effective_max, s))
            if effective_max == effective_min:
                normalized_scores.append(0.5)
            else:
                normalized_scores.append((s_clipped - effective_min) / (effective_max - effective_min))

    # 3. 核心：预算分配算法
    # 理论公式：k_i = k_min + lambda * normalized_score_i
    # 我们需要找到一个 lambda，使得 sum(k_i) = target_total
    # 但由于有 k_min 和 k_max 的约束，最简单的方法是线性缩放后做残差补偿
    
    # 初步分配（浮点数）
    # 先保证每层都有 k_min
    current_total = num_layers * k_min
    remaining_budget = target_total - current_total
    
    if remaining_budget < 0:
        print(f"警告: k_min ({k_min}) 太大，总数已超过预算！")
        # 强制处理：所有层设为 k_min
        initial_ks = [float(k_min)] * num_layers
    else:
        # 将剩余预算按归一化分数比例分配
        sum_norm = sum(normalized_scores)
        initial_ks = []
        for ns in normalized_scores:
            # 比例分配：k_min + (这一层的占比 * 剩余总预算)
            k_float = k_min + (ns / sum_norm * remaining_budget if sum_norm > 0 else remaining_budget / num_layers)
            # 裁剪到 k_max
            k_float = min(k_max, k_float)
            initial_ks.append(k_float)

    # 4. 整数化与残差修复 (Greedy Rounding)
    # 这里的技巧是：先向下取整，然后把差额补在“小数部分最大”的层上
    allocated_ks = [int(math.floor(k)) for k in initial_ks]
    diff = target_total - sum(allocated_ks)
    
    # 记录每个层的小数部分和索引，用于排序补差
    residuals = []
    for i in range(num_layers):
        # 只有还没达到 k_max 的层才能继续加
        if allocated_ks[i] < k_max:
            residuals.append({
                'index': i,
                'frac': initial_ks[i] - allocated_ks[i]
            })
    
    # 按小数部分从大到小排序
    residuals.sort(key=lambda x: x['frac'], reverse=True)
    
    # 补齐差额
    for i in range(min(len(residuals), int(diff))):
        allocated_ks[residuals[i]['index']] += 1

    # 如果 diff < 0 (即向下取整后还多了，通常不会发生)，逻辑类似，减去小数部分最小的

    # 5. 保存结果
    allocation_result = {}
    sorted_data = sorted(sensitivity_data, key=lambda x: int(x['layer_idx']))
    for i, item in enumerate(sensitivity_data):
        allocation_result[str(item['layer_idx'])] = allocated_ks[i]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(allocation_result, f, indent=4)
    
    # 打印最终统计
    final_sum = sum(allocated_ks)
    print(f"分配完成！最终总专家数: {final_sum} (目标: {target_total})")
    print(f"分配统计: Min={min(allocated_ks)}, Max={max(allocated_ks)}, Avg={final_sum/num_layers:.2f}")
    
    # 简单的分布可视化
    from collections import Counter
    counts = Counter(allocated_ks)
    for kv in sorted(counts.items()):
        print(f"k={kv[0]}: {kv[1]}层 {'█'*kv[1]}")

# 使用示例
if __name__ == "__main__":
    # INPUT_FILE = "./sensitivity_results_1k_hard_easy/layer_sensitivities_ratio.json"
    # OUTPUT_FILE = "./sensitivity_results_1k_hard_easy/expert_allocation_budget.json"
    # INPUT_FILE = "./deepseek_sensitivity_results_ration/layer_sensitivities_ratio.json"
    # OUTPUT_FILE = "./deepseek_sensitivity_results_ration/deepseek_expert_allocation_budget.json"
    INPUT_FILE = "./gptoss_sensitivity_results/layer_sensitivities_ratio.json"
    OUTPUT_FILE = "./gptoss_sensitivity_results/gptoss_expert_allocation_budget.json"
    
    # 配置
    TARGET_AVG_K = 4    # 保持平均 topk
    K_MIN = 2          # 允许的最少专家数
    K_MAX = 6         # 允许的最多专家数
    
    calculate_expert_allocation_budget(INPUT_FILE, OUTPUT_FILE, TARGET_AVG_K, K_MIN, K_MAX, method='robust')
    # calculate_expert_allocation_budget(INPUT_FILE, OUTPUT_FILE, TARGET_AVG_K, K_MIN, K_MAX, method='rank')
