import json
import os
import math

def calculate_expert_allocation_advanced(input_path, output_path, k_min, k_max, method='robust'):
    """
    Args:
        method (str): 
            'standard' - 普通 Min-Max (你原来用的)
            'robust'   - 去除极端值后的 Min-Max (推荐，能拉开差距)
            'rank'     - 强制均匀分布 (基于排名)
    """
    
    # 1. 读取文件
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        sensitivity_data = json.load(f)

    # 提取分数
    scores = [item['ratio_score'] for item in sensitivity_data]
    if not scores: return

    allocation_result = {}
    
    # 按照 layer_idx 排序数据方便后续处理
    sorted_data = sorted(sensitivity_data, key=lambda x: int(x['layer_idx']))
    
    # ==========================================================
    # 核心算法逻辑分支
    # ==========================================================
    
    if method == 'rank':
        # --- 方法 B: 基于排名的强制均匀分布 ---
        print("正在使用: Rank-Based (基于排名) 策略")
        
        # 按分数从小到大排序，获取对应的 layer_idx
        # enumerate 返回 (rank, item)
        score_ranks = sorted(sensitivity_data, key=lambda x: x['ratio_score'])
        total_layers = len(score_ranks)
        
        # 创建一个临时字典存结果
        temp_map = {}
        for rank, item in enumerate(score_ranks):
            # rank 0 是最低分，rank (total-1) 是最高分
            # 归一化排名到 [0, 1]
            normalized_rank = rank / (total_layers - 1) if total_layers > 1 else 1
            
            k = k_min + normalized_rank * (k_max - k_min)
            k = int(round(k))
            temp_map[item['layer_idx']] = k
            
        # 填回结果
        for layer in sorted_data:
            allocation_result[str(layer['layer_idx'])] = temp_map[layer['layer_idx']]

    else:
        # --- 方法 A: 基于数值的归一化 (Standard / Robust) ---
        
        if method == 'robust':
            print("正在使用: Robust Min-Max (去极端值) 策略")
            # 排序以计算分位数
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            
            # 取 5% 和 95% 位置的值作为边界
            p_low = sorted_scores[int(n * 0.05)]
            p_high = sorted_scores[int(n * 0.95)]
            
            effective_min = p_low
            effective_max = p_high
        else:
            print("正在使用: Standard Min-Max (标准) 策略")
            effective_min = min(scores)
            effective_max = max(scores)
            
        print(f"有效分数范围: {effective_min:.4f} ~ {effective_max:.4f}")

        for layer in sorted_data:
            score = layer['ratio_score']
            
            # 截断：超过边界的直接按边界算
            score_clipped = max(effective_min, min(effective_max, score))
            
            if effective_max == effective_min:
                allocated_k = k_max
            else:
                normalized = (score_clipped - effective_min) / (effective_max - effective_min)
                allocated_k = k_min + normalized * (k_max - k_min)
                allocated_k = int(round(allocated_k))
            
            allocation_result[str(layer['layer_idx'])] = allocated_k

    # 3. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(allocation_result, f, indent=4)
    
    print(f"计算完成，已保存至 {output_path}")
    
    # 简单的分布统计
    vals = list(allocation_result.values())
    print(f"分配统计: Min={min(vals)}, Max={max(vals)}, Average={sum(vals)/len(vals):.2f}")
    # 打印前5个看一眼
    print("预览:", {k: allocation_result[k] for k in list(allocation_result)[:5]})

# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 输入文件名 (请确保此文件在同一目录下，或填写绝对路径)
    INPUT_FILE = "./sensitivity_results_1k_hard_easy/layer_sensitivities_ratio.json"
    
    # 输出文件名
    OUTPUT_FILE = "./sensitivity_results_1k_hard_easy/expert_allocation.json"
    
    # 这里的范围根据你的数据可以适当调整
    # 比如如果想拉大差距，可以把 k_min 设得更小一点
    K_MIN = 8   
    K_MAX = 12

    # 选项: 'standard' (原版), 'robust' (推荐), 'rank' (最均匀)
    # 建议先试 'robust'，如果还是太集中，就用 'rank'
    calculate_expert_allocation_advanced(INPUT_FILE, OUTPUT_FILE, K_MIN, K_MAX, method='robust')