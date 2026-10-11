"""
可视化层敏感性分析结果
"""

import json
import argparse
from pathlib import Path
import matplotlib. pyplot as plt
import seaborn as sns
import numpy as np


def visualize_sensitivity(sensitivity_file: str, output_dir: str = None):
    """可视化敏感性分析结果"""
    
    # 加载数据
    with open(sensitivity_file, 'r') as f:
        sensitivities = json.load(f)
    
    # 提取数据
    layers = [s['layer_idx'] for s in sensitivities]
    sensitivity_scores = [s['sensitivity_score'] for s in sensitivities]
    avg_loss_changes = [s['avg_loss_change'] for s in sensitivities]
    
    # 按层索引排序
    sorted_data = sorted(zip(layers, sensitivity_scores, avg_loss_changes))
    layers, sensitivity_scores, avg_loss_changes = zip(*sorted_data)
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图 1: 敏感性分数
    axes[0].bar(layers, sensitivity_scores, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Layer Index', fontsize=12)
    axes[0].set_ylabel('Sensitivity Score |ΔL|', fontsize=12)
    axes[0].set_title('Layer Sensitivity Scores', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 添加中位数线
    median_sensitivity = np.median(sensitivity_scores)
    axes[0].axhline(median_sensitivity, color='red', linestyle='--', 
                    label=f'Median = {median_sensitivity:.6f}')
    axes[0].legend()
    
    # 图 2: 平均 Loss 变化
    colors = ['green' if x > 0 else 'red' for x in avg_loss_changes]
    axes[1].bar(layers, avg_loss_changes, color=colors, alpha=0. 7)
    axes[1]. set_xlabel('Layer Index', fontsize=12)
    axes[1].set_ylabel('Average Loss Change', fontsize=12)
    axes[1].set_title('Average Loss Change per Layer', fontsize=14, fontweight='bold')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if output_dir:
        output_path = Path(output_dir) / 'sensitivity_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt. show()


def visualize_topk_allocation(topk_file: str, sensitivity_file: str, output_dir: str = None):
    """可视化 Top-K 分配方案"""
    
    # 加载数据
    with open(topk_file, 'r') as f:
        topk_allocation = json.load(f)
    
    with open(sensitivity_file, 'r') as f:
        sensitivities = json.load(f)
    
    # 构建映射
    layer_to_sensitivity = {s['layer_idx']: s['sensitivity_score'] for s in sensitivities}
    
    # 提取数据（排除 "default" 键）
    layers = sorted([int(k) for k in topk_allocation.keys() if k != "default"])
    topk_values = [topk_allocation[str(layer)] for layer in layers]
    sensitivity_values = [layer_to_sensitivity. get(layer, 0) for layer in layers]
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # 敏感性曲线
    color1 = 'tab:blue'
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Sensitivity Score', color=color1, fontsize=12)
    ax1.plot(layers, sensitivity_values, color=color1, marker='o', 
             markersize=4, linewidth=2, label='Sensitivity Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)
    
    # Top-K 柱状图
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Top-K Value', color=color2, fontsize=12)
    ax2. bar(layers, topk_values, color=color2, alpha=0.5, width=0.6, label='Top-K')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Layer Sensitivity and Top-K Allocation', fontsize=14, fontweight='bold')
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图表
    if output_dir:
        output_path = Path(output_dir) / 'topk_allocation_visualization. png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize layer sensitivity analysis results")
    
    parser.add_argument("--sensitivity_file", type=str, required=True,
                       help="Path to layer_sensitivities.json")
    parser.add_argument("--topk_file", type=str,
                       help="Path to topk_allocation.json (optional)")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for saving figures")
    
    args = parser.parse_args()
    
    # 可视化敏感性分数
    print("Generating sensitivity visualization...")
    visualize_sensitivity(args.sensitivity_file, args. output_dir)
    
    # 可视化 Top-K 分配（如果提供）
    if args.topk_file:
        print("Generating Top-K allocation visualization...")
        visualize_topk_allocation(args.topk_file, args.sensitivity_file, args.output_dir)


if __name__ == "__main__":
    main()