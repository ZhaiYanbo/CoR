"""
分析C4数据集的token长度分布
"""

import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def analyze_dataset_lengths(dataset_path: str, tokenizer_path: str, max_samples: int = None):
    """分析数据集中文本的token长度分布"""
    
    print(f"正在加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print(f"正在分析数据集: {dataset_path}")
    lengths = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="统计长度")):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line. strip())
                text = data. get('text', '') if isinstance(data, dict) else str(data)
                
                # Tokenize并获取长度
                tokens = tokenizer.encode(text, add_special_tokens=True)
                lengths.append(len(tokens))
                
            except json.JSONDecodeError:
                continue
    
    lengths = np.array(lengths)
    
    # 统计信息
    print("\n" + "="*60)
    print("数据集长度统计")
    print("="*60)
    print(f"总样本数: {len(lengths)}")
    print(f"\n基础统计:")
    print(f"  最小长度: {lengths.min()}")
    print(f"  最大长度: {lengths.max()}")
    print(f"  平均长度: {lengths.mean():.2f}")
    print(f"  中位数:   {np.median(lengths):.2f}")
    print(f"  标准差:   {lengths.std():.2f}")
    
    print(f"\n百分位数:")
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 100]
    for p in percentiles:
        val = np.percentile(lengths, p)
        count = (lengths <= val).sum()
        ratio = count / len(lengths) * 100
        print(f"  {p:5.1f}%: {val:8.0f} tokens  (覆盖 {ratio:6.2f}% 样本)")
    
    # 不同max_length的覆盖率
    print(f"\n不同max_length的样本覆盖率:")
    test_lengths = [512, 1024, 1536, 2048, 3072, 4096]
    for max_len in test_lengths:
        covered = (lengths <= max_len).sum()
        ratio = covered / len(lengths) * 100
        truncated = len(lengths) - covered
        print(f"  max_length={max_len:4d}: 覆盖 {ratio:6.2f}% ({covered}/{len(lengths)})，截断 {truncated} 个样本")
    
    # 计算被截断的token比例
    print(f"\n不同max_length下被截断的token比例:")
    for max_len in test_lengths:
        total_tokens = lengths.sum()
        truncated_tokens = np.maximum(0, lengths - max_len). sum()
        ratio = truncated_tokens / total_tokens * 100
        print(f"  max_length={max_len:4d}: {ratio:6.3f}% tokens会被截断")
    
    # 绘制分布图
    print(f"\n正在生成分布图...")
    plt.figure(figsize=(15, 5))
    
    # 直方图
    plt.subplot(1, 3, 1)
    plt. hist(lengths, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Length')
    plt. ylabel('Frequency')
    plt.title('Token Length Distribution')
    plt.axvline(np.median(lengths), color='r', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
    plt.axvline(np.percentile(lengths, 95), color='g', linestyle='--', label=f'95%: {np.percentile(lengths, 95):.0f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CDF累积分布
    plt.subplot(1, 3, 2)
    sorted_lengths = np.sort(lengths)
    cdf = np. arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, cdf)
    plt.xlabel('Token Length')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    for max_len in [512, 1024, 2048]:
        idx = np.searchsorted(sorted_lengths, max_len)
        if idx < len(cdf):
            plt.axvline(max_len, color='r', linestyle='--', alpha=0.5)
            plt.text(max_len, cdf[idx], f'{max_len}\n({cdf[idx]:.1f}%)', fontsize=8)
    
    # Box plot
    plt.subplot(1, 3, 3)
    plt.boxplot(lengths, vert=True)
    plt.ylabel('Token Length')
    plt.title('Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_plot = dataset_path.replace('.json', '_length_distribution.png')
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✓ 分布图已保存到: {output_plot}")
    
    # 保存详细统计到文件
    output_stats = dataset_path.replace('.json', '_length_stats.txt')
    with open(output_stats, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f. write("数据集长度统计\n")
        f.write("="*60 + "\n")
        f.write(f"总样本数: {len(lengths)}\n\n")
        f.write(f"基础统计:\n")
        f.write(f"  最小长度: {lengths.min()}\n")
        f.write(f"  最大长度: {lengths. max()}\n")
        f.write(f"  平均长度: {lengths.mean():.2f}\n")
        f.write(f"  中位数:   {np.median(lengths):.2f}\n")
        f.write(f"  标准差:   {lengths.std():.2f}\n\n")
        
        f.write(f"百分位数:\n")
        for p in percentiles:
            val = np.percentile(lengths, p)
            count = (lengths <= val).sum()
            ratio = count / len(lengths) * 100
            f.write(f"  {p:5.1f}%: {val:8.0f} tokens  (覆盖 {ratio:6.2f}% 样本)\n")
        
        f.write(f"\n不同max_length的样本覆盖率:\n")
        for max_len in test_lengths:
            covered = (lengths <= max_len).sum()
            ratio = covered / len(lengths) * 100
            truncated = len(lengths) - covered
            f.write(f"  max_length={max_len:4d}: 覆盖 {ratio:6.2f}% ({covered}/{len(lengths)})，截断 {truncated} 个样本\n") 
    
    print(f"✓ 详细统计已保存到: {output_stats}")
    
    # 返回推荐的max_length
    recommended_max_length = int(np.percentile(lengths, 95))
    print(f"\n{'='*60}")
    print(f"推荐配置:")
    print(f"{'='*60}")
    print(f"如果想覆盖95%的样本: max_length={recommended_max_length}")
    print(f"如果想覆盖99%的样本: max_length={int(np.percentile(lengths, 99))}")
    print(f"如果显存充足: max_length=2048 (覆盖 {(lengths<=2048).sum()/len(lengths)*100:.2f}%)")
    print(f"如果显存紧张: max_length=1024 (覆盖 {(lengths<=1024).sum()/len(lengths)*100:. 2f}%)")
    
    return lengths

def main():
    parser = argparse.ArgumentParser(description="分析数据集token长度分布")
    parser.add_argument("--dataset_path", type=str, 
                       default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/c4-train.00000-of-01024.json")
    parser.add_argument("--tokenizer_path", type=str,
                       default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/models/Qwen3-30B-A3B")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="分析的最大样本数（None表示全部，建议先用10000快速测试）")
    
    args = parser.parse_args()
    
    analyze_dataset_lengths(
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()