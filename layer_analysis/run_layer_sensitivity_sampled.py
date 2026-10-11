"""
添加采样功能的启动脚本
"""

import argparse
import logging
import sys
import os
import random
import json
from pathlib import Path

# 确保这里的导入名与您上一个保存的文件名一致
# 如果上一个文件保存为 layer_sensitivity_analyzer.py，请修改下行导入
from layer_sensitivity_analyzer_fast import LayerSensitivityAnalyzer, SensitivityConfig


def sample_tokens(tokens_file: str, sample_ratio: float = 0.1, output_suffix: str = "sampled", output_file: str = None):
    """
    对 tokens 文件进行采样（支持 hard/easy tokens）
    
    Args:
        tokens_file: 原始 tokens 文件
        sample_ratio: 采样比例 (0.1 = 10%)
        output_suffix: 输出文件后缀 (如 "sampled", "hard_sampled", "easy_sampled")
        output_file: 指定输出文件路径
    """
    import json
    
    # 读取所有 tokens
    tokens = []
    with open(tokens_file, 'r') as f:
        for line in f:
            if line.strip():
                tokens.append(json.loads(line))
    
    # 按 sample_idx 分组
    sample_groups = {}
    for token in tokens:
        sample_idx = token['sample_idx']
        if sample_idx not in sample_groups:
            sample_groups[sample_idx] = []
        sample_groups[sample_idx].append(token)
    
    # 采样样本
    all_sample_indices = list(sample_groups.keys())
    num_samples_to_keep = max(1, int(len(all_sample_indices) * sample_ratio))
    
    sampled_indices = random.sample(all_sample_indices, num_samples_to_keep)
    
    # 收集采样后的 tokens
    sampled_tokens = []
    for idx in sampled_indices:
        sampled_tokens.extend(sample_groups[idx])
    
    # 保存
    if output_file is None:
        base = tokens_file.replace('.jsonl', '')
        output_file = f'{base}_{output_suffix}_{int(sample_ratio*100)}pct.jsonl'
    
    with open(output_file, 'w') as f:
        for token in sampled_tokens:
            f.write(json.dumps(token) + '\n')
    
    print(f"Sampled {len(sampled_indices)} samples ({sample_ratio:.1%}) from {len(all_sample_indices)} total")
    print(f"Sampled {len(sampled_tokens)} tokens from {len(tokens)} total")
    print(f"Saved to: {output_file}")
    
    return output_file


# Backward compatibility alias
def sample_hard_tokens(hard_tokens_file: str, sample_ratio: float = 0.1, output_file: str = None):
    """
    对困难样本进行采样 (backward compatibility wrapper)
    
    Args:
        hard_tokens_file: 原始困难样本文件
        sample_ratio: 采样比例 (0.1 = 10%)
        output_file: 输出文件路径
    """
    return sample_tokens(hard_tokens_file, sample_ratio, output_suffix="sampled", output_file=output_file)


def main():
    parser = argparse.ArgumentParser(description="Layer Sensitivity Analysis with Sampling")
    
    # 必需参数
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hard_tokens", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    
    # 采样参数
    parser.add_argument("--sample_ratio", type=float, default=None,
                       help="Sample ratio for hard tokens (e.g., 0.1 for 10%%). None = use all samples")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (alternative to sample_ratio)")
    
    # 其他参数
    parser.add_argument("--dataset_text_field", type=str, default="text")
    parser.add_argument("--perturbation_scale", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--generate_topk", action="store_true")
    parser.add_argument("--min_topk", type=int, default=1)
    parser.add_argument("--max_topk", type=int, default=8)
    parser.add_argument("--allocation_strategy", type=str, default="proportional")
    parser.add_argument("--use_distributed", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./sensitivity_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--easy_tokens", type=str, help="Path to easy tokens JSONL for control group")
    parser.add_argument("--ratio_eps", type=float, default=1e-6, help="Small epsilon to avoid divide-by-zero in ratio")
    parser.add_argument("--log_file", type=str)
    
    # CEI Analysis parameters (new)
    parser.add_argument("--comprehensive", action="store_true",
                       help="Enable comprehensive analysis: Layer Sensitivity + Expert CEI scores")
    parser.add_argument("--cei_sample_limit", type=int, default=1000,
                       help="Max tokens per expert for CEI calculation (default: 1000)")
    parser.add_argument("--cei_critical_threshold", type=float, default=0.01,
                       help="CEI threshold for critical experts (default: 0.01)")
    parser.add_argument("--cei_harmful_threshold", type=float, default=-0.001,
                       help="CEI threshold for harmful experts (default: -0.001)")
    
    args = parser.parse_args()
    
    # 设置日志
    rank = int(os.environ.get('RANK', 0))
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # ============ 采样逻辑（统一处理 hard 和 easy）============
    hard_tokens_file = args.hard_tokens
    easy_tokens_file = args.easy_tokens
    
    if rank == 0:
        # 确定采样比例
        sample_ratio = None
        
        if args.sample_ratio is not None:
            sample_ratio = args.sample_ratio
            logger.info(f"Using sample ratio: {sample_ratio:.1%}")
        
        elif args.max_samples is not None:
            # 根据 hard tokens 计算采样比例
            import json
            total_samples = set()
            with open(args.hard_tokens, 'r') as f:
                for line in f:
                    if line.strip():
                        total_samples.add(json.loads(line)['sample_idx'])
            
            sample_ratio = min(1.0, args.max_samples / len(total_samples))
            logger.info(f"Max samples: {args.max_samples}, Total: {len(total_samples)}, Ratio: {sample_ratio:.1%}")
        
        # 采样 hard tokens
        if sample_ratio is not None:
            logger.info(f"Sampling hard tokens...")
            hard_tokens_file = sample_tokens(
                args.hard_tokens, 
                sample_ratio, 
                output_suffix="hard_sampled"
            )
        
        # 采样 easy tokens（如果提供）
        if args.easy_tokens and sample_ratio is not None:
            logger.info(f"Sampling easy tokens with same ratio...")
            easy_tokens_file = sample_tokens(
                args.easy_tokens, 
                sample_ratio, 
                output_suffix="easy_sampled"
            )
        
        # 如果有 easy tokens 但没有采样参数，警告用户
        if args.easy_tokens and sample_ratio is None:
            logger.warning(
                "⚠️  Easy tokens provided but no sampling specified. "
                "This may take a long time if easy tokens file is large. "
                "Consider using --max_samples or --sample_ratio."
            )
    
    # 等待主进程完成采样
    if args.use_distributed:
        import torch.distributed as dist
        dist.barrier()
    
    # 创建配置
    config = SensitivityConfig(
        model_name=args.model,
        hard_tokens_file=hard_tokens_file,
        easy_tokens_file=easy_tokens_file,  # Use sampled easy tokens if available
        dataset_path=args.dataset_path,
        dataset_text_field=args.dataset_text_field,
        perturbation_scale=args.perturbation_scale,
        device=args.device,
        max_length=args.max_length,
        output_dir=args.output_dir,
        use_distributed=args.use_distributed,
        world_size=int(os.environ.get('WORLD_SIZE', 1)),
        rank=rank,
        local_rank=int(os.environ.get('LOCAL_RANK', 0)),
        ratio_eps=args.ratio_eps,  # New: epsilon for safe division in ratio calculation
        # CEI Analysis configuration (new)
        enable_cei_analysis=args.comprehensive,
        cei_sample_limit=args.cei_sample_limit,
        cei_critical_threshold=args.cei_critical_threshold,
        cei_harmful_threshold=args.cei_harmful_threshold
    )
    
    if rank == 0:
        print("\n" + "=" * 80)
        if args.comprehensive:
            print("HD-AAV Comprehensive Analysis")
            print("Layer Sensitivity + Expert CEI Score")
        else:
            print("Layer Sensitivity Analysis - Optimized with Sampling")
        print("=" * 80)
        
        # Show sampling configuration
        if args.sample_ratio or args.max_samples:
            print("\n📊 SAMPLING CONFIGURATION")
            print("=" * 80)
            if args.sample_ratio:
                print(f"Sample ratio: {args.sample_ratio:.1%}")
            if args.max_samples:
                print(f"Max samples: {args.max_samples}")
            
            print(f"\n🔴 Hard tokens:")
            print(f"   Original: {args.hard_tokens}")
            print(f"   Sampled:  {hard_tokens_file}")
            
            if args.easy_tokens:
                print(f"\n🟢 Easy tokens:")
                print(f"   Original: {args.easy_tokens}")
                print(f"   Sampled:  {easy_tokens_file}")
            print("=" * 80)
        else:
            print("No sampling (using full datasets)")
        
        print(f"\nModel: {args.model}")
        print(f"Dataset: {args.dataset_path}")
        
        if args.comprehensive:
            print(f"\n🔍 Comprehensive mode enabled:")
            print(f"  - CEI sample limit: {args.cei_sample_limit}")
            print(f"  - Critical threshold: {args.cei_critical_threshold}")
            print(f"  - Harmful threshold: {args.cei_harmful_threshold}")
        
        print(f"\nOutput: {args.output_dir}")
        print("=" * 80 + "\n")
    
    # 运行分析
    try:
        analyzer = LayerSensitivityAnalyzer(config)
        
        if rank == 0:
            print(f"\n✓ Loaded {len(analyzer.hard_tokens)} hard tokens")
            if analyzer.has_easy_tokens:
                print(f"✓ Loaded {len(analyzer.easy_tokens)} easy tokens")
            print(f"✓ Will process {len(analyzer.hard_token_map)} unique samples")
            print(f"✓ Model: {analyzer.num_layers} layers, {analyzer.num_experts} experts/layer")
            
            if args.comprehensive:
                print("\n🚀 Running comprehensive analysis (Layer Sensitivity + Expert CEI)...")
            else:
                print("\nRunning layer sensitivity analysis...")
        
        # Choose analysis mode based on --comprehensive flag
        if args.comprehensive:
            # Comprehensive mode: Layer Sensitivity + Expert CEI
            sensitivities, expert_cei_scores = analyzer.run_comprehensive_analysis()
            easy_sensitivities = None
            
            if rank == 0 and sensitivities:
                print("\n✅ Comprehensive analysis completed!")
                print(f"\nTop 5 Most Sensitive Layers:")
                for i, s in enumerate(sensitivities[:5], 1):
                    print(f"  {i}. Layer {s.layer_idx}: {s.sensitivity_score:.6f}")
                
                if expert_cei_scores:
                    print(f"\n📊 Expert CEI Summary:")
                    critical_count = sum(1 for e in expert_cei_scores if e.interpretation == "critical")
                    harmful_count = sum(1 for e in expert_cei_scores if e.interpretation == "harmful")
                    redundant_count = sum(1 for e in expert_cei_scores if e.interpretation == "redundant")
                    print(f"  🔥 Critical experts: {critical_count}")
                    print(f"  ❌ Harmful experts: {harmful_count}")
                    print(f"  💤 Redundant experts: {redundant_count}")
        else:
            # Standard mode: Layer Sensitivity only (with optional easy tokens)
            sensitivities, easy_sensitivities = analyzer.run_full_analysis()
            
            if rank == 0 and sensitivities:
                print("\n✅ Analysis completed!")
                print(f"\nTop 5 Most Sensitive Layers (Hard Tokens):")
                for i, s in enumerate(sensitivities[:5], 1):
                    print(f"  {i}. Layer {s.layer_idx}: {s.sensitivity_score:.6f}")
        
        if rank == 0 and sensitivities:
            # If easy tokens provided, compute ratio scores (only in standard mode)
            ratio_results = None
            if not args.comprehensive and analyzer.has_easy_tokens and easy_sensitivities:
                print("\n✓ Computing ratio scores (hard/easy) with vectorization...")
                ratio_results = analyzer.compute_ratio_scores()
                
                print(f"\nTop 5 Layers by Ratio (Hard/Easy):")
                for i, r in enumerate(ratio_results[:5], 1):
                    print(f"  {i}. Layer {r['layer_idx']}: {r['ratio_score']:.6f} "
                          f"(hard={r['hard_score']:.6f}, easy={r['easy_score']:.6f})")
            
            # Generate topk allocation (if requested)
            topk_allocation = None
            if args.generate_topk:
                topk_allocation = analyzer.generate_topk_allocation(
                    args.min_topk, args.max_topk, args.allocation_strategy
                )
            
            # Save all results
            analyzer.save_results(topk_allocation, ratio_results)
            
            print(f"\n✓ Results saved to: {args.output_dir}")
    
    finally:
        if 'analyzer' in locals():
            analyzer.cleanup()


if __name__ == "__main__":
    main()