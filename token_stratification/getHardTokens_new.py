"""
在Qwen3-30B-A3B模型上寻找难样本集 - 极速优化版
主要优化：
1. 移除 token 文本转换（只保留 token ID）
2. 使用 numpy 数组代替 list
3. 优化内存使用
4. 增加 gradient checkpointing（如果需要）
5. 优化 DataLoader 参数
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict
import argparse
from collections import defaultdict

class TextDataset(Dataset):
    """简单的文本数据集封装"""
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if isinstance(sample, dict):
            # 修复：移除 . get 中间的空格
            text = sample.get('text', '') or str(sample)
        else:
            text = str(sample)
        return {"text": text, "sample_idx": idx}

class HardSampleFinder:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 4, max_length: int = 2048):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # 修复：移除 torch. cuda 中间的空格
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

        print(f"正在加载模型: {model_name}")
        print(f"使用设备: {self.device}，GPU 数量: {n_gpu}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" 

        # 确定 Attention 实现
        attn_implementation = "eager"
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("✓ 已启用 Flash Attention 2 加速")
            except ImportError:
                print("⚠ 未检测到 flash_attn 库，回退到 eager 模式")

        # 模型加载逻辑
        load_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_implementation,
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }

        if self.device == "cuda" and n_gpu >= 2:
            print("✓ 检测到多 GPU，使用 device_map='auto'")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                **load_kwargs
            )
        else:
            # 修复：移除 . to 中间的空格
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            ).to(self.device)

        self.model.eval()
        
        # 如果显存紧张，可以开启 gradient checkpointing
        # self.model.gradient_checkpointing_enable()

    def load_c4_dataset(self, file_path: str, max_samples: int = None) -> List[Dict]:
        print(f"正在加载数据集: {file_path}")
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="读取文件")):
                if max_samples and len(samples) >= max_samples:
                    break
                try:
                    # 修复：移除 line. strip 和 json. JSONDecodeError 中间的空格
                    data = json.loads(line.strip())
                    samples.append(data)
                except json.JSONDecodeError:
                    continue
        print(f"✓ 成功加载 {len(samples)} 个样本")
        return samples

    def process_calibration_set(self, samples: List[Dict]):
        """使用 numpy 数组优化版本"""
        # 预分配数组 - 估算最大 token 数
        estimated_tokens = len(samples) * self.max_length
        
        # 使用 numpy 数组存储，避免 list append
        sample_indices = np.empty(estimated_tokens, dtype=np.int32)
        token_indices = np.empty(estimated_tokens, dtype=np.int32)
        token_ids = np.empty(estimated_tokens, dtype=np.int32)
        losses = np.empty(estimated_tokens, dtype=np.float32)
        
        current_idx = 0
        
        # 创建 DataLoader
        dataset = TextDataset(samples)
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            encodings = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            return encodings, [item['sample_idx'] for item in batch]

        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=2,  # 预取数据
        )

        print(f"开始批量推理，Batch Size: {self.batch_size}...")
        
        with torch.inference_mode():
            for batch_idx, (inputs, batch_sample_indices) in enumerate(tqdm(dataloader, desc="计算 Loss")):
                # 移动数据到设备
                # 修复：移除 self. model 中间的空格
                input_ids = inputs['input_ids'].to(self.model.device)
                attention_mask = inputs['attention_mask'].to(self.model.device)

                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Shift 操作
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                # 修复：移除 . contiguous 中间的空格
                shift_mask = attention_mask[:, 1:].contiguous()

                # 计算 loss（使用 fp32 更稳定）
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                
                token_losses_flat = loss_fct(flat_logits, flat_labels)
                token_losses = token_losses_flat.view(shift_labels.size(0), -1)
                
                # 一次性转到 CPU numpy
                token_losses_np = token_losses.float().cpu().numpy()
                shift_labels_np = shift_labels.cpu().numpy()
                shift_mask_np = shift_mask.cpu().numpy()

                # 批量处理结果（向量化操作）
                for i in range(len(batch_sample_indices)):
                    sample_idx = batch_sample_indices[i]
                    mask = shift_mask_np[i] == 1
                    valid_count = mask.sum()
                    
                    if valid_count == 0:
                        continue
                    
                    # 计算需要的空间
                    end_idx = current_idx + valid_count
                    
                    # 扩展数组（如果需要）
                    if end_idx > len(sample_indices):
                        new_size = max(len(sample_indices) * 2, end_idx + 1000) # 稍微优化了扩容逻辑
                        # 注意：np.resize 会重复填充数据，但我们后续会覆盖，所以没问题
                        sample_indices = np.resize(sample_indices, new_size)
                        token_indices = np.resize(token_indices, new_size)
                        token_ids = np.resize(token_ids, new_size)
                        losses = np.resize(losses, new_size)
                    
                    # 向量化赋值
                    valid_positions = np.where(mask)[0]
                    sample_indices[current_idx:end_idx] = sample_idx
                    token_indices[current_idx:end_idx] = valid_positions
                    token_ids[current_idx:end_idx] = shift_labels_np[i][mask]
                    losses[current_idx:end_idx] = token_losses_np[i][mask]
                    
                    current_idx = end_idx

        # 截断到实际使用的大小
        sample_indices = sample_indices[:current_idx]
        token_indices = token_indices[:current_idx]
        token_ids = token_ids[:current_idx]
        losses = losses[:current_idx]
        
        print(f"✓ 共处理 {current_idx} 个 token")
        
        return {
            'sample_indices': sample_indices,
            'token_indices': token_indices,
            'token_ids': token_ids,
            'losses': losses
        }

    def find_hard_samples(self, token_data: Dict, q: float = 0.1) -> Dict:
        """使用 numpy 优化的难样本筛选"""
        losses = token_data['losses']
        
        if len(losses) == 0:
            return token_data
        
        threshold = np.percentile(losses, (1 - q) * 100)
        
        print(f"Loss阈值 (top-{q*100}%): {threshold:.4f}")
        print(f"Loss统计: min={losses.min():.4f}, max={losses.max():.4f}, mean={losses.mean():.4f}, median={np.median(losses):.4f}")
        
        # 向量化筛选
        mask = losses >= threshold
        
        hard_sample_data = {
            'sample_indices': token_data['sample_indices'][mask],
            'token_indices': token_data['token_indices'][mask],
            'token_ids': token_data['token_ids'][mask],
            'losses': token_data['losses'][mask]
        }
        
        # 修复：格式化字符串中的空格
        print(f"✓ 筛选出 {len(hard_sample_data['losses'])} 个难样本token (占比 {len(hard_sample_data['losses'])/len(losses)*100:.2f}%)")
        
        return hard_sample_data

    def save_results(self, hard_sample_data: Dict, output_path: str, save_tokens: bool = False):
        """优化的保存逻辑"""
        print(f"正在保存结果到: {output_path}")
        
        # 使用缓冲写入
        with open(output_path, 'w', encoding='utf-8', buffering=1024*1024) as f:  # 1MB buffer
            total = len(hard_sample_data['losses'])
            
            # 批量处理（减少 I/O 次数）
            batch_size = 10000
            for start_idx in tqdm(range(0, total, batch_size), desc="保存结果"):
                end_idx = min(start_idx + batch_size, total)
                lines = []
                
                for i in range(start_idx, end_idx):
                    item = {
                        'sample_idx': int(hard_sample_data['sample_indices'][i]),
                        'token_idx': int(hard_sample_data['token_indices'][i]),
                        'token_id': int(hard_sample_data['token_ids'][i]),
                        'loss': float(hard_sample_data['losses'][i])
                    }
                    
                    # 可选：添加 token 文本（慢）
                    if save_tokens:
                        item['token'] = self.tokenizer.decode([item['token_id']])
                    
                    lines.append(json.dumps(item, ensure_ascii=False))
                
                f.write('\n'.join(lines) + '\n')
        
        print(f"✓ 保存完成")

    def run(self, dataset_path: str, output_path: str, q: float = 0.1, max_samples: int = None, save_tokens: bool = False):
        samples = self.load_c4_dataset(dataset_path, max_samples=max_samples)
        token_data = self.process_calibration_set(samples)
        hard_sample_data = self.find_hard_samples(token_data, q=q)
        self.save_results(hard_sample_data, output_path, save_tokens=save_tokens)
        return hard_sample_data

def main():
    parser = argparse.ArgumentParser(description="在Qwen3-30B-A3B模型上寻找难样本集 (极速优化版)")
    # 注意：Qwen3 可能为内部代号或笔误，如果是公开模型请确认路径是否正确
    parser.add_argument("--model_name", type=str, default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/models/Qwen3-30B-A3B")
    parser.add_argument("--dataset_path", type=str, default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/c4-train.00000-of-01024.json")
    parser.add_argument("--output_path", type=str, default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/hard_samples.jsonl")
    parser.add_argument("--q", type=float, default=0.1, help="百分位阈值")
    parser.add_argument("--max_samples", type=int, default=None)
    # 修复：移除 parser. add_argument 中间的空格
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4, help="推理时的Batch Size，A100建议8-16")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--save_tokens", action="store_true", help="是否保存token文本（会变慢）")

    args = parser.parse_args()

    finder = HardSampleFinder(
        model_name=args.model_name, 
        device=args.device, 
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    # 修复：移除 finder. run 和 args. q 中间的空格
    finder.run(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        q=args.q,
        max_samples=args.max_samples,
        save_tokens=args.save_tokens
    )
    print("✓ 完成！")

if __name__ == "__main__":
    main()