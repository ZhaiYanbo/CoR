"""
在Qwen3-30B-A3B模型上寻找难样本集 - 优化版
优化点：Batch推理、Flash Attention、向量化Loss计算
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict
import argparse

def build_device_map_for_multi_gpu():
    return "auto"

class TextDataset(Dataset):
    """简单的文本数据集封装"""
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if isinstance(sample, dict):
            text = sample.get('text', '') or str(sample)
        else:
            text = str(sample)
        return {"text": text, "original_sample": sample, "sample_idx": idx}

class HardSampleFinder:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size

        # 设备选择
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

        print(f"正在加载模型: {model_name}")
        print(f"使用设备: {self.device}，GPU 数量: {n_gpu}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # 补齐 pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 【修改】计算Loss通常使用右侧Padding，便于对齐label
        self.tokenizer.padding_side = "right" 

        # 确定 Attention 实现 (Ampere架构以上如A100/A10/3090/4090推荐使用flash_attention_2)
        attn_implementation = "eager"
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("已启用 Flash Attention 2 加速")
            except ImportError:
                print("未检测到 flash_attn 库，回退到 eager 模式")

        # 模型加载逻辑
        load_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_implementation
        }

        if self.device == "cuda":
            if n_gpu >= 2:
                # 多卡
                print("检测到多 GPU，使用 device_map='auto'")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    **load_kwargs
                )
            else:
                # 单卡
                print("单 GPU 环境")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    **load_kwargs
                ).to(self.device)
        else:
            # CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                **load_kwargs
            ).to(self.device)

        self.model.eval()
        # 预先定义 Loss 函数，reduction='none' 表示返回每个 token 的 loss
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def load_c4_dataset(self, file_path: str, max_samples: int = None) -> List[Dict]:
        print(f"正在加载数据集: {file_path}")
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="读取文件")):
                if max_samples and len(samples) >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                except json.JSONDecodeError:
                    continue
        print(f"成功加载 {len(samples)} 个样本")
        return samples

    def process_calibration_set(self, samples: List[Dict]):
        all_token_losses = []
        
        # 创建 DataLoader
        dataset = TextDataset(samples)
        
        # Collate function: 负责将 batch 内的文本 tokenize 并 padding 到同一长度
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            # 批量 Tokenize
            encodings = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            return encodings, batch

        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=4,  # 使用多进程预处理数据
            pin_memory=True if self.device == "cuda" else False
        )

        print(f"开始批量推理，Batch Size: {self.batch_size}...")
        
        with torch.inference_mode():
            for batch_idx, (inputs, batch_meta) in enumerate(tqdm(dataloader, desc="计算 Loss")):
                # 移动数据到 GPU (对于 device_map='auto'，input_ids 只需要在 CPU 或主 GPU，transformers 会处理)
                # 但为了安全起见，我们通常把 inputs 放到 model.device 或第一个 gpu
                if self.model.device.type != 'meta':
                    input_ids = inputs['input_ids'].to(self.model.device)
                    attention_mask = inputs['attention_mask'].to(self.model.device)
                else:
                    # 多卡自动分配情况，通常传给 model 即可，但在计算 loss 时需要 device 一致
                    input_ids = inputs['input_ids'].to("cuda")
                    attention_mask = inputs['attention_mask'].to("cuda")

                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # --- 批量计算 Token-level Loss ---
                # Shift操作：Logits 是预测下一个 token，所以 logits[..., :-1, :] 预测的是 input_ids[..., 1:]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:].contiguous()

                # Flatten 用于计算 CrossEntropy
                # shape: [batch_size * (seq_len-1), vocab_size]
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                
                # 计算所有 token 的 loss (不求和)
                token_losses_flat = self.loss_fct(flat_logits, flat_labels)
                
                # 恢复形状: [batch_size, seq_len-1]
                token_losses = token_losses_flat.view(shift_labels.size(0), -1)
                
                # 应用 mask：padding 部分的 loss 设为 0 或忽略 (这里我们只收集非 padding 的)
                # 转回 CPU 处理结果
                token_losses_cpu = token_losses.float().cpu().numpy()
                shift_labels_cpu = shift_labels.cpu().numpy()
                shift_mask_cpu = shift_mask.cpu().numpy()

                # --- 整理结果 ---
                for i in range(len(batch_meta)): # 遍历 batch 内的每个样本
                    sample_idx = batch_meta[i]['sample_idx']
                    # 获取有效长度 (mask 为 1 的部分)
                    valid_indices = np.where(shift_mask_cpu[i] == 1)[0]
                    
                    if len(valid_indices) == 0:
                        continue
                        
                    # 提取对应的 losses 和 token ids
                    valid_losses = token_losses_cpu[i][valid_indices]
                    valid_ids = shift_labels_cpu[i][valid_indices]
                    
                    # 批量转 token 文本 (比较慢，如果不需要存具体 token 文本可以跳过，只存 ID)
                    # 为了性能，建议这里不转回 string，或者只对难样本转。
                    # 这里保持原有逻辑，但注意 convert_ids_to_tokens 比较慢
                    valid_tokens = self.tokenizer.convert_ids_to_tokens(valid_ids)

                    for t_idx, (loss, token) in enumerate(zip(valid_losses, valid_tokens)):
                        # 这里将 token_idx 修正为原始序列中的位置
                        original_pos = valid_indices[t_idx] 
                        all_token_losses.append({
                            'sample_idx': sample_idx,
                            'token_idx': int(original_pos),
                            'token': token,
                            'loss': float(loss)
                        })

        return all_token_losses

    def find_hard_samples(self, all_token_losses: List[Dict], q: float = 0.1) -> List[Dict]:
        if not all_token_losses:
            return []

        # 使用 numpy 快速处理
        losses = np.array([item['loss'] for item in all_token_losses], dtype=np.float32)
        threshold = np.percentile(losses, (1 - q) * 100)

        print(f"Loss阈值 (top-{q*100}%): {threshold:.4f}")
        print(f"Loss统计: min={losses.min():.4f}, max={losses.max():.4f}, mean={losses.mean():.4f}")

        # 筛选
        hard_samples = [item for item in all_token_losses if item['loss'] >= threshold]

        print(f"筛选出 {len(hard_samples)} 个难样本token (共 {len(all_token_losses)} 个token)")
        return hard_samples

    def save_results(self, hard_samples: List[Dict], output_path: str):
        print(f"正在保存结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in hard_samples:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def run(self, dataset_path: str, output_path: str, q: float = 0.1, max_samples: int = None):
        samples = self.load_c4_dataset(dataset_path, max_samples=max_samples)
        all_token_losses = self.process_calibration_set(samples)
        hard_samples = self.find_hard_samples(all_token_losses, q=q)
        self.save_results(hard_samples, output_path)
        return hard_samples

def main():
    parser = argparse.ArgumentParser(description="在Qwen3-30B-A3B模型上寻找难样本集 (高速版)")
    parser.add_argument("--model_name", type=str, default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/models/Qwen3-30B-A3B")
    parser.add_argument("--dataset_path", type=str, default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/c4-train.00000-of-01024.json")
    parser.add_argument("--output_path", type=str, default="/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/hard_samples.jsonl")
    parser.add_argument("--q", type=float, default=0.1, help="百分位阈值")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4, help="推理时的Batch Size，根据显存调整")

    args = parser.parse_args()

    finder = HardSampleFinder(model_name=args.model_name, device=args.device, batch_size=args.batch_size)
    finder.run(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        q=args.q,
        max_samples=args.max_samples
    )
    print("完成！")

if __name__ == "__main__":
    main()