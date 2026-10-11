"""
Layer Sensitivity Analyzer for MoE Models - V3
优化本地 JSON 文件数据集加载和多卡支持
"""

import json
import logging
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Constants for numerical stability
MIN_WEIGHT_THRESHOLD = 1e-8  # Minimum weight sum threshold for normalization


@dataclass
class HardToken:
    """困难样本信息"""
    sample_idx: int
    token_idx: int
    token: str
    loss: float
    global_position: Optional[int] = None


@dataclass
class LayerSensitivity:
    """层敏感性结果"""
    layer_idx: int
    sensitivity_score: float  # |ΔL_l|
    avg_loss_change: float
    max_loss_change: float
    std_loss_change: float
    num_hard_tokens: int


@dataclass
class ExpertCEIScore:
    """Expert CEI (Counterfactual Expert Impact) Score"""
    layer_idx: int
    expert_idx: int
    cei_score: float  # Rescue score (positive=critical, negative=harmful, near zero=redundant)
    num_tokens: int  # Number of hard tokens processed
    avg_routing_weight: float  # Average routing weight
    tokens_handled: List[Tuple[int, int]]  # [(sample_idx, token_idx)]
    interpretation: str = ""  # "critical" / "harmful" / "redundant"


@dataclass
class SensitivityConfig:
    """敏感性分析配置"""
    model_name: str
    hard_tokens_file: str
    
    # 数据集配置
    dataset_path: Optional[str] = None  # 本地 JSON/JSONL 文件路径
    dataset_name: str = "c4"
    dataset_split: str = "validation"
    dataset_text_field: str = "text"  # JSON 中文本字段的名称
    num_samples: int = 100
    
    # 分析参数
    perturbation_scale: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    output_dir: str = "./sensitivity_results"
    batch_size: int = 1
    trust_remote_code: bool = True
    
    # 多卡配置
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    dist_backend: str = "nccl"
    
    # 内存优化
    load_in_chunks: bool = False  # 是否分块加载大 JSON 文件
    chunk_size: int = 1000  # 每块加载的样本数
    
    # Comparative analysis (new)
    easy_tokens_file: Optional[str] = None  # Easy tokens file for control group
    ratio_eps: float = 1e-6  # Epsilon for ratio calculation to prevent division by zero
    
    # CEI (Counterfactual Expert Impact) analysis configuration
    enable_cei_analysis: bool = False  # Enable comprehensive expert CEI analysis
    cei_sample_limit: int = 1000  # Max tokens per expert to sample for CEI calculation
    cei_critical_threshold: float = 0.01  # CEI > threshold = critical expert
    cei_harmful_threshold: float = -0.001  # CEI < threshold = harmful expert


class LayerSensitivityAnalyzer:
    """层敏感性分析器 - 优化 JSON/JSONL 文件支持"""
    
    def __init__(self, config: SensitivityConfig):
        self.config = config
        
        # 初始化分布式环境
        if config.use_distributed:
            self._init_distributed()
        
        self.device = torch.device(f"cuda:{config.local_rank}" if config.use_distributed 
                                  else config.device)
        self.is_main_process = config.rank == 0
        
        if self.is_main_process:
            logger.info(f"Initializing LayerSensitivityAnalyzer...")
            logger.info(f"Distributed: {config.use_distributed}, World size: {config.world_size}, Rank: {config.rank}")
        
        # 加载模型和分词器
        self._load_model()
        
        # 加载困难样本
        self.hard_tokens: List[HardToken] = []
        self.hard_token_map: Dict[int, List[int]] = defaultdict(list)
        self._load_hard_tokens()
        
        # Load easy samples (if provided)
        self.easy_tokens: List[HardToken] = []
        self.easy_token_map: Dict[int, List[int]] = defaultdict(list)
        self.has_easy_tokens = config.easy_tokens_file is not None
        if self.has_easy_tokens:
            self._load_easy_tokens()
        
        # 加载数据集
        self.dataset_texts: List[str] = []
        self._load_dataset()
        
        # 结果存储
        self.layer_sensitivities: List[LayerSensitivity] = []
        self.original_losses: Dict[Tuple[int, int], float] = {}
        
        # Storage for comparative analysis results (new)
        self.easy_layer_sensitivities: List[LayerSensitivity] = []
        self.easy_original_losses: Dict[Tuple[int, int], float] = {}
        self.ratio_results: Optional[List[Dict]] = None  # Initialize to None, computed when needed
        
        # CEI Analysis storage (new)
        self.expert_cei_scores: List[ExpertCEIScore] = []
        # routing_info[layer_idx][expert_idx] = [(sample_idx, token_idx), ...]
        self.routing_info: Dict[int, Dict[int, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        # routing_weights[layer_idx][expert_idx] = [weight1, weight2, ...]
        self.routing_weights: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        
    def _init_distributed(self):
        """初始化分布式训练环境"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.config.rank = int(os.environ['RANK'])
            self.config.world_size = int(os.environ['WORLD_SIZE'])
            self.config.local_rank = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            self.config.rank = int(os.environ['SLURM_PROCID'])
            self.config.world_size = int(os.environ['SLURM_NTASKS'])
            self.config.local_rank = int(os.environ['SLURM_LOCALID'])
        
        torch.cuda.set_device(self.config.local_rank)
        dist.init_process_group(
            backend=self.config.dist_backend,
            init_method='env://',
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        logger.info(f"Initialized distributed process: rank={self.config.rank}, "
                   f"world_size={self.config.world_size}, local_rank={self.config.local_rank}")
    
    def _load_model(self):
        """加载模型"""
        if self.is_main_process:
            logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        if self.config.use_distributed:
            # 多卡模式
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16,
                device_map=None
            )
            self.model = self.model.to(self.device)
        else:
            # 单卡模式
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_experts = getattr(self.model.config, 'num_experts', None)
        
        if self.is_main_process:
            logger.info(f"Model loaded: {self.num_layers} layers, {self.num_experts} experts per layer")
    
    def _load_hard_tokens(self):
        """加载 JSONL 格式的困难样本"""
        if self.is_main_process:
            logger.info(f"Loading hard tokens from: {self.config.hard_tokens_file}")
        
        hard_tokens_path = Path(self.config.hard_tokens_file)
        if not hard_tokens_path.exists():
            raise FileNotFoundError(f"Hard tokens file not found: {hard_tokens_path}")
        
        with open(hard_tokens_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    hard_token = HardToken(
                        sample_idx=data['sample_idx'],
                        token_idx=data['token_idx'],
                        token=data['token'],
                        loss=data['loss']
                    )
                    self.hard_tokens.append(hard_token)
                    self.hard_token_map[data['sample_idx']].append(data['token_idx'])
        
        if self.is_main_process:
            logger.info(f"Loaded {len(self.hard_tokens)} hard tokens from {len(self.hard_token_map)} samples")
            samples_with_hard_tokens = sorted(self.hard_token_map.keys())
            logger.info(f"Sample indices range: [{min(samples_with_hard_tokens)}, {max(samples_with_hard_tokens)}]")

    def _load_easy_tokens(self):
        """Load easy tokens in JSONL format (control group)"""
        if self.is_main_process:
            logger.info(f"Loading easy tokens from: {self.config.easy_tokens_file}")
        
        easy_tokens_path = Path(self.config.easy_tokens_file)
        if not easy_tokens_path.exists():
            raise FileNotFoundError(f"Easy tokens file not found: {easy_tokens_path}")
        
        with open(easy_tokens_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    easy_token = HardToken(
                        sample_idx=data['sample_idx'],
                        token_idx=data['token_idx'],
                        token=data['token'],
                        loss=data['loss']
                    )
                    self.easy_tokens.append(easy_token)
                    self.easy_token_map[data['sample_idx']].append(data['token_idx'])
        
        if self.is_main_process:
            logger.info(f"Loaded {len(self.easy_tokens)} easy tokens from {len(self.easy_token_map)} samples")
            if self.easy_token_map:  # Only log range if there are easy tokens
                samples_with_easy_tokens = sorted(self.easy_token_map.keys())
                logger.info(f"Sample indices range: [{min(samples_with_easy_tokens)}, {max(samples_with_easy_tokens)}]")

    def _load_dataset(self):
        """Load dataset - optimized JSON/JSONL file support"""
        if self.is_main_process:
            logger.info(f"Loading dataset...")
        
        # [Key optimization] Merge indices of hard and easy samples
        all_sample_indices = set(self.hard_token_map.keys())
        if self.has_easy_tokens:
            all_sample_indices.update(self.easy_token_map.keys())
        
        max_sample_idx = max(all_sample_indices)
        
        # Calculate number of samples needed (only load samples with tokens)
        num_samples_needed = max_sample_idx + 1
        
        if self.is_main_process:
            logger.info(f"Total samples needed: {num_samples_needed}")
            logger.info(f"Samples with hard tokens: {len(self.hard_token_map)}")
            if self.has_easy_tokens:
                logger.info(f"Samples with easy tokens: {len(self.easy_token_map)}")
            logger.info(f"Unique samples to load: {len(all_sample_indices)}")
            logger.info(f"Optimization: Will only process {len(all_sample_indices)} samples (skip {num_samples_needed - len(all_sample_indices)})")
        
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required!")
        
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        if self.is_main_process:
            logger.info(f"Loading dataset from: {dataset_path}")
        
        # 加载数据集（只加载需要的索引）
        self.dataset_texts = self._load_json_dataset_sparse(
            dataset_path, 
            num_samples_needed,
            all_sample_indices  # 传入需要的样本索引（合并后的）
        )
        
        if self.is_main_process:
            logger.info(f"Loaded {len(self.dataset_texts)} texts from dataset")
            if len(self.dataset_texts) > 0:
                # 找到第一个非空文本
                first_valid = next((t for t in self.dataset_texts if t), None)
                if first_valid:
                    logger.info(f"Sample text preview (first 100 chars): {first_valid[:100]}...")
    
    def _load_json_dataset_sparse(
        self, 
        json_path: Path, 
        max_sample_idx: int,
        required_indices: Set[int]
    ) -> List[str]:
        """
        稀疏加载 JSONL 数据集 - 只加载包含困难 token 的样本
        这样可以大幅减少内存占用和加载时间
        """
        
        # 初始化：用空字符串填充，只有需要的索引才会被填充
        texts = [""] * (max_sample_idx + 1)
        text_field = self.config.dataset_text_field
        
        loaded_count = 0
        
        if self.is_main_process:
            logger.info(f"Sparse loading: only loading {len(required_indices)} out of {max_sample_idx + 1} samples")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # JSONL 格式：逐行读取
                for i, line in enumerate(f):
                    # 只加载需要的样本
                    if i not in required_indices:
                        if i > max_sample_idx:
                            break
                        continue
                    
                    if not line.strip():
                        continue
                    
                    try:
                        item = json.loads(line)
                        
                        if isinstance(item, dict) and text_field in item:
                            text = item[text_field]
                            if text and len(str(text).strip()) > 0:
                                texts[i] = str(text)
                                loaded_count += 1
                        elif isinstance(item, str):
                            texts[i] = item
                            loaded_count += 1
                    
                    except json.JSONDecodeError as e:
                        if self.is_main_process:
                            logger.warning(f"Line {i}: JSON decode error: {e}")
                        continue
                    
                    # 早停：如果已经加载了所有需要的样本
                    if loaded_count >= len(required_indices):
                        if self.is_main_process:
                            logger.info(f"All required samples loaded at line {i}")
                        break
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        if loaded_count == 0:
            raise ValueError(f"No texts loaded!  Check your dataset format.")
        
        if self.is_main_process:
            logger.info(f"Sparse loading completed: {loaded_count}/{len(required_indices)} samples loaded")
        
        return texts
    
    def _compute_token_losses(
        self,
        token_map: Dict[int, List[int]],
        desc_prefix: str = "Computing losses"
    ) -> Dict[Tuple[int, int], float]:
        """
        Generic method: compute original loss for specified token set
        
        Args:
            token_map: Token index mapping {sample_idx: [token_idx, ...]}
            desc_prefix: Progress bar description prefix
            
        Returns:
            losses: {(sample_idx, token_idx): loss}
        """
        samples_to_process = sorted(token_map.keys())
        
        if self.is_main_process:
            logger.info(f"{desc_prefix}: Processing {len(samples_to_process)} samples")
        
        # 多卡：分配样本
        if self.config.use_distributed:
            samples_per_rank = len(samples_to_process) // self.config.world_size
            start_idx = self.config.rank * samples_per_rank
            end_idx = start_idx + samples_per_rank if self.config.rank < self.config.world_size - 1 else len(samples_to_process)
            samples_to_process = samples_to_process[start_idx:end_idx]
            
            logger.info(f"Rank {self.config.rank}: Processing {len(samples_to_process)} samples (indices {start_idx} to {end_idx-1})")
        
        local_losses = {}
        processed_count = 0
        skipped_count = 0
        
        with torch.no_grad():
            desc = f"[Rank {self.config.rank}] {desc_prefix}" if self.config.use_distributed else desc_prefix
            iterator = tqdm(samples_to_process, desc=desc) if self.is_main_process or not self.config.use_distributed else samples_to_process
            
            for sample_idx in iterator:
                if sample_idx >= len(self.dataset_texts):
                    logger.warning(f"Sample {sample_idx} exceeds dataset length ({len(self.dataset_texts)}), skipping")
                    skipped_count += 1
                    continue
                
                text = self.dataset_texts[sample_idx]
                
                # 检查文本有效性
                if not text or len(text.strip()) == 0:
                    logger.warning(f"Sample {sample_idx} is empty, skipping")
                    skipped_count += 1
                    continue
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs.input_ids)
                
                # 计算 token-level loss
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = inputs.input_ids[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Store loss for specified tokens
                token_indices = token_map[sample_idx]
                for token_idx in token_indices:
                    if token_idx < len(token_losses):
                        loss_val = float(token_losses[token_idx].cpu())
                        # NaN/Inf protection
                        if not np.isnan(loss_val) and not np.isinf(loss_val):
                            local_losses[(sample_idx, token_idx)] = loss_val
                        else:
                            logger.warning(f"Skipping invalid original loss for token ({sample_idx}, {token_idx}): {loss_val}")
                
                processed_count += 1
        
        if self.is_main_process:
            logger.info(f"{desc_prefix}: Processed: {processed_count}, Skipped: {skipped_count}")
        
        # 多卡：收集所有进程的结果
        if self.config.use_distributed:
            all_losses = [None] * self.config.world_size
            dist.all_gather_object(all_losses, local_losses)
            
            if self.is_main_process:
                gathered_losses = {}
                for losses_dict in all_losses:
                    if losses_dict:
                        gathered_losses.update(losses_dict)
            else:
                gathered_losses = local_losses
            
            # 广播到所有进程
            if self.is_main_process:
                losses_to_broadcast = gathered_losses
            else:
                losses_to_broadcast = None
            
            losses_list = [losses_to_broadcast]
            dist.broadcast_object_list(losses_list, src=0)
            return losses_list[0]
        else:
            return local_losses
    
    def compute_original_losses(self):
        """Compute original loss (supports simultaneous calculation for hard and easy tokens)"""
        if self.is_main_process:
            logger.info("Computing original losses...")
        
        # 计算困难 token 的 loss
        if self.is_main_process:
            logger.info("Computing losses for hard tokens...")
        self.original_losses = self._compute_token_losses(
            self.hard_token_map,
            desc_prefix="Hard tokens"
        )
        
        if self.is_main_process:
            logger.info(f"Computed original losses for {len(self.original_losses)} hard tokens")
        
        # 如果有简单 token，也计算它们的 loss
        if self.has_easy_tokens:
            if self.is_main_process:
                logger.info("Computing losses for easy tokens...")
            self.easy_original_losses = self._compute_token_losses(
                self.easy_token_map,
                desc_prefix="Easy tokens"
            )
            
            if self.is_main_process:
                logger.info(f"Computed original losses for {len(self.easy_original_losses)} easy tokens")

    def _analyze_layer_sensitivity_for_tokens(
        self,
        layer_idx: int,
        token_map: Dict[int, List[int]],
        original_losses: Dict[Tuple[int, int], float],
        desc_prefix: str = "Layer"
    ) -> Optional[LayerSensitivity]:
        """
        通用方法：分析指定层对特定 token 集合的敏感性
        
        Args:
            layer_idx: 层索引
            token_map: token 索引映射
            original_losses: 原始 loss 字典
            desc_prefix: 描述前缀
            
        Returns:
            LayerSensitivity 对象（只在主进程）
        """
        layers = self._get_model_layers()
        target_layer = layers[layer_idx]
        
        samples_to_process = sorted(token_map.keys())
        
        # 多卡：分配样本
        if self.config.use_distributed:
            samples_per_rank = len(samples_to_process) // self.config.world_size
            start_idx = self.config.rank * samples_per_rank
            end_idx = start_idx + samples_per_rank if self.config.rank < self.config.world_size - 1 else len(samples_to_process)
            samples_to_process = samples_to_process[start_idx:end_idx]
        
        local_perturbed_losses = {}
        
        # 注册扰动 hook
        def perturbation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                perturbed_hidden_states = (1 + self.config.perturbation_scale) * hidden_states
                return (perturbed_hidden_states,) + output[1:]
            else:
                return (1 + self.config.perturbation_scale) * output
        
        hook = target_layer.register_forward_hook(perturbation_hook)
        
        with torch.no_grad():
            desc = f"{desc_prefix} {layer_idx}"
            if self.config.use_distributed:
                desc = f"[Rank {self.config.rank}] " + desc
            
            iterator = tqdm(samples_to_process, desc=desc, leave=False) if (self.is_main_process or not self.config.use_distributed) else samples_to_process
            
            for sample_idx in iterator:
                if sample_idx >= len(self.dataset_texts):
                    continue
                
                text = self.dataset_texts[sample_idx]
                
                if not text or len(text.strip()) == 0:
                    continue
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs.input_ids)
                
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = inputs.input_ids[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                token_indices = token_map[sample_idx]
                for token_idx in token_indices:
                    if token_idx < len(token_losses):
                        loss_val = float(token_losses[token_idx].cpu())
                        # NaN/Inf protection for perturbed losses
                        if not np.isnan(loss_val) and not np.isinf(loss_val):
                            local_perturbed_losses[(sample_idx, token_idx)] = loss_val
                        else:
                            logger.debug(f"Skipping invalid perturbed loss for layer {layer_idx}, token ({sample_idx}, {token_idx}): {loss_val}")
        
        hook.remove()
        
        # 多卡：收集结果
        if self.config.use_distributed:
            all_perturbed_losses = [None] * self.config.world_size
            dist.all_gather_object(all_perturbed_losses, local_perturbed_losses)
            
            if self.is_main_process:
                perturbed_losses = {}
                for losses_dict in all_perturbed_losses:
                    if losses_dict:
                        perturbed_losses.update(losses_dict)
            else:
                perturbed_losses = local_perturbed_losses
        else:
            perturbed_losses = local_perturbed_losses
        
        # 计算 loss 变化（只在主进程）
        if self.is_main_process:
            loss_changes = []
            for key in original_losses.keys():
                if key in perturbed_losses:
                    orig_loss = original_losses[key]
                    pert_loss = perturbed_losses[key]
                    delta_loss = pert_loss - orig_loss
                    # NaN/Inf validation for delta
                    if not np.isnan(delta_loss) and not np.isinf(delta_loss):
                        loss_changes.append(delta_loss)
                    else:
                        logger.debug(f"Skipping invalid delta loss for key {key}: orig={orig_loss}, pert={pert_loss}")
            
            if not loss_changes:
                return LayerSensitivity(
                    layer_idx=layer_idx,
                    sensitivity_score=0.0,
                    avg_loss_change=0.0,
                    max_loss_change=0.0,
                    std_loss_change=0.0,
                    num_hard_tokens=0
                )
            
            loss_changes_array = np.array(loss_changes)
            
            sensitivity = LayerSensitivity(
                layer_idx=layer_idx,
                sensitivity_score=float(np.mean(np.abs(loss_changes_array))),
                avg_loss_change=float(np.mean(loss_changes_array)),
                max_loss_change=float(np.max(np.abs(loss_changes_array))),
                std_loss_change=float(np.std(loss_changes_array)),
                num_hard_tokens=len(loss_changes)
            )
            
            return sensitivity
        else:
            return None

    def analyze_layer_sensitivity(self, layer_idx: int) -> LayerSensitivity:
        """分析单层的敏感性（支持多卡）- 保持向后兼容"""
        return self._analyze_layer_sensitivity_for_tokens(
            layer_idx,
            self.hard_token_map,
            self.original_losses,
            desc_prefix="Hard Layer"
        )
    
    def analyze_layer_sensitivity_easy(self, layer_idx: int) -> Optional[LayerSensitivity]:
        """分析单层对简单 token 的敏感性"""
        if not self.has_easy_tokens:
            return None
        return self._analyze_layer_sensitivity_for_tokens(
            layer_idx,
            self.easy_token_map,
            self.easy_original_losses,
            desc_prefix="Easy Layer"
        )
    
    def run_full_analysis(self) -> Tuple[List[LayerSensitivity], Optional[List[LayerSensitivity]]]:
        """
        Run complete analysis (supports multi-GPU and comparative analysis)
        
        Returns:
            (hard_sensitivities, easy_sensitivities)
            If no easy tokens, easy_sensitivities is None
        """
        if self.is_main_process:
            logger.info("Starting layer sensitivity analysis...")
        
        # 步骤 1: 计算原始 loss（同时计算 hard 和 easy）
        self.compute_original_losses()
        
        # 同步所有进程
        if self.config.use_distributed:
            dist.barrier()
        
        # 步骤 2: 对每一层进行敏感性分析（Hard tokens）
        if self.is_main_process:
            self.layer_sensitivities = []
            iterator = tqdm(range(self.num_layers), desc="Analyzing hard tokens")
        else:
            iterator = range(self.num_layers)
        
        for layer_idx in iterator:
            sensitivity = self.analyze_layer_sensitivity(layer_idx)
            
            if self.is_main_process and sensitivity is not None:
                self.layer_sensitivities.append(sensitivity)
            
            # 同步所有进程
            if self.config.use_distributed:
                dist.barrier()
        
        # 步骤 3: 如果有 easy tokens，也进行分析
        if self.has_easy_tokens:
            if self.is_main_process:
                self.easy_layer_sensitivities = []
                iterator = tqdm(range(self.num_layers), desc="Analyzing easy tokens")
            else:
                iterator = range(self.num_layers)
            
            for layer_idx in iterator:
                sensitivity = self.analyze_layer_sensitivity_easy(layer_idx)
                
                if self.is_main_process and sensitivity is not None:
                    self.easy_layer_sensitivities.append(sensitivity)
                
                # 同步所有进程
                if self.config.use_distributed:
                    dist.barrier()
        
        # 只在主进程排序和返回
        if self.is_main_process:
            self.layer_sensitivities.sort(key=lambda x: x.sensitivity_score, reverse=True)
            logger.info("Hard tokens analysis completed!")
            
            if self.has_easy_tokens:
                self.easy_layer_sensitivities.sort(key=lambda x: x.sensitivity_score, reverse=True)
                logger.info("Easy tokens analysis completed!")
                return self.layer_sensitivities, self.easy_layer_sensitivities
            else:
                return self.layer_sensitivities, None
        else:
            if self.has_easy_tokens:
                return [], []
            else:
                return [], None
    
    def compute_ratio_scores(self) -> List[Dict]:
        """
        Compute hard/easy ratio scores (using NumPy vectorization)
        
        Returns:
            List of results sorted by ratio_score in descending order
        """
        if not self.has_easy_tokens:
            raise ValueError("Cannot compute ratio scores without easy tokens!")
        
        if not self.is_main_process:
            return []
        
        logger.info("Computing ratio scores (hard/easy) using vectorized operations...")
        
        # 创建层索引到敏感性分数的映射
        hard_scores = np.zeros(self.num_layers)
        easy_scores = np.zeros(self.num_layers)
        
        for s in self.layer_sensitivities:
            hard_scores[s.layer_idx] = s.sensitivity_score
        
        for s in self.easy_layer_sensitivities:
            easy_scores[s.layer_idx] = s.sensitivity_score
        
        # 向量化计算比值（加 epsilon 防止除零）
        ratio_scores = hard_scores / (easy_scores + self.config.ratio_eps)
        
        # 构建结果列表
        ratio_results = []
        for layer_idx in range(self.num_layers):
            ratio_results.append({
                "layer_idx": layer_idx,
                "hard_score": float(hard_scores[layer_idx]),
                "easy_score": float(easy_scores[layer_idx]),
                "ratio_score": float(ratio_scores[layer_idx])
            })
        
        # 按 ratio_score 降序排序
        ratio_results.sort(key=lambda x: x["ratio_score"], reverse=True)
        
        logger.info(f"Computed ratio scores for {len(ratio_results)} layers")
        return ratio_results
    
    def get_layer_sensitivity_by_ratio(self) -> List[Dict]:
        """
        Get layer sensitivity ranking based on hard/easy ratio
        ratio_score = hard_score / easy_score
        Higher ratio indicates layer is more sensitive to hard tokens (relative to easy tokens)
        
        Returns:
            List of layer sensitivities sorted by ratio_score in descending order
            
        Note:
            When easy tokens are not available, ratio_score is set to float('inf') to indicate
            that hard token sensitivity cannot be compared. Results are sorted by hard_score.
        """
        if not self.has_easy_tokens:
            logger.warning("No easy tokens provided, falling back to hard sensitivity scores")
            return [
                {
                    "layer_idx": s.layer_idx,
                    "hard_score": s.sensitivity_score,
                    "easy_score": 0.0,
                    "ratio_score": float('inf')  # Cannot compute ratio without easy tokens
                }
                for s in sorted(self.layer_sensitivities, key=lambda x: x.sensitivity_score, reverse=True)
            ]
        
        return self.compute_ratio_scores()
    
    def generate_topk_allocation(
        self,
        min_topk: int = 1,
        max_topk: int = 8,
        allocation_strategy: str = "proportional"
    ) -> Dict[int, int]:
        """生成 Top-K 分配（只在主进程）"""
        if not self.is_main_process:
            return {}
        
        logger.info(f"Generating Top-K allocation (strategy={allocation_strategy})...")
        
        layer_to_sensitivity = {s.layer_idx: s.sensitivity_score for s in self.layer_sensitivities}
        topk_allocation = {}
        
        if allocation_strategy == "proportional":
            sensitivity_scores = np.array([layer_to_sensitivity[i] for i in range(self.num_layers)])
            min_score = sensitivity_scores.min()
            max_score = sensitivity_scores.max()
            
            if max_score > min_score:
                normalized_scores = (sensitivity_scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(sensitivity_scores) * 0.5
            
            for layer_idx in range(self.num_layers):
                topk = int(min_topk + normalized_scores[layer_idx] * (max_topk - min_topk))
                topk_allocation[layer_idx] = topk
        
        elif allocation_strategy == "threshold":
            sensitivity_scores = [layer_to_sensitivity[i] for i in range(self.num_layers)]
            median_sensitivity = np.median(sensitivity_scores)
            
            for layer_idx in range(self.num_layers):
                if layer_to_sensitivity[layer_idx] >= median_sensitivity:
                    topk_allocation[layer_idx] = max_topk
                else:
                    topk_allocation[layer_idx] = min_topk
        
        elif allocation_strategy == "quantile":
            sensitivity_scores = [layer_to_sensitivity[i] for i in range(self.num_layers)]
            q75 = np.percentile(sensitivity_scores, 75)
            q25 = np.percentile(sensitivity_scores, 25)
            
            for layer_idx in range(self.num_layers):
                score = layer_to_sensitivity[layer_idx]
                if score >= q75:
                    topk = max_topk
                elif score <= q25:
                    topk = min_topk
                else:
                    topk = (min_topk + max_topk) // 2
                topk_allocation[layer_idx] = topk
        
        else:
            raise ValueError(f"Unknown allocation strategy: {allocation_strategy}")
        
        allocation_counts = defaultdict(int)
        for topk in topk_allocation.values():
            allocation_counts[topk] += 1
        
        logger.info("Top-K allocation summary:")
        for topk in sorted(allocation_counts.keys()):
            logger.info(f"  Top-{topk}: {allocation_counts[topk]} layers")
        
        return topk_allocation
    
    def save_results(self, topk_allocation: Optional[Dict[int, int]] = None, ratio_results: Optional[List[Dict]] = None):
        """Save results (main process only)"""
        if not self.is_main_process:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细敏感性分数（Hard tokens）
        sensitivity_file = output_dir / "layer_sensitivities.json"
        with open(sensitivity_file, 'w') as f:
            json.dump([asdict(s) for s in self.layer_sensitivities], f, indent=2)
        logger.info(f"Saved hard token sensitivities to {sensitivity_file}")
        
        # 保存 easy tokens 敏感性分数（如果有）
        if self.has_easy_tokens and self.easy_layer_sensitivities:
            easy_sensitivity_file = output_dir / "layer_sensitivities_easy.json"
            with open(easy_sensitivity_file, 'w') as f:
                json.dump([asdict(s) for s in self.easy_layer_sensitivities], f, indent=2)
            logger.info(f"Saved easy token sensitivities to {easy_sensitivity_file}")
        
        # 保存比值结果（如果有）
        if ratio_results:
            ratio_file = output_dir / "layer_sensitivities_ratio.json"
            with open(ratio_file, 'w') as f:
                json.dump(ratio_results, f, indent=2)
            logger.info(f"Saved ratio scores to {ratio_file}")
        
        # 保存 Expert CEI scores（如果有）
        if self.expert_cei_scores:
            # Drop any non-finite CEI scores defensively
            finite_cei_scores = [
                score for score in self.expert_cei_scores
                if np.isfinite(score.cei_score)
            ]
            if len(finite_cei_scores) != len(self.expert_cei_scores):
                logger.warning(f"Filtered out {len(self.expert_cei_scores) - len(finite_cei_scores)} experts with non-finite CEI scores")
                self.expert_cei_scores = finite_cei_scores
        
            cei_file = output_dir / "expert_cei_scores.json"
            cei_data = []
            for score in self.expert_cei_scores:
                score_dict = asdict(score)
                # tokens_handled 可能很大，只保存数量
                score_dict['tokens_handled'] = len(score.tokens_handled)
                cei_data.append(score_dict)
            
            with open(cei_file, 'w') as f:
                json.dump(cei_data, f, indent=2)
            logger.info(f"Saved expert CEI scores to {cei_file}")
        
        # 保存 Top-K 分配
        if topk_allocation:
            topk_file = output_dir / "topk_allocation.json"
            with open(topk_file, 'w') as f:
                json.dump(topk_allocation, f, indent=2)
            logger.info(f"Saved Top-K allocation to {topk_file}")
            
            expert_config_file = output_dir / "expert_config.json"
            expert_config = {str(k): v for k, v in topk_allocation.items()}
            default_topk = int(np.median(list(topk_allocation.values())))
            expert_config["default"] = default_topk
            
            with open(expert_config_file, 'w') as f:
                json.dump(expert_config, f, indent=2)
            logger.info(f"Saved expert config to {expert_config_file}")
        
        # 生成报告
        self._generate_report(output_dir, topk_allocation, ratio_results)
    
    def _generate_report(self, output_dir: Path, topk_allocation: Optional[Dict[int, int]] = None, ratio_results: Optional[List[Dict]] = None):
        """Generate report (main process only)"""
        if not self.is_main_process:
            return
        
        # Generate standard report
        report_file = output_dir / "sensitivity_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Layer Sensitivity Analysis Report\n\n")
            f.write(f"**Model**: {self.config.model_name}\n\n")
            f.write(f"**Dataset**: {self.config.dataset_path or self.config.dataset_name}\n\n")
            f.write(f"**Perturbation Scale**: δ = {self.config.perturbation_scale}\n\n")
            f.write(f"**Hard Tokens**: {len(self.hard_tokens)} tokens from {len(self.hard_token_map)} samples\n\n")
            if self.has_easy_tokens:
                f.write(f"**Easy Tokens**: {len(self.easy_tokens)} tokens from {len(self.easy_token_map)} samples\n\n")
            f.write(f"**Dataset Size**: {len(self.dataset_texts)} texts\n\n")
            
            if self.config.use_distributed:
                f.write(f"**Multi-GPU**: {self.config.world_size} GPUs\n\n")
            
            f.write("## Layer Sensitivity Ranking (Hard Tokens)\n\n")
            f.write("| Rank | Layer | Sensitivity Score | Avg Loss Change | Max Loss Change | Std Dev |\n")
            f.write("|------|-------|-------------------|-----------------|-----------------|----------|\n")
            
            for rank, sensitivity in enumerate(self.layer_sensitivities, 1):
                f.write(f"| {rank} | {sensitivity.layer_idx} | {sensitivity.sensitivity_score:.6f} | "
                       f"{sensitivity.avg_loss_change:+.6f} | {sensitivity.max_loss_change:.6f} | "
                       f"{sensitivity.std_loss_change:.6f} |\n")
            
            # 如果有比值结果，生成比值报告
            if ratio_results:
                f.write("\n## Layer Sensitivity Ratio Ranking (Hard/Easy)\n\n")
                f.write("| Rank | Layer | Hard Score | Easy Score | Ratio Score |\n")
                f.write("|------|-------|------------|------------|-------------|\n")
                
                for rank, result in enumerate(ratio_results, 1):
                    f.write(f"| {rank} | {result['layer_idx']} | {result['hard_score']:.6f} | "
                           f"{result['easy_score']:.6f} | {result['ratio_score']:.6f} |\n")
            
            if topk_allocation:
                f.write("\n## Top-K Allocation Strategy\n\n")
                f.write("| Layer | Sensitivity Score | Recommended Top-K |\n")
                f.write("|-------|-------------------|-------------------|\n")
                
                for layer_idx in range(self.num_layers):
                    sensitivity = next((s for s in self.layer_sensitivities if s.layer_idx == layer_idx), None)
                    topk = topk_allocation.get(layer_idx, "N/A")
                    if sensitivity:
                        f.write(f"| {layer_idx} | {sensitivity.sensitivity_score:.6f} | {topk} |\n")
            
            f.write("\n## Interpretation\n\n")
            f.write("- **Sensitivity Score**: |ΔL_l| - Absolute average loss change\n")
            f.write("- **High Sensitivity**: Critical layer → More experts\n")
            f.write("- **Low Sensitivity**: Less critical → Fewer experts\n")
            if ratio_results:
                f.write("\n### Ratio Score (Hard/Easy)\n\n")
                f.write("The ratio score indicates how much more sensitive a layer is to hard tokens compared to easy tokens:\n")
                f.write("- **Ratio > 1**: Layer is more sensitive to hard tokens (important for difficult samples)\n")
                f.write("- **Ratio ≈ 1**: Layer processes hard and easy tokens similarly\n")
                f.write("- **Ratio < 1**: Layer is less sensitive to hard tokens (unusual, may indicate robustness or redundancy)\n")
                f.write("\nHigher ratio scores suggest layers that are particularly critical for handling difficult samples.\n")
        
        logger.info(f"Generated report at {report_file}")
        
        # Generate comprehensive report if CEI analysis was performed
        if self.expert_cei_scores:
            self._generate_comprehensive_report(output_dir)
    
    def _generate_comprehensive_report(self, output_dir: Path):
        """Generate comprehensive HD-AAV report with layer + expert analysis"""
        if not self.is_main_process:
            return
        
        report_file = output_dir / "comprehensive_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# HD-AAV Comprehensive Analysis Report\n\n")
            f.write(f"**Model**: {self.config.model_name}\n\n")
            f.write(f"**Analysis Date**: Generated automatically\n\n")
            f.write(f"**Hard Tokens**: {len(self.hard_tokens)} tokens from {len(self.hard_token_map)} samples\n\n")
            
            # Summary statistics
            critical_count = sum(1 for e in self.expert_cei_scores if e.interpretation == "critical")
            harmful_count = sum(1 for e in self.expert_cei_scores if e.interpretation == "harmful")
            redundant_count = sum(1 for e in self.expert_cei_scores if e.interpretation == "redundant")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Experts Analyzed**: {len(self.expert_cei_scores)}\n")
            f.write(f"- **🔥 Critical Experts**: {critical_count} (Must Keep)\n")
            f.write(f"- **❌ Harmful Experts**: {harmful_count} (Consider Pruning)\n")
            f.write(f"- **💤 Redundant Experts**: {redundant_count} (Can Prune)\n\n")
            
            # Layer Sensitivity Ranking (top 10)
            f.write("## Layer Sensitivity Ranking (Top 10)\n\n")
            f.write("| Rank | Layer | Sensitivity | Top-K Recommendation |\n")
            f.write("|------|-------|-------------|----------------------|\n")
            
            for rank, sensitivity in enumerate(self.layer_sensitivities[:10], 1):
                score = sensitivity.sensitivity_score
                if score > 0.05:
                    recommendation = "High (4-8)"
                elif score > 0.02:
                    recommendation = "Medium (2-4)"
                else:
                    recommendation = "Low (1-2)"
                
                f.write(f"| {rank} | {sensitivity.layer_idx} | {score:.6f} | {recommendation} |\n")
            
            # Expert CEI Analysis by Layer
            f.write("\n## Expert CEI Analysis\n\n")
            
            # Group experts by layer
            experts_by_layer = defaultdict(list)
            for expert in self.expert_cei_scores:
                experts_by_layer[expert.layer_idx].append(expert)
            
            # Get top 5 most sensitive layers
            top_layers = [s.layer_idx for s in self.layer_sensitivities[:5]]
            
            for layer_idx in top_layers:
                if layer_idx not in experts_by_layer:
                    continue
                
                layer_experts = experts_by_layer[layer_idx]
                layer_sensitivity = next((s for s in self.layer_sensitivities if s.layer_idx == layer_idx), None)
                
                if layer_sensitivity:
                    f.write(f"\n### Layer {layer_idx} (Sensitivity: {layer_sensitivity.sensitivity_score:.6f})\n\n")
                else:
                    f.write(f"\n### Layer {layer_idx}\n\n")
                
                f.write("| Expert | CEI Score | Interpretation | Num Tokens | Avg Weight | Action |\n")
                f.write("|--------|-----------|----------------|------------|------------|--------|\n")
                
                # Sort experts by CEI score (descending)
                layer_experts.sort(key=lambda e: e.cei_score, reverse=True)
                
                for expert in layer_experts:
                    # Determine action
                    if expert.interpretation == "critical":
                        action = "✅ KEEP"
                        emoji = "🔥"
                    elif expert.interpretation == "harmful":
                        action = "🗑️ PRUNE"
                        emoji = "❌"
                    else:
                        action = "⚠️ PRUNE"
                        emoji = "💤"
                    
                    f.write(f"| {expert.expert_idx} | {expert.cei_score:+.6f} | {emoji} {expert.interpretation.capitalize()} | "
                           f"{expert.num_tokens} | {expert.avg_routing_weight:.4f} | {action} |\n")
                
                # Recommendation for this layer
                critical_experts = [e.expert_idx for e in layer_experts if e.interpretation == "critical"]
                prune_experts = [e.expert_idx for e in layer_experts if e.interpretation in ["harmful", "redundant"]]
                
                f.write(f"\n**Recommendation for Layer {layer_idx}**:\n")
                if critical_experts:
                    f.write(f"- Keep experts: {critical_experts}\n")
                if prune_experts:
                    f.write(f"- Consider pruning: {prune_experts}\n")
                f.write("\n")
            
            # Interpretation guide
            f.write("\n## Interpretation Guide\n\n")
            f.write("### CEI Score\n")
            f.write("- **CEI > 0.01**: 🔥 **Critical** - Expert significantly reduces loss for hard tokens. Must keep!\n")
            f.write("- **-0.001 < CEI < 0.01**: 💤 **Redundant** - Expert has minimal impact. Safe to prune.\n")
            f.write("- **CEI < -0.001**: ❌ **Harmful** - Expert increases loss. Consider pruning.\n\n")
            
            f.write("### Layer Sensitivity\n")
            f.write("- **High Sensitivity (> 0.05)**: Critical layer, needs more experts (Top-K = 4-8)\n")
            f.write("- **Medium Sensitivity (0.02-0.05)**: Moderate importance (Top-K = 2-4)\n")
            f.write("- **Low Sensitivity (< 0.02)**: Less critical (Top-K = 1-2)\n\n")
            
            f.write("### Recommended Actions\n")
            f.write("1. **Prune redundant and harmful experts** to reduce model size\n")
            f.write("2. **Allocate more Top-K budget** to high-sensitivity layers\n")
            f.write("3. **Keep all critical experts** identified by CEI analysis\n")
            f.write("4. **Validate pruning** on a held-out test set before deployment\n")
        
        logger.info(f"Generated comprehensive report at {report_file}")
    
    def _get_layer_router(self, layer):
        """
        Get the router module from a layer
        
        Args:
            layer: Model layer object
            
        Returns:
            Router module or None if not found
        """
        # Try common MoE architecture patterns
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            # Mixtral/Qwen MoE style
            return layer.mlp.gate
        elif hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
            # Deepseek MoE style
            return layer.block_sparse_moe.gate
        elif hasattr(layer, 'moe') and hasattr(layer.moe, 'gate'):
            # Generic MoE style
            return layer.moe.gate
        elif hasattr(layer, 'router'):
            # Direct router naming
            return layer.router
        else:
            logger.warning(f"Cannot find router in layer. Available attributes: {dir(layer)}")
            return None
    
    def _collect_routing_info_for_sample(self, layer_idx: int, sample_idx: int):
        """
        Collect routing information for a single sample
        
        Args:
            layer_idx: Layer index
            sample_idx: Sample index
        """
        if sample_idx >= len(self.dataset_texts):
            return
        
        text = self.dataset_texts[sample_idx]
        if not text or len(text.strip()) == 0:
            return
        
        layers = self._get_model_layers()
        target_layer = layers[layer_idx]
        router = self._get_layer_router(target_layer)
        
        if router is None:
            logger.warning(f"No router found for layer {layer_idx}, skipping routing collection")
            return
        
        # Temporary storage for current sample's routing information
        temp_routing_info = defaultdict(list)
        temp_routing_weights = defaultdict(list)
        
        # Get hard token positions for current sample
        hard_token_positions = set(self.hard_token_map.get(sample_idx, []))
        
        def routing_collection_hook(module, input, output):
            """Hook to collect routing information for current sample"""
            try:
                # DeepSeek-V2's gate outputs (topk_idx, topk_weight) tuple
                # where topk_idx is int64 expert indices, topk_weight is float routing weights
                if isinstance(output, tuple) and len(output) >= 2:
                    # Check if it's (topk_idx, topk_weight) format
                    first_elem = output[0]
                    second_elem = output[1]
                    
                    # If first element is integer type and second is float type, it's DeepSeek-V2 format
                    if first_elem.dtype in (torch.long, torch.int, torch.int32, torch.int64) and \
                       second_elem.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float):
                        # DeepSeek-V2 format: (topk_idx, topk_weight)
                        top_indices = first_elem  # [batch*seq, top_k]
                        top_weights = second_elem  # [batch*seq, top_k]
                        
                        # Flatten if 3D
                        if top_indices.dim() == 3:
                            batch_size, seq_len, top_k = top_indices.shape
                            top_indices = top_indices.view(-1, top_k)
                            top_weights = top_weights.view(-1, top_k)
                        
                        for token_idx in range(len(top_indices)):
                            if token_idx in hard_token_positions:
                                expert_indices = top_indices[token_idx].cpu().tolist()
                                weights = top_weights[token_idx].cpu().tolist()
                                
                                for expert_idx, weight in zip(expert_indices, weights):
                                    temp_routing_info[expert_idx].append((sample_idx, token_idx))
                                    temp_routing_weights[expert_idx].append(weight)
                    else:
                        # Other model format: router_logits
                        router_logits = first_elem
                        
                        if router_logits.dim() == 3:
                            batch_size, seq_len, num_experts = router_logits.shape
                            router_logits = router_logits.view(-1, num_experts)
                        
                        top_k = getattr(self.model.config, 'num_experts_per_tok',
                                       getattr(self.model.config, 'experts_per_token', 4))
                        top_k = min(top_k, router_logits.shape[-1])
                        
                        # Ensure float type
                        if router_logits.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                            router_logits = router_logits.float()
                        
                        router_weights = F.softmax(router_logits, dim=-1)
                        top_weights, top_indices = torch.topk(router_weights, k=top_k, dim=-1)
                        
                        for token_idx in range(len(top_indices)):
                            if token_idx in hard_token_positions:
                                expert_indices = top_indices[token_idx].cpu().tolist()
                                weights = top_weights[token_idx].cpu().tolist()
                                
                                for expert_idx, weight in zip(expert_indices, weights):
                                    temp_routing_info[expert_idx].append((sample_idx, token_idx))
                                    temp_routing_weights[expert_idx].append(weight)
                else:
                    # Non-tuple output
                    router_logits = output
                    
                    if router_logits.dim() == 3:
                        batch_size, seq_len, num_experts = router_logits.shape
                        router_logits = router_logits.view(-1, num_experts)
                    
                    top_k = getattr(self.model.config, 'num_experts_per_tok',
                                   getattr(self.model.config, 'experts_per_token', 4))
                    top_k = min(top_k, router_logits.shape[-1])
                    
                    if router_logits.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                        router_logits = router_logits.float()
                    
                    router_weights = F.softmax(router_logits, dim=-1)
                    top_weights, top_indices = torch.topk(router_weights, k=top_k, dim=-1)
                    
                    for token_idx in range(len(top_indices)):
                        if token_idx in hard_token_positions:
                            expert_indices = top_indices[token_idx].cpu().tolist()
                            weights = top_weights[token_idx].cpu().tolist()
                            
                            for expert_idx, weight in zip(expert_indices, weights):
                                temp_routing_info[expert_idx].append((sample_idx, token_idx))
                                temp_routing_weights[expert_idx].append(weight)
            
            except Exception as e:
                logger.warning(f"Error in routing collection hook: {e}")
            
            return output  # Important: return output unchanged
        
        hook = router.register_forward_hook(routing_collection_hook)
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True
                ).to(self.device)
                
                _ = self.model(**inputs)
            
            # Append to global routing info (instead of overwriting)
            for expert_idx, tokens in temp_routing_info.items():
                self.routing_info[layer_idx][expert_idx].extend(tokens)
            for expert_idx, weights in temp_routing_weights.items():
                self.routing_weights[layer_idx][expert_idx].extend(weights)
        
        finally:
            hook.remove()
    
    def compute_expert_cei_scores_batch(self, layer_idx: int) -> List[ExpertCEIScore]:
        """
        Batch compute CEI scores for all experts in this layer
        
        Core idea:
        1. Get all tokens handled by all experts at once
        2. Ablate each expert one by one, compute loss only for relevant tokens
        3. Batch processing to avoid redundant tokenization and forward passes
        
        Args:
            layer_idx: Layer index
            
        Returns:
            List of ExpertCEIScore objects
        """
        if not self.is_main_process:
            return []
        
        if self.num_experts is None:
            logger.warning("Model does not have MoE experts, skipping CEI analysis")
            return []
        
        logger.info(f"Computing CEI scores for layer {layer_idx}...")
        
        cei_scores = []
        layers = self._get_model_layers()
        target_layer = layers[layer_idx]
        router = self._get_layer_router(target_layer)
        
        if router is None:
            logger.warning(f"No router found for layer {layer_idx}, skipping CEI analysis")
            return []
        
        for expert_idx in range(self.num_experts):
            # Find hard tokens handled by this expert
            tokens_handled = self.routing_info[layer_idx].get(expert_idx, [])
            
            if len(tokens_handled) == 0:
                logger.debug(f"Expert {expert_idx} in layer {layer_idx} handles no hard tokens, skipping")
                continue
            
            # Smart sampling: if too many tokens, sample to reduce computation
            if len(tokens_handled) > self.config.cei_sample_limit:
                import random
                tokens_handled = random.sample(tokens_handled, self.config.cei_sample_limit)
                logger.info(f"Sampled {self.config.cei_sample_limit} tokens for expert {expert_idx} in layer {layer_idx}")
            
            # Create ablation hook: set this expert's weight to 0
            def ablation_hook(module, input, output):
                """Counterfactual reasoning: assume expert e is absent"""
                try:
                    if isinstance(output, tuple) and len(output) >= 2:
                        first_elem = output[0]
                        second_elem = output[1]
                        
                        # DeepSeek-V2 format: (topk_idx, topk_weight)
                        if first_elem.dtype in (torch.long, torch.int, torch.int32, torch.int64) and \
                           second_elem.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float):
                            topk_idx = first_elem.clone()
                            topk_weight = second_elem.clone()
                            
                            # Set target expert's weight to 0 and renormalize
                            # Find positions of the target expert
                            mask = (topk_idx == expert_idx)
                            topk_weight = topk_weight.masked_fill(mask, 0.0)
                            
                            # Renormalize weights
                            weight_sum = topk_weight.sum(dim=-1, keepdim=True)
                            weight_sum = torch.clamp(weight_sum, min=1e-6)  # Avoid division by zero
                            topk_weight = topk_weight / weight_sum
                            
                            if len(output) > 2:
                                return (topk_idx, topk_weight) + output[2:]
                            else:
                                return (topk_idx, topk_weight)
                        else:
                            # Other model format: router_logits
                            router_logits = first_elem
                            other_outputs = output[1:]
                            
                            if router_logits.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                                router_logits = router_logits.float()
                            
                            ablated_logits = router_logits.clone()
                            ablated_logits[..., expert_idx] = float('-inf')
                            
                            return (ablated_logits,) + other_outputs
                    else:
                        # Non-tuple output
                        router_logits = output
                        
                        if router_logits.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                            router_logits = router_logits.float()
                        
                        ablated_logits = router_logits.clone()
                        ablated_logits[..., expert_idx] = float('-inf')
                        
                        return ablated_logits
                
                except Exception as e:
                    logger.warning(f"Error in ablation hook: {e}")
                    return output  # Return original output on error
            
            hook = router.register_forward_hook(ablation_hook)
            
            # 只计算受影响的 samples 的 loss
            ablated_losses = {}
            affected_samples = set(t[0] for t in tokens_handled)
            
            try:
                with torch.no_grad():
                    for sample_idx in affected_samples:
                        if sample_idx >= len(self.dataset_texts):
                            continue
                        
                        text = self.dataset_texts[sample_idx]
                        if not text or len(text.strip()) == 0:
                            continue
                        
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            max_length=self.config.max_length,
                            truncation=True
                        ).to(self.device)
                        
                        outputs = self.model(**inputs, labels=inputs.input_ids)
                        
                        # 计算 token-level loss
                        shift_logits = outputs.logits[:, :-1, :].contiguous()
                        shift_labels = inputs.input_ids[:, 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        token_losses = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # 只记录相关 tokens
                        for (s_idx, token_idx) in tokens_handled:
                            if s_idx == sample_idx and token_idx < len(token_losses):
                                ablated_losses[(sample_idx, token_idx)] = float(token_losses[token_idx].cpu())
            
            finally:
                hook.remove()
            
            # 计算 CEI score
            cei_values = []
            for token_key in tokens_handled:
                if token_key in self.original_losses and token_key in ablated_losses:
                    original_loss = self.original_losses[token_key]
                    ablated_loss = ablated_losses[token_key]
                    # CEI = loss_without_expert - loss_with_expert
                    # 正值 = 没有这个专家，loss 上升（救火队员）
                    cei_values.append(ablated_loss - original_loss)
            
            if not cei_values:
                continue
            
            # 计算平均 CEI score 和解释
            cei_score_value = float(np.mean(cei_values))
            
            # 解释
            if cei_score_value > self.config.cei_critical_threshold:
                interpretation = "critical"
            elif cei_score_value < self.config.cei_harmful_threshold:
                interpretation = "harmful"
            else:
                interpretation = "redundant"
            
            # 计算平均路由权重
            expert_weights = self.routing_weights[layer_idx].get(expert_idx, [])
            avg_weight = float(np.mean(expert_weights)) if expert_weights else 0.0
            
            cei_scores.append(ExpertCEIScore(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                cei_score=cei_score_value,
                num_tokens=len(cei_values),
                avg_routing_weight=avg_weight,
                tokens_handled=tokens_handled,
                interpretation=interpretation
            ))
        
        logger.info(f"Computed CEI scores for {len(cei_scores)} experts in layer {layer_idx}")
        return cei_scores
    
    def compute_expert_cei_scores_batch_fixed(self, layer_idx: int) -> List[ExpertCEIScore]:
        """
        Fixed version: Batch compute CEI scores for all experts in this layer with NaN protection
        
        Fixes for NaN issues:
        1. Filter valid tokens (check if token exists in original_losses)
        2. Add safe weight normalization using torch.where
        3. Add NaN/Inf checks at each computation step
        4. Add debug logging for troubleshooting
        
        Args:
            layer_idx: Layer index
            
        Returns:
            List of ExpertCEIScore objects
        """
        if not self.is_main_process:
            return []
        
        if self.num_experts is None:
            logger.warning("Model does not have MoE experts, skipping CEI analysis")
            return []
        
        logger.info(f"Computing CEI scores (FIXED) for layer {layer_idx}...")
        
        cei_scores = []
        layers = self._get_model_layers()
        target_layer = layers[layer_idx]
        router = self._get_layer_router(target_layer)
        
        if router is None:
            logger.warning(f"No router found for layer {layer_idx}, skipping CEI analysis")
            return []
        
        for expert_idx in range(self.num_experts):
            # Find hard tokens handled by this expert
            tokens_handled = self.routing_info[layer_idx].get(expert_idx, [])
            
            if len(tokens_handled) == 0:
                logger.debug(f"Expert {expert_idx} in layer {layer_idx} handles no hard tokens, skipping")
                continue
            
            # FIX 1: Filter valid tokens (must exist in original_losses and be finite)
            # valid_tokens is a list of tuples, which is compatible with random.sample()
            valid_tokens = [
                t for t in tokens_handled
                if t in self.original_losses and np.isfinite(self.original_losses[t])
            ]
            if len(valid_tokens) == 0:
                logger.warning(f"Expert {layer_idx}.{expert_idx}: No valid tokens (all filtered), skipping")
                continue
            
            logger.debug(f"Expert {layer_idx}.{expert_idx}: {len(valid_tokens)}/{len(tokens_handled)} valid tokens")
            
            # Smart sampling: if too many tokens, sample to reduce computation
            if len(valid_tokens) > self.config.cei_sample_limit:
                valid_tokens = random.sample(valid_tokens, self.config.cei_sample_limit)
                logger.info(f"Sampled {self.config.cei_sample_limit} tokens for expert {expert_idx} in layer {layer_idx}")
            
            # Create ablation hook: set this expert's weight to 0
            def ablation_hook(module, input, output):
                """Counterfactual reasoning: assume expert e is absent"""
                try:
                    if isinstance(output, tuple) and len(output) >= 2:
                        first_elem = output[0]
                        second_elem = output[1]
                        
                        # DeepSeek-V2 format: (topk_idx, topk_weight)
                        if first_elem.dtype in (torch.long, torch.int, torch.int32, torch.int64) and \
                           second_elem.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float):
                            topk_idx = first_elem.clone()
                            topk_weight = second_elem.clone()
                            
                            # Set target expert's weight to 0
                            mask = (topk_idx == expert_idx)
                            topk_weight = topk_weight.masked_fill(mask, 0.0)
                            
                            # FIX 2: Safe weight normalization using torch.where
                            weight_sum = topk_weight.sum(dim=-1, keepdim=True)
                            valid_mask = weight_sum > MIN_WEIGHT_THRESHOLD
                            
                            # If weight_sum is effectively zero (all mass on this expert), skip ablation to avoid NaNs
                            if not valid_mask.any():
                                return output
                            
                            topk_weight = torch.where(
                                valid_mask,
                                topk_weight / weight_sum.clamp(min=MIN_WEIGHT_THRESHOLD),
                                topk_weight
                            )
                            
                            if len(output) > 2:
                                return (topk_idx, topk_weight) + output[2:]
                            else:
                                return (topk_idx, topk_weight)
                        else:
                            # Other model format: router_logits
                            router_logits = first_elem
                            other_outputs = output[1:]
                            
                            if router_logits.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                                router_logits = router_logits.float()
                            
                            ablated_logits = router_logits.clone()
                            # Use large negative instead of -inf to avoid NaN downstream
                            ablated_logits[..., expert_idx] = -1e4
                            
                            return (ablated_logits,) + other_outputs
                    else:
                        # Non-tuple output
                        router_logits = output
                        
                        if router_logits.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                            router_logits = router_logits.float()
                        
                        ablated_logits = router_logits.clone()
                        ablated_logits[..., expert_idx] = -1e4
                        
                        return ablated_logits
                
                except Exception as e:
                    logger.warning(f"Error in ablation hook: {e}")
                    return output  # Return original output on error
            
            hook = router.register_forward_hook(ablation_hook)
            
            # Compute ablated losses for affected samples
            ablated_losses = {}
            affected_samples = set(t[0] for t in valid_tokens)
            
            try:
                with torch.no_grad():
                    for sample_idx in affected_samples:
                        if sample_idx >= len(self.dataset_texts):
                            logger.warning(f"Sample {sample_idx} out of range (dataset size: {len(self.dataset_texts)})")
                            continue
                        
                        text = self.dataset_texts[sample_idx]
                        if not text or len(text.strip()) == 0:
                            logger.warning(f"Sample {sample_idx} has empty text")
                            continue
                        
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            max_length=self.config.max_length,
                            truncation=True
                        ).to(self.device)
                        
                        outputs = self.model(**inputs, labels=inputs.input_ids)
                        
                        # Calculate token-level loss
                        shift_logits = outputs.logits[:, :-1, :].contiguous()
                        shift_labels = inputs.input_ids[:, 1:].contiguous()
                        
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                        token_losses = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Record relevant tokens with validation
                        for (s_idx, token_idx) in valid_tokens:
                            if s_idx == sample_idx and token_idx < len(token_losses):
                                loss_val = float(token_losses[token_idx].cpu())
                                
                                # FIX 3: NaN/Inf check
                                token_key = (sample_idx, token_idx)
                                if np.isnan(loss_val) or np.isinf(loss_val):
                                    # Fall back to original loss to avoid losing this token
                                    fallback = self.original_losses.get(token_key)
                                    if fallback is not None and np.isfinite(fallback):
                                        ablated_losses[token_key] = float(fallback)
                                        logger.debug(f"Using fallback original loss for {token_key}: {fallback}")
                                    else:
                                        logger.warning(f"Invalid loss for token {s_idx}.{token_idx}: {loss_val}")
                                    continue
                                
                                ablated_losses[token_key] = loss_val
            
            finally:
                hook.remove()
            
            # Calculate CEI scores with validation
            cei_values = []
            for token_key in valid_tokens:
                if token_key in self.original_losses and token_key in ablated_losses:
                    original_loss = self.original_losses[token_key]
                    ablated_loss = ablated_losses[token_key]
                    
                    # Skip any non-finite inputs that slipped through earlier
                    if not np.isfinite(original_loss) or not np.isfinite(ablated_loss):
                        continue
                    
                    # CEI = loss_without_expert - loss_with_expert
                    cei = ablated_loss - original_loss
                    
                    # FIX 3: Validate CEI value
                    if not np.isnan(cei) and not np.isinf(cei):
                        cei_values.append(cei)
                    else:
                        logger.debug(f"Invalid CEI for token {token_key}: orig={original_loss}, ablated={ablated_loss}")
            
            # FIX 4: Check for empty cei_values
            if not cei_values:
                logger.warning(f"Expert {layer_idx}.{expert_idx}: No valid CEI values (filtered: {len(valid_tokens)}, computed: {len(ablated_losses)})")
                continue
            
            # Calculate mean CEI score with an extra nan filter
            cei_array = np.asarray(cei_values, dtype=np.float64)
            finite_mask = np.isfinite(cei_array)
            if not finite_mask.any():
                logger.warning(f"Expert {layer_idx}.{expert_idx}: All CEI values non-finite after filtering")
                continue
            
            cei_score_value = float(np.mean(cei_array[finite_mask]))
            
            # FIX 3: Validate final CEI score
            if np.isnan(cei_score_value) or np.isinf(cei_score_value):
                logger.warning(f"Expert {layer_idx}.{expert_idx}: Invalid mean CEI score: {cei_score_value}")
                continue
            
            # Interpret CEI score
            if cei_score_value > self.config.cei_critical_threshold:
                interpretation = "critical"
            elif cei_score_value < self.config.cei_harmful_threshold:
                interpretation = "harmful"
            else:
                interpretation = "redundant"
            
            # Calculate average routing weight
            expert_weights = self.routing_weights[layer_idx].get(expert_idx, [])
            avg_weight = float(np.mean(expert_weights)) if expert_weights else 0.0
            
            cei_scores.append(ExpertCEIScore(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                cei_score=cei_score_value,
                num_tokens=len(cei_values),
                avg_routing_weight=avg_weight,
                tokens_handled=valid_tokens,
                interpretation=interpretation
            ))
            
            logger.debug(f"Expert {layer_idx}.{expert_idx}: CEI={cei_score_value:.6f}, tokens={len(cei_values)}, interpretation={interpretation}")
        
        logger.info(f"Computed CEI scores for {len(cei_scores)} experts in layer {layer_idx}")
        return cei_scores
    
    def run_comprehensive_analysis(self) -> Tuple[List[LayerSensitivity], List[ExpertCEIScore]]:
        """
        Complete layer sensitivity + expert CEI analysis in one pass
        
        Workflow:
        1. Compute original losses for hard tokens (easy tokens computed separately later)
        2. Per-layer analysis:
           a. Compute layer sensitivity for hard tokens
           b. Collect routing information (if CEI enabled)
           c. Batch compute CEI scores for all experts in this layer
        3. Compute easy token layer sensitivity (if available)
        4. Compute ratio scores (hard/easy) if easy tokens available
        5. Generate comprehensive report
        
        Time complexity: O(2N) vs original O(N×E)
        Speedup: E/2 (assuming 8 experts = 4x)
        
        Returns:
            (layer_sensitivities, expert_cei_scores)
        """
        if self.is_main_process:
            logger.info("=" * 80)
            logger.info("Starting Comprehensive HD-AAV Analysis")
            logger.info("Layer Sensitivity + Expert CEI Score")
            if self.has_easy_tokens:
                logger.info("With Easy Tokens Comparison (Ratio Scores)")
            logger.info("=" * 80)
        
        # Step 1: Compute original loss (both hard and easy)
        self.compute_original_losses()
        
        # Sync all processes
        if self.config.use_distributed:
            dist.barrier()
        
        # Step 2: Per-layer analysis (hard tokens)
        if self.is_main_process:
            self.layer_sensitivities = []
            self.expert_cei_scores = []
            iterator = tqdm(range(self.num_layers), desc="Analyzing Hard Tokens")
        else:
            iterator = range(self.num_layers)
        
        for layer_idx in iterator:
            # 2a. Compute layer sensitivity for hard tokens
            sensitivity = self.analyze_layer_sensitivity(layer_idx)
            
            if self.is_main_process and sensitivity is not None:
                self.layer_sensitivities.append(sensitivity)
            
            # 2b. If CEI enabled, collect routing info and compute CEI scores
            if self.config.enable_cei_analysis and self.is_main_process:
                # Collect routing information
                if self.is_main_process:
                    logger.debug(f"Collecting routing info for layer {layer_idx}...")
                
                samples_to_process = sorted(self.hard_token_map.keys())
                for sample_idx in samples_to_process:
                    self._collect_routing_info_for_sample(layer_idx, sample_idx)
                
                # Compute CEI scores for all experts in this layer (using fixed version)
                layer_cei_scores = self.compute_expert_cei_scores_batch_fixed(layer_idx)
                self.expert_cei_scores.extend(layer_cei_scores)
            
            # Sync all processes
            if self.config.use_distributed:
                dist.barrier()
        
        # Step 3: Compute easy tokens layer sensitivity (if available)
        if self.has_easy_tokens and self.is_main_process:
            logger.info("=" * 80)
            logger.info("Analyzing Easy Tokens for Ratio Calculation")
            logger.info("=" * 80)
            
            self.easy_layer_sensitivities = []
            iterator = tqdm(range(self.num_layers), desc="Analyzing Easy Tokens")
            
            for layer_idx in iterator:
                sensitivity = self.analyze_layer_sensitivity_easy(layer_idx)
                if sensitivity is not None:
                    self.easy_layer_sensitivities.append(sensitivity)
            
            # Compute ratio scores
            logger.info("Computing ratio scores (hard/easy)...")
            self.ratio_results = self.compute_ratio_scores()
            
            logger.info("Easy tokens analysis and ratio calculation completed!")
        
        # 只在主进程排序和返回
        if self.is_main_process:
            self.layer_sensitivities.sort(key=lambda x: x.sensitivity_score, reverse=True)
            logger.info("Hard tokens layer sensitivity analysis completed!")
            
            if self.has_easy_tokens:
                self.easy_layer_sensitivities.sort(key=lambda x: x.sensitivity_score, reverse=True)
                logger.info("Easy tokens layer sensitivity analysis completed!")
                
                # Log top ratio scores
                if self.ratio_results:
                    logger.info("\nTop 5 Layers by Ratio (Hard/Easy):")
                    for i, r in enumerate(self.ratio_results[:5], 1):
                        logger.info(f"  {i}. Layer {r['layer_idx']}: {r['ratio_score']:.6f} "
                                  f"(hard={r['hard_score']:.6f}, easy={r['easy_score']:.6f})")
            
            if self.config.enable_cei_analysis:
                logger.info(f"\nExpert CEI analysis completed! Total experts analyzed: {len(self.expert_cei_scores)}")
                
                # 统计 CEI 结果
                critical_count = sum(1 for e in self.expert_cei_scores if e.interpretation == "critical")
                harmful_count = sum(1 for e in self.expert_cei_scores if e.interpretation == "harmful")
                redundant_count = sum(1 for e in self.expert_cei_scores if e.interpretation == "redundant")
                
                logger.info(f"  Critical experts: {critical_count}")
                logger.info(f"  Harmful experts: {harmful_count}")
                logger.info(f"  Redundant experts: {redundant_count}")
            
            return self.layer_sensitivities, self.expert_cei_scores
        else:
            return [], []
    
    def _get_model_layers(self):
        """获取模型层"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'layers'):
            return self.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Cannot find model layers")
    
    def cleanup(self):
        """清理分布式环境"""
        if self.config.use_distributed:
            dist.destroy_process_group()
