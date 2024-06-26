from dataclasses import dataclass
from typing import Any, Dict
from collections import OrderedDict
import torch

@dataclass
class GPTConfig:
    model_type:str = "gpt2"
    n_layer: int = None
    n_head: int = None
    n_embed: int = None
    vocab_size: int = 50257
    block_size: int = 1024
    embed_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-5
    weight_decay: float = 0.1

@dataclass
class TrainerConfig:
    max_epochs: int = None
    grad_norm_clip: float = None
    snapshot_path: str = None
    save_every: int = None
    use_amp: bool = None

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int

@dataclass
class DataConfig:
    path: str = None
    block_size: int = None
    train_split: float = None
    truncate: float = 1.0
    batch_size: int = None
    data_loader_workers: int = None
