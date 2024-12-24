from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class ModelConfig:
    name: str = "t5-base"
    pretrained: bool = True
    base_model: str = "t5-base"
    model_type: str = "t5"
    vocab_size: int = 32128
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_length: int = 512
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01

@dataclass
class DataConfig:
    max_length: int = 512
    batch_size: int = 32
    num_workers: int = 4
    dataset_path: str = "path/to/dataset"
    train_split: float = 0.8
    shuffle: bool = True

@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"