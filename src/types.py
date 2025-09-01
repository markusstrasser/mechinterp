from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class TrainConfig:
    # Core model parameters
    p: int = 113
    n_layers: int = 1
    n_heads: int = 3
    d_model: int = 32
    d_ffn: int = 128

    # Training parameters
    lr: float = 0.001
    weight_decay: float = 1.0
    steps: int = 60001

    # Dataset parameters
    n_examples: int = 12769
    frac_train: float = 0.3

    # Execution parameters
    seed: int = 43
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    # W&B logging and checkpointing
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    eval_interval: int = 1000 # Default value, adjust as needed
    checkpoint_intervals: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 20000, 40001])