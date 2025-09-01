from typing import TypedDict, List, Optional


class TrainConfig(TypedDict):
    # Model architecture
    p: int  # Prime for modular arithmetic
    n_layers: int
    n_heads: int
    d_model: int
    d_ffn: int
    
    # Training parameters
    lr: float
    weight_decay: float
    steps: int
    eval_interval: int
    seed: int
    device: str
    
    # Dataset parameters
    n_examples: int
    frac_train: float
    
    # Optional fields
    probes: List[str]  # List of probe names to run
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    checkpoint_interval: Optional[int]


type ProbeOutput = dict[str, float | str | list[int]]
