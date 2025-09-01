from typing import TypedDict

class TrainConfig(TypedDict):
    lr: float
    batch_size: int
    epochs: int
    model_name: str
    seed: int
    device: str
    d_model: int
    n_heads: int
    n_layers: int
    d_ffn: int
    weight_decay: float
    steps: int
    eval_interval: int


type ProbeOutput = dict[str, float | str | list[int]]
