import json
from pathlib import Path
from typing import Dict, Any

import modal
import torch

from scripts.train import DEFAULT_CONFIG, train

app = modal.App("marim")
img = (modal.Image.debian_slim()
       .uv_pip_install("torch==2.8.0", "transformer-lens")
       .add_local_python_source("model_setup")
       )


@app.function(image=img, gpu="B200", enable_memory_snapshot=True)
def run_training(cfg) -> Dict[str, Any]:
    print("STARTING TRAINING", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    model, history = train(cfg)
    #picklable across the wire.
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # History from model_setup.py is numeric lists already; safe to return.
    return {
        "config": cfg,
        "model_state_dict": state,
        "history": history,
        # small convenience: last-step metrics
        "final": {k: history[k][-1] for k in ["steps","train_loss","test_loss","test_acc","l2_norm","gini_embed","gini_unembed"]},
    }


@app.local_entrypoint()
def main():
    res = run_training.remote(DEFAULT_CONFIG)  # returns your dict

    cfg = res["config"]
    run_name = f"p{cfg['p']}_d{cfg['d_model']}_seed{cfg['seed']}"
    run_dir = Path("../runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    base = run_dir / run_name

    torch.save(res["model_state_dict"], str(base.with_suffix(".pt")))
    torch.save(res["history"],         str(base.with_suffix(".history.pt")))
    with open(base.with_suffix(".history.json"), "w") as f:
        json.dump(res["history"], f)

    print("saved:", run_dir)
