import torch, os, json
from src.model import DEFAULT_CONFIG, train

if __name__ == "__main__":
    # Local run preserves old behavior for convenience.
    cfg={**DEFAULT_CONFIG, "steps": 10001}
    m, his = train(cfg)
    state = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    # History from model_setup.py is numeric lists already; safe to return.
    SAVE_ROOT = "runs"
    run_name = f"p{cfg['p']}_d{cfg['d_model']}_seed{cfg['seed']}"
    run_dir = os.path.join(SAVE_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    base = os.path.join(run_dir, run_name)
    torch.save(state, base + ".pt")
    try:
        json.dump(his, open(base + ".history.json", "w"))
    except Exception as e:
        print("JSON save failed:", e)

