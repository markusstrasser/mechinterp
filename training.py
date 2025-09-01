from model_setup import train
import torch
import json
import os

# Best score (small)
# d_ffn: 512
# d_model: 64
# lr: 0.001
# seed: 42
# weight_decay: 1 # or 2.5

config = {
    'p': 113, #SHOULD BE PRIME NUMBER!
    'n_examples': 113*113,
    'frac_train': 0.3,
    'n_layers': 1,
    'n_head': 3,
    'd_model': 32,
    'd_ffn': 128,
    'lr': 1e-3,
    'weight_decay': 4,
    'steps': 60001,
    'eval_interval': 1000,
    'seed': 43,
    'device': "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Training with device: {config['device']}")
SAVE_ROOT = "runs"
run_name = f"p{config['p']}_d{config['d_model']}_seed{config['seed']}"
run_dir = os.path.join(SAVE_ROOT, run_name)
os.makedirs(run_dir, exist_ok=True)
base_path = os.path.join(run_dir, run_name)

MODEL_PATH = base_path + ".pt"
HISTORY_PATH = base_path + ".history.pt"
HISTORY_JSON_PATH = base_path + ".history.json"


if __name__ == "__main__":
    model, history = train(config)
    print("--- Training Finished ---")
    print("History object is available:", history is not None)
    print("Available history keys:", list(history.keys()))
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model weights saved to: {MODEL_PATH}")
    torch.save(history, HISTORY_PATH)
    print(f"✅ History dictionary saved to: {HISTORY_PATH}")

    try:
        json.dump(history, open(HISTORY_JSON_PATH, "w"), default=lambda o: (
            o.detach().cpu().tolist() if isinstance(o, torch.Tensor) else o
        ))
        print(f"✅ History (JSON) saved to: {HISTORY_JSON_PATH}")
    except Exception as e:
        print(f"⚠️ Could not save history JSON: {e}")

    # Example (optional) load-back usage:
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=config['device']))
    # loaded_history = torch.load(HISTORY_PATH)
    # model.eval()
