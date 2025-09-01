import argparse
import yaml
import torch
import wandb
from pathlib import Path
import sys
import pprint

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import create_model
from src.types import TrainConfig
from src.probes import probes
from src.data import generate_dataset

def main():
    parser = argparse.ArgumentParser(description="Run post-hoc analysis on a trained model artifact.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path (e.g., 'entity/project/artifact:version').")
    args = parser.parse_args()

    print(f"--- Running analysis on artifact: {args.artifact_path} ---")

    # 1. Initialize W&B and download the artifact
    run = wandb.init(job_type="analysis")
    artifact = run.use_artifact(args.artifact_path, type="model")
    artifact_dir = artifact.download()

    # 2. Load config and model state from the artifact
    model_config_dict = artifact.metadata
    config = TrainConfig(**model_config_dict)

    model_path = Path(artifact_dir) / "checkpoint.pt" # Adjust if filename differs
    state_dict = torch.load(model_path)

    model = create_model(config)
    model.load_state_dict(state_dict)

    # 3. Generate the same dataset for evaluation
    _, _, test_data, test_labels = generate_dataset(vars(config))

    # 4. Run all available probes on the loaded model
    all_metrics = {}
    probe_kwargs = {
        "model": model,
        "test_data": test_data,
        "test_labels": test_labels,
        "config": config,
    }

    for name, probe_func in probes.items():
        print(f"Running probe: {name}...")
        try:
            all_metrics.update(probe_func(**probe_kwargs))
        except Exception as e:
            print(f"Probe {name} failed: {e}")

    # 5. Log results and print
    print("\n--- Analysis Complete ---")
    pprint.pprint(all_metrics)

    # Log results to the new analysis run
    run.log(all_metrics)
    run.finish()

if __name__ == "__main__":
    main()