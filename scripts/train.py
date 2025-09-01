import sys
import argparse
import yaml
from pathlib import Path
import wandb
import torch

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import create_model
from src.runner import train
from src.probes import PROBE_REGISTRY
from src.types import TrainConfig


def load_config(config_path: str) -> TrainConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default device if not specified
    if 'device' not in config:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train a model on modular arithmetic task')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    args = parser.parse_args()

    config = load_config(args.config)

    # --- MODIFIED: Initialize W&B run object ---
    wandb_run = None
    if not args.no_wandb and 'wandb_project' in config:
        wandb_run = wandb.init(
            project=config.get('wandb_project', 'grokking'),
            entity=config.get('wandb_entity', None),
            config=config,
            name=f"p{config['p']}_d{config['d_model']}_seed{config['seed']}"
        )
    # --- END MODIFIED ---

    probe_names = config.get('probes', [])
    probes = [PROBE_REGISTRY[name] for name in probe_names if name in PROBE_REGISTRY]

    model = create_model(config)

    # --- MODIFIED: Pass the wandb_run object ---
    model, history = train(config, model, probes, wandb_run=wandb_run)
    # --- END MODIFIED ---

    # --- MODIFIED: Remove redundant final save, just finish the run ---
    if wandb_run:
        wandb_run.finish()
    # --- END MODIFIED ---

    print(f"\nTraining complete! Final test accuracy: {history.get('test_acc', [0])[-1]:.4f}")

if __name__ == "__main__":
    main()