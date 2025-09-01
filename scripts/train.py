import argparse
import yaml
from pathlib import Path
import sys
import wandb

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import create_model
from src.runner import train
from src.types import TrainConfig

def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return TrainConfig(**config_dict)

def main():
    parser = argparse.ArgumentParser(description='Train a model and save checkpoints.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML file.')
    args = parser.parse_args()

    config = load_config(args.config)

    wandb_run = None
    if config.wandb_project:
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config),
            name=f"p{config.p}_d{config.d_model}_seed{config.seed}"
        )

    model = create_model(config)
    train(config, model, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.finish()

    print("\n--- Training Run Complete ---")

if __name__ == "__main__":
    main()