import argparse
import tomllib  # Replaces yaml
from pathlib import Path
import sys
import wandb

# Add project root to path FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import checkpoint_name_from_config
from src.model import create_model
from src.train import train
from src.types import TrainConfig

def load_config(path: str) -> TrainConfig:
    """Loads a flat TOML file directly into the TrainConfig dataclass."""
    with open(path, "rb") as f:
        config_dict = tomllib.load(f)
        return TrainConfig(**config_dict)

def main():
    parser = argparse.ArgumentParser(description='Train a model and save checkpoints.')
    parser.add_argument(
        '--config',
        type=str,
        required=True, # No more default, config is mandatory
        help='Path to the TOML configuration file.'
    )
    args = parser.parse_args()

    config = load_config(args.config)

    wandb_run = None
    if config.wandb_project:
        wandb_run = wandb.init(
            project=config.wandb_project,
            config=vars(config),
            name=checkpoint_name_from_config(config, with_extension=False),
        )

    model = create_model(config)
    train(config, model, wandb_run=wandb_run)

    if wandb_run:
        wandb_run.finish()

    print("\n--- Training Run Complete ---")

if __name__ == "__main__":
    main()