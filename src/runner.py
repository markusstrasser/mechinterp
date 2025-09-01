import time
import torch
from torch import nn
from transformer_lens import HookedTransformer

from .data import generate_dataset
from .types import TrainConfig

def train(config: TrainConfig, model: HookedTransformer, wandb_run=None):
    """
    A simplified training loop that only trains and saves checkpoints.
    """
    train_data, train_labels, test_data, test_labels = generate_dataset(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    print("Starting training...")
    for step in range(config.steps):
        model.train()
        logits = model(train_data)[:, -1, :]
        loss = loss_fn(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step:6d} | Train Loss {loss.item():.4f}")
            if wandb_run:
                wandb_run.log({"train_loss": loss.item()}, step=step)

        # --- Checkpointing Logic ---
        if wandb_run and step in config.checkpoint_intervals:
            artifact = wandb_run.Artifact(
                name=f"model-{wandb_run.id}",
                type="model",
                metadata={k: getattr(config, k) for k in config.__annotations__}
            )
            model_path = f"checkpoint_step_{step}.pt"
            torch.save(model.state_dict(), model_path)
            artifact.add_file(model_path)
            wandb_run.log_artifact(artifact, aliases=[f"step_{step}"])
            print(f"--- Saved W&B artifact for step {step} ---")

    return model