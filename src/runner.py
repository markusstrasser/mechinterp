# In src/runner.py

import time
import torch
from torch import nn
from typing import List, Callable
from transformer_lens import HookedTransformer

from .data import generate_dataset
from .types import TrainConfig


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Add wandb as an argument with a default value
def train(config: TrainConfig, model: HookedTransformer, probes: List[Callable] = None, wandb_run=None):
    """
    Main training loop with probe system and W&B artifact integration.
    """
    train_data, train_labels, test_data, test_labels = generate_dataset(config)

    starttime = time.time()
    torch.manual_seed(config["seed"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    history = {} # Keep history for local return value
    loss_fn = nn.CrossEntropyLoss()

    for step in range(config["steps"]):
        model.train()
        logits = model(train_data)[:, -1, :]
        loss = loss_fn(logits, train_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config['eval_interval'] == 0:
            metrics = {"train_loss": loss.item()}

            if probes:
                probe_kwargs = {
                    "model": model, "test_data": test_data, "test_labels": test_labels,
                    "config": config, "step": step
                }
                for probe_func in probes:
                    try:
                        metrics.update(probe_func(**probe_kwargs))
                    except Exception as e:
                        print(f"Probe {probe_func.__name__} failed: {e}")

            if wandb_run:
                plottable_metrics = {k: v for k, v in flatten_dict(metrics).items() if isinstance(v, (int, float))}
                wandb_run.log(plottable_metrics, step=step)

            checkpoint_interval = config.get("checkpoint_interval")
            if wandb_run and checkpoint_interval and (step % checkpoint_interval == 0 or step == config["steps"] - 1):
                artifact = wandb_run.Artifact(
                    name=f"model-{wandb_run.id}",
                    type="model",
                    metadata={"step": step, **config}
                )
                model_path = f"checkpoint_step_{step}.pt"
                torch.save(model.state_dict(), model_path)
                artifact.add_file(model_path)
                wandb_run.log_artifact(artifact, aliases=[f"step_{step}", "latest"])
                print(f"Saved W&B artifact for step {step}")
            # --- END NEW ---

            # Store metrics in local history
            for key, value in metrics.items():
                if key not in history: history[key] = []
                history[key].append(value)

            elapsed = time.time() - starttime
            print(
                f"Step {step:6d} | Time {elapsed:8.2f}s | "
                f"Train Loss {metrics.get('train_loss', 0):.4f} | "
                f"Test Acc {metrics.get('test_acc', 0):.4f}"
            )

    return model, history