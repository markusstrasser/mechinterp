import tempfile
import time
import torch
from torch import nn
from transformer_lens import HookedTransformer
import wandb
import os

from .data import generate_dataset
from .types import TrainConfig
from .probes import ProbeContext, trigger_probes
from tabulate import tabulate


def flatten_no_parent_keys(d):
    items = {}
    for v in d.values():
        if isinstance(v, dict):
            items.update(flatten_no_parent_keys(v))
        else:
            items[id(v)] = v  # careful here
    return items
def log_dict(d, with_keys=True, float_fmt=".3f", str_max=10):
    def fmt(v):
        if isinstance(v, float):
            return format(v, float_fmt)
        s = str(v)
        if isinstance(v, str) and len(s) > str_max:
            return s[:str_max-1] + "…"
        return s

    keys = " | ".join(str(k) for k in d.keys())
    vals = " | ".join(fmt(v) for v in d.values())

    if with_keys:
        print(keys)
    print(vals)

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
    #? STEPS are EPOCHS IN THIS case because we're doing FULL BATCH GRADIENT DESCENT
    for step in range(config.steps):
        model.train()
        logits = model(train_data)[:, -1, :]
        loss = loss_fn(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        is_eval = step % config.eval_interval == 0
        is_checkpoint = step in config.checkpoint_intervals
        first_eval = step == 0
        if is_eval:

            step_ctx: ProbeContext = {
                "model": model,
                "config": config,
                "step": step,
                "epoch": step,  # As established, step is equivalent to epoch here
                "test_data": test_data,
                "test_labels": test_labels,
            }

            probe_metrics = trigger_probes('manual', step_ctx)

            log_data = {
                "train_loss": loss.item(),
                **probe_metrics
            }

            log_dict(log_data, with_keys=first_eval)

            if wandb_run:
                wandb_run.log(log_data, step=step)

        if is_checkpoint:
            if wandb_run:
                with tempfile.TemporaryDirectory() as tmpdir:
                    # The local filename is irrelevant, but we can still make it descriptive.
                    model_filename = f"p{config.p}-d{config.d_model}-s{step}.pt"
                    model_path = os.path.join(tmpdir, model_filename)
                    torch.save(model.state_dict(), model_path)

                    artifact = wandb.Artifact(
                        name=f"model-{wandb_run.id}",
                        type="model",
                        metadata={k: getattr(config, k) for k in config.__annotations__}
                    )

                    artifact.add_file(model_path, name="model.pt")
                    wandb_run.log_artifact(artifact, aliases=[f"step_{step}"])

                print(f"--- Saved checkpoint to W&B artifact for step {step} ---")
    return model