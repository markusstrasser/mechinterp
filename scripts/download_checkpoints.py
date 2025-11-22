"""Download model checkpoints from W&B artifacts."""

import wandb
import json
from pathlib import Path


def download_checkpoint(entity: str, project: str, run_id: str, step: int, output_dir: Path):
    """Download a specific checkpoint from W&B."""
    api = wandb.Api()

    # Get the run
    run = api.run(f"{entity}/{project}/{run_id}")

    # Checkpoints are saved as artifacts with naming convention from utils.checkpoint_name_from_config
    # Format: "checkpoint-{run_name}-step{step}"
    artifact_name = f"checkpoint-{run.name}-step{step}"

    try:
        # Download the artifact
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}:latest", type="model")
        artifact_dir = artifact.download(root=str(output_dir / run_id / f"step_{step}"))
        print(f"✅ Downloaded {artifact_name} to {artifact_dir}")
        return artifact_dir
    except Exception as e:
        print(f"❌ Failed to download {artifact_name}: {e}")
        return None


def list_run_artifacts(entity: str, project: str, run_id: str):
    """List all artifacts for a run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"\nArtifacts for run {run.name} ({run_id}):")
    for artifact in run.logged_artifacts():
        print(f"  - {artifact.name} (type: {artifact.type}, version: {artifact.version})")

    return run


if __name__ == "__main__":
    import sys

    entity = sys.argv[1] if len(sys.argv) > 1 else "discoelysium-neuromatch"
    project = "mod-arith-grokking"

    # Load the best comparison
    with open('wandb_data/comparison_metadata.json') as f:
        comparison = json.load(f)

    grokked_run_id = comparison['grokked']['run_id']
    almost_run_id = comparison['almost']['run_id']
    grok_step = comparison['grokked']['grok_step']

    print(f"Grokked run: {grokked_run_id}")
    print(f"Almost run: {almost_run_id}")
    print(f"Grokking step: {grok_step}")

    # List artifacts for both runs
    print("\n" + "="*80)
    list_run_artifacts(entity, project, grokked_run_id)
    print("\n" + "="*80)
    list_run_artifacts(entity, project, almost_run_id)
    print("\n" + "="*80)

    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)

    # Download key checkpoints for the grokked run
    print("\nDownloading checkpoints for GROKKED run...")

    # Download checkpoints at key moments:
    # 1. Early (step 100) - before grokking
    # 2. Just before grokking (step closest to but before grok_step)
    # 3. At grokking (grok_step)
    # 4. After grokking (final step)

    checkpoints_to_download = comparison['grokked']['config']['checkpoint_intervals']
    print(f"Available checkpoint intervals: {checkpoints_to_download}")

    # Find checkpoints around grokking moment
    before_grok = [s for s in checkpoints_to_download if s < grok_step]
    at_or_after_grok = [s for s in checkpoints_to_download if s >= grok_step]

    key_steps_grokked = [
        100,  # Early
        before_grok[-1] if before_grok else 100,  # Just before grokking
        at_or_after_grok[0] if at_or_after_grok else grok_step,  # At/after grokking
        checkpoints_to_download[-1],  # Final
    ]
    key_steps_grokked = sorted(set(key_steps_grokked))
    print(f"Downloading grokked checkpoints at steps: {key_steps_grokked}")

    for step in key_steps_grokked:
        download_checkpoint(entity, project, grokked_run_id, step, output_dir)

    # Download checkpoints for the almost run (final state primarily)
    print("\nDownloading checkpoints for ALMOST run...")
    almost_checkpoints = comparison['almost']['config']['checkpoint_intervals']
    key_steps_almost = [
        100,  # Early
        almost_checkpoints[-1],  # Final
    ]
    print(f"Downloading almost checkpoints at steps: {key_steps_almost}")

    for step in key_steps_almost:
        download_checkpoint(entity, project, almost_run_id, step, output_dir)

    print("\n✅ Checkpoint download complete!")
    print(f"Checkpoints saved to: {output_dir.absolute()}")
