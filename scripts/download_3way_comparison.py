"""Download checkpoints for 3-way comparison: no grok, 98% grok, 100% grok."""

import wandb
import json
from pathlib import Path


def download_artifact_by_step(entity: str, project: str, run_id: str, step: int, label: str, output_dir: Path):
    """Download checkpoint at specific step from WandB."""
    api = wandb.Api()

    # Try to find the artifact for this step
    # WandB artifacts are named like "model-{run_id}:v{version}"
    # We need to iterate through versions to find the one at the desired step

    try:
        # Get the run
        run = api.run(f"{entity}/{project}/{run_id}")

        # List all artifacts for this run
        artifacts = list(run.logged_artifacts())

        print(f"\n🔍 Looking for step {step} in run {run_id}...")
        print(f"   Found {len(artifacts)} total artifacts")

        # Find artifact closest to desired step
        best_artifact = None
        best_step_diff = float('inf')

        for artifact in artifacts:
            if artifact.type != 'model':
                continue

            metadata = artifact.metadata
            artifact_step = metadata.get('steps', metadata.get('step', 0))

            step_diff = abs(artifact_step - step)
            if step_diff < best_step_diff:
                best_step_diff = step_diff
                best_artifact = artifact

        if best_artifact is None:
            print(f"❌ No model artifacts found for {run_id}")
            return None, None

        actual_step = best_artifact.metadata.get('steps', best_artifact.metadata.get('step', 0))
        test_acc = best_artifact.metadata.get('test_acc', 0)

        print(f"   Best match: step {actual_step} (requested {step}), test_acc={test_acc:.4f}")

        # Download
        download_path = output_dir / run_id / label
        download_path.mkdir(parents=True, exist_ok=True)

        artifact_dir = best_artifact.download(root=str(download_path))

        # Save metadata
        with open(download_path / 'metadata.json', 'w') as f:
            json.dump(best_artifact.metadata, f, indent=2)

        print(f"✅ {label:30s} step={actual_step:6d}, test_acc={test_acc:.4f}")
        return download_path, best_artifact.metadata

    except Exception as e:
        print(f"❌ Failed to download {label} from {run_id}: {e}")
        return None, None


if __name__ == "__main__":
    import sys

    entity = sys.argv[1] if len(sys.argv) > 1 else "discoelysium-neuromatch"
    project = "mod-arith-grokking"

    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("DOWNLOADING CHECKPOINTS FOR 3-WAY COMPARISON")
    print("="*80)

    # Define the 3 runs and their key checkpoints
    runs_to_download = {
        "5bon0t2j": {  # No grok (WD=2.0, 27% accuracy)
            "description": "No grokking control (WD=2.0)",
            "checkpoints": [
                (1000, "early"),
                (10000, "mid"),
                (40000, "final"),
            ]
        },
        "e332cujg": {  # 98% grok (WD=3.0, MPS)
            "description": "Almost grokked (98.4%, MPS)",
            "checkpoints": [
                (1000, "early"),
                (10000, "pre_grok"),
                (42000, "just_before_grok"),
                (44000, "grokking_moment"),
                (60000, "final"),
            ]
        },
        "1z2q8rx3": {  # 100% grok (WD=3.0, CPU, ultra-dense)
            "description": "Full grokking (100%, ultra-dense)",
            "checkpoints": [
                (1000, "early"),
                (2000, "pre_grok_1"),
                (3000, "pre_grok_2"),
                (3900, "grokking_moment"),  # Grokked at ~3.9k
                (4000, "post_grok_1"),
                (5000, "post_grok_2"),
                (10000, "stable"),
                (20000, "final"),
            ]
        },
    }

    all_results = {}

    for run_id, run_info in runs_to_download.items():
        print(f"\n{'='*80}")
        print(f"RUN: {run_id} - {run_info['description']}")
        print(f"{'='*80}")

        run_results = {}

        for step, label in run_info["checkpoints"]:
            path, metadata = download_artifact_by_step(
                entity, project, run_id, step, label, output_dir
            )

            if metadata:
                run_results[label] = {
                    'requested_step': step,
                    'actual_step': metadata.get('steps', metadata.get('step', 0)),
                    'path': str(path),
                    'test_acc': metadata.get('test_acc', 0),
                    'train_loss': metadata.get('train_loss', 0),
                    'metadata': metadata,
                }

        all_results[run_id] = {
            'description': run_info['description'],
            'checkpoints': run_results,
        }

    # Save comprehensive manifest
    manifest_file = output_dir / '3way_comparison_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\n📦 Checkpoints saved to: {output_dir.absolute()}")
    print(f"📄 Manifest: {manifest_file}")

    # Summary
    print(f"\n📊 SUMMARY:")
    for run_id, results in all_results.items():
        desc = results['description']
        n_checkpoints = len(results['checkpoints'])
        print(f"   {run_id}: {n_checkpoints} checkpoints - {desc}")
    print()
