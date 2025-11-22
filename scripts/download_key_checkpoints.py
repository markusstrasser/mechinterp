"""Download key model checkpoints for circuit analysis."""

import wandb
import json
from pathlib import Path


def download_artifact_by_version(entity: str, project: str, run_id: str, version: int, output_dir: Path):
    """Download an artifact by version number."""
    api = wandb.Api()

    artifact_name = f"model-{run_id}:v{version}"
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")

        # Get step from metadata
        step = artifact.metadata.get('test_acc', 'unknown')  # We'll figure out step from position
        test_acc = artifact.metadata.get('test_acc', 0)

        # Download
        download_path = output_dir / run_id / f"v{version}"
        artifact_dir = artifact.download(root=str(download_path))

        # Save metadata
        with open(download_path / 'metadata.json', 'w') as f:
            json.dump(artifact.metadata, f, indent=2)

        print(f"✅ v{version}: test_acc={test_acc:.4f} -> {download_path}")
        return download_path, artifact.metadata
    except Exception as e:
        print(f"❌ Failed to download v{version}: {e}")
        return None, None


if __name__ == "__main__":
    import sys

    entity = sys.argv[1] if len(sys.argv) > 1 else "discoelysium-neuromatch"
    project = "mod-arith-grokking"

    # Load comparison
    with open('wandb_data/comparison_metadata.json') as f:
        comparison = json.load(f)

    grokked_run_id = comparison['grokked']['run_id']
    almost_run_id = comparison['almost']['run_id']

    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print(f"DOWNLOADING GROKKED RUN CHECKPOINTS: {grokked_run_id}")
    print("="*80)

    # Download key versions for grokked run
    # Based on inspection: v0=step100, then checkpoints at intervals
    # Looking for: early (v0), mid-training (v10), just before grok, at grok, final
    # The run has 23 versions (v0-v22), grokked at step 39600

    # Let's download: v0 (early), v10 (mid), v20, v21, v22 (around grokking)
    grokked_versions = [0, 5, 10, 15, 20, 21, 22]  # Sample across training
    print(f"Downloading versions: {grokked_versions}")

    grokked_checkpoints = {}
    for v in grokked_versions:
        path, metadata = download_artifact_by_version(entity, project, grokked_run_id, v, output_dir)
        if metadata:
            grokked_checkpoints[f"v{v}"] = {
                'path': str(path),
                'test_acc': metadata.get('test_acc', 0),
                'train_loss': metadata.get('train_loss', 0),
                'metadata': metadata,
            }

    print("\n" + "="*80)
    print(f"DOWNLOADING ALMOST RUN CHECKPOINTS: {almost_run_id}")
    print("="*80)

    # Download versions for almost run (mainly final state for comparison)
    almost_versions = [0, 10, 22]  # Early, mid, final
    print(f"Downloading versions: {almost_versions}")

    almost_checkpoints = {}
    for v in almost_versions:
        path, metadata = download_artifact_by_version(entity, project, almost_run_id, v, output_dir)
        if metadata:
            almost_checkpoints[f"v{v}"] = {
                'path': str(path),
                'test_acc': metadata.get('test_acc', 0),
                'train_loss': metadata.get('train_loss', 0),
                'metadata': metadata,
            }

    # Save checkpoint manifest
    manifest = {
        'grokked': {
            'run_id': grokked_run_id,
            'checkpoints': grokked_checkpoints,
        },
        'almost': {
            'run_id': almost_run_id,
            'checkpoints': almost_checkpoints,
        }
    }

    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "="*80)
    print("✅ CHECKPOINT DOWNLOAD COMPLETE!")
    print(f"Checkpoints saved to: {output_dir.absolute()}")
    print(f"Manifest saved to: {output_dir / 'manifest.json'}")
    print("="*80)
