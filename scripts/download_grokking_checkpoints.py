"""Download checkpoints capturing the grokking moment."""

import wandb
import json
from pathlib import Path


def download_artifact_by_version(entity: str, project: str, run_id: str, version: int, label: str, output_dir: Path):
    """Download an artifact by version number."""
    api = wandb.Api()

    artifact_name = f"model-{run_id}:v{version}"
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
        test_acc = artifact.metadata.get('test_acc', 0)

        # Download
        download_path = output_dir / run_id / label
        artifact_dir = artifact.download(root=str(download_path))

        # Save metadata
        with open(download_path / 'metadata.json', 'w') as f:
            json.dump(artifact.metadata, f, indent=2)

        print(f"✅ {label:20s} (v{version}): test_acc={test_acc:.4f}, path={download_path}")
        return download_path, artifact.metadata
    except Exception as e:
        print(f"❌ Failed to download {label} (v{version}): {e}")
        return None, None


if __name__ == "__main__":
    import sys

    entity = sys.argv[1] if len(sys.argv) > 1 else "discoelysium-neuromatch"
    project = "mod-arith-grokking"

    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)

    # Use e332cujg as the grokked run (has checkpoints at grokking moment)
    grokked_run_id = "e332cujg"
    almost_run_id = "5bon0t2j"

    print("="*80)
    print(f"DOWNLOADING GROKKED RUN: {grokked_run_id}")
    print(f"Grokking happens between v23 (0.27) and v24 (0.98)")
    print("="*80)

    # Download key checkpoints:
    # - v0: Early training
    # - v10: Mid training
    # - v22: Pre-grokking plateau
    # - v23: Just before grokking (0.27)
    # - v24: Right after grokking (0.98) ⭐ THE MOMENT!
    # - v30: Stabilized after grokking
    # - v42: Final

    grokked_downloads = [
        (0, "early"),
        (10, "mid_training"),
        (22, "pre_grok_plateau"),
        (23, "just_before_grok"),  # Last moment before grokking
        (24, "grokking_moment"),    # ⭐ THE GROKKING MOMENT!
        (30, "post_grok_stable"),
        (42, "final"),
    ]

    grokked_checkpoints = {}
    for version, label in grokked_downloads:
        path, metadata = download_artifact_by_version(entity, project, grokked_run_id, version, label, output_dir)
        if metadata:
            grokked_checkpoints[label] = {
                'version': version,
                'path': str(path),
                'test_acc': metadata.get('test_acc', 0),
                'train_loss': metadata.get('train_loss', 0),
                'fourier_sparsity': metadata.get('fourier_sparsity', 0),
                'circulant_score': metadata.get('circulant_score', 0),
                'metadata': metadata,
            }

    print("\n" + "="*80)
    print(f"DOWNLOADING ALMOST-GROKKED RUN: {almost_run_id} (wd=2)")
    print("="*80)

    # Download checkpoints for comparison
    almost_downloads = [
        (0, "early"),
        (10, "mid_training"),
        (22, "final"),  # Final state - plateaued at 0.27
    ]

    almost_checkpoints = {}
    for version, label in almost_downloads:
        path, metadata = download_artifact_by_version(entity, project, almost_run_id, version, label, output_dir)
        if metadata:
            almost_checkpoints[label] = {
                'version': version,
                'path': str(path),
                'test_acc': metadata.get('test_acc', 0),
                'train_loss': metadata.get('train_loss', 0),
                'fourier_sparsity': metadata.get('fourier_sparsity', 0),
                'circulant_score': metadata.get('circulant_score', 0),
                'metadata': metadata,
            }

    # Save manifest
    manifest = {
        'grokked_run': {
            'run_id': grokked_run_id,
            'weight_decay': 3.0,
            'description': 'Successfully grokked from 0.27 to 0.98 between v23 and v24',
            'checkpoints': grokked_checkpoints,
        },
        'almost_run': {
            'run_id': almost_run_id,
            'weight_decay': 2.0,
            'description': 'Plateaued at 0.27 - did not grok',
            'checkpoints': almost_checkpoints,
        },
        'key_comparison': {
            'just_before_grok': f"checkpoints/{grokked_run_id}/just_before_grok",
            'grokking_moment': f"checkpoints/{grokked_run_id}/grokking_moment",
            'almost_final': f"checkpoints/{almost_run_id}/final",
        }
    }

    manifest_file = output_dir / 'grokking_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "="*80)
    print("✅ CHECKPOINT DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\n📦 Checkpoints saved to: {output_dir.absolute()}")
    print(f"📄 Manifest: {manifest_file}")
    print(f"\n🎯 KEY COMPARISON:")
    print(f"   Just before grok (v23): test_acc = {grokked_checkpoints['just_before_grok']['test_acc']:.4f}")
    print(f"   Grokking moment (v24):  test_acc = {grokked_checkpoints['grokking_moment']['test_acc']:.4f}  ⭐")
    print(f"   Almost-grokked final:   test_acc = {almost_checkpoints['final']['test_acc']:.4f}")
    print()
