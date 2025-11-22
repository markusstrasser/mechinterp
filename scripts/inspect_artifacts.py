"""Inspect W&B artifacts to understand their structure."""

import wandb


def inspect_artifact(entity: str, project: str, run_id: str, artifact_version: str = "v0"):
    """Inspect a specific artifact."""
    api = wandb.Api()

    artifact_name = f"model-{run_id}:{artifact_version}"
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}")
        print(f"\n📦 Artifact: {artifact.name}")
        print(f"   Type: {artifact.type}")
        print(f"   Created: {artifact.created_at}")
        print(f"   Size: {artifact.size} bytes")
        print(f"   Metadata: {artifact.metadata}")
        print(f"   Files:")
        for file in artifact.files():
            print(f"     - {file.name} ({file.size} bytes)")
        return artifact
    except Exception as e:
        print(f"❌ Error inspecting {artifact_name}: {e}")
        return None


if __name__ == "__main__":
    import sys
    import json

    entity = sys.argv[1] if len(sys.argv) > 1 else "discoelysium-neuromatch"
    project = "mod-arith-grokking"

    # Load comparison
    with open('wandb_data/comparison_metadata.json') as f:
        comparison = json.load(f)

    grokked_run_id = comparison['grokked']['run_id']
    almost_run_id = comparison['almost']['run_id']

    print("="*80)
    print(f"GROKKED RUN: {grokked_run_id}")
    print("="*80)

    # Inspect first few artifacts to understand structure
    for i in range(min(5, 23)):  # First 5 artifacts
        inspect_artifact(entity, project, grokked_run_id, f"v{i}")

    print("\n" + "="*80)
    print(f"ALMOST RUN: {almost_run_id}")
    print("="*80)

    for i in range(min(3, 23)):  # First 3 artifacts
        inspect_artifact(entity, project, almost_run_id, f"v{i}")
