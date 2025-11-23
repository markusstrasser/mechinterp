"""Create inventory of downloaded checkpoints for 3-way comparison."""

import json
from pathlib import Path
from collections import defaultdict

def inventory_run(run_id: str, checkpoint_dir: Path):
    """Get inventory of checkpoints for a run."""
    run_dir = checkpoint_dir / run_id

    if not run_dir.exists():
        return None

    checkpoints = []

    for ckpt_dir in sorted(run_dir.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        model_path = ckpt_dir / "model.pt"
        metadata_path = ckpt_dir / "metadata.json"

        if not model_path.exists():
            continue

        # Load metadata if available
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        step = metadata.get('steps', metadata.get('step', 'unknown'))
        test_acc = metadata.get('test_acc', 'unknown')
        train_loss = metadata.get('train_loss', 'unknown')

        checkpoints.append({
            'label': ckpt_dir.name,
            'step': step,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'path': str(model_path)
        })

    # Sort by step if available
    checkpoints.sort(key=lambda x: x['step'] if isinstance(x['step'], int) else 999999)

    return checkpoints


if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints")

    runs = {
        "5bon0t2j": "No grokking (WD=2.0, 27% final)",
        "e332cujg": "Almost grokked (WD=3.0, 98.4%, MPS)",
        "1z2q8rx3": "Full grokking (WD=3.0, 100%, CPU, ultra-dense)"
    }

    print("=" * 80)
    print("CHECKPOINT INVENTORY - 3-WAY COMPARISON")
    print("=" * 80)

    inventory = {}

    for run_id, description in runs.items():
        print(f"\n{run_id}: {description}")
        print("-" * 80)

        checkpoints = inventory_run(run_id, checkpoint_dir)

        if checkpoints is None:
            print(f"  ⚠️  No checkpoints found")
            continue

        inventory[run_id] = {
            'description': description,
            'checkpoints': checkpoints,
            'count': len(checkpoints)
        }

        print(f"  Total checkpoints: {len(checkpoints)}\n")

        # Show key checkpoints
        if len(checkpoints) <= 15:
            # Show all
            for ckpt in checkpoints:
                step = ckpt['step']
                acc = ckpt['test_acc']
                label = ckpt['label']
                if isinstance(acc, float):
                    print(f"    {label:25s} step={step:6d}, acc={acc:.4f}")
                else:
                    print(f"    {label:25s} step={step}")
        else:
            # Show first 5, last 5, and any labeled ones
            labeled = [c for c in checkpoints if not c['label'].startswith('v')]
            unlabeled = [c for c in checkpoints if c['label'].startswith('v')]

            print(f"  Named checkpoints ({len(labeled)}):")
            for ckpt in labeled[:10]:
                step = ckpt['step']
                acc = ckpt['test_acc']
                label = ckpt['label']
                if isinstance(acc, float):
                    print(f"    {label:25s} step={step:6d}, acc={acc:.4f}")
                else:
                    print(f"    {label:25s} step={step}")

            if len(unlabeled) > 0:
                print(f"\n  Version checkpoints: {len(unlabeled)} (v0 through v{len(unlabeled)-1})")
                # Show step range
                if unlabeled:
                    first_step = unlabeled[0]['step']
                    last_step = unlabeled[-1]['step']
                    print(f"    Step range: {first_step} to {last_step}")

    # Save inventory
    output_file = checkpoint_dir / "checkpoint_inventory.json"
    with open(output_file, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for run_id, data in inventory.items():
        print(f"  {run_id}: {data['count']:3d} checkpoints - {data['description']}")

    print(f"\n📄 Full inventory saved to: {output_file}")
    print("\n✅ All checkpoints available locally - ready for analysis!")
