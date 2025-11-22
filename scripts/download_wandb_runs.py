"""Download and analyze W&B runs to find best grokking vs almost-grokking comparison."""

import wandb
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any


def download_all_runs(entity: str, project: str = "mod-arith-grokking") -> pd.DataFrame:
    """Download metrics from all runs in the project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    run_data = []
    for run in runs:
        print(f"Processing run: {run.name} ({run.id})")
        print(f"  State: {run.state}")
        print(f"  Config: {run.config}")

        # Get summary metrics (final values)
        summary = run.summary._json_dict

        # Get config
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}

        run_info = {
            'run_id': run.id,
            'run_name': run.name,
            'state': run.state,
            'created_at': run.created_at,
            'config': config,
            'summary': summary,
        }

        # Download history for finished runs
        if run.state == "finished":
            history = run.history(samples=10000)  # Get all data points
            run_info['history'] = history

            # Calculate grokking metrics
            if 'test_acc' in history.columns and 'train_acc' in history.columns:
                max_test_acc = history['test_acc'].max()
                final_test_acc = history['test_acc'].iloc[-1] if len(history) > 0 else 0

                # Check if grokking occurred (test acc > 0.95 at some point)
                grokked = max_test_acc > 0.95

                # Find grokking step (when test acc first crosses 0.9)
                grok_steps = history[history['test_acc'] > 0.9]['_step'].values
                grok_step = int(grok_steps[0]) if len(grok_steps) > 0 else None

                run_info['grokked'] = grokked
                run_info['max_test_acc'] = max_test_acc
                run_info['final_test_acc'] = final_test_acc
                run_info['grok_step'] = grok_step

                print(f"  Grokked: {grokked}, Max test acc: {max_test_acc:.3f}, Grok step: {grok_step}")

        run_data.append(run_info)

    return run_data


def find_best_comparison(run_data: list) -> Dict[str, Any]:
    """Find the best grokking vs almost-grokking comparison.

    Criteria:
    - Minimal model (fewest parameters)
    - One run that grokked (test_acc > 0.95)
    - One run that almost grokked (0.7 < test_acc < 0.95)
    - Similar configs (same architecture, different weight decay or other hyperparams)
    """
    finished_runs = [r for r in run_data if r['state'] == 'finished' and 'grokked' in r]

    grokked_runs = [r for r in finished_runs if r['grokked']]
    almost_grokked_runs = [r for r in finished_runs if not r['grokked'] and r['max_test_acc'] > 0.7]

    print(f"\nFound {len(grokked_runs)} grokked runs")
    print(f"Found {len(almost_grokked_runs)} almost-grokked runs")

    # Find minimal model
    def model_size(run):
        cfg = run['config']
        return cfg.get('d_model', 32) * cfg.get('n_layers', 1) * cfg.get('n_heads', 3)

    if grokked_runs:
        grokked_runs.sort(key=model_size)
        print("\nGrokked runs (sorted by model size):")
        for r in grokked_runs[:5]:
            cfg = r['config']
            print(f"  {r['run_name']}: d_model={cfg.get('d_model')}, layers={cfg.get('n_layers')}, "
                  f"test_acc={r['max_test_acc']:.3f}, grok_step={r['grok_step']}")

    if almost_grokked_runs:
        almost_grokked_runs.sort(key=model_size)
        print("\nAlmost-grokked runs (sorted by model size):")
        for r in almost_grokked_runs[:5]:
            cfg = r['config']
            print(f"  {r['run_name']}: d_model={cfg.get('d_model')}, layers={cfg.get('n_layers')}, "
                  f"test_acc={r['max_test_acc']:.3f}")

    return {
        'grokked_runs': grokked_runs,
        'almost_grokked_runs': almost_grokked_runs,
    }


def save_run_data(run_data: list, output_dir: Path):
    """Save run data to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as JSON
    summary_data = []
    for r in run_data:
        # Don't include history in summary (too large)
        summary = {k: v for k, v in r.items() if k != 'history'}
        summary_data.append(summary)

    with open(output_dir / 'runs_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Save histories as separate CSV files
    for r in run_data:
        if 'history' in r and r['history'] is not None:
            history_file = output_dir / f"history_{r['run_id']}.csv"
            r['history'].to_csv(history_file, index=False)
            print(f"Saved history to {history_file}")


if __name__ == "__main__":
    import sys

    # Get entity from wandb config or environment
    api = wandb.Api()

    # Get entity from command line or use default
    entity = sys.argv[1] if len(sys.argv) > 1 else "discoelysium-neuromatch"

    if not entity:
        print("Error: Please provide your W&B entity name")
        exit(1)

    print(f"Downloading runs from {entity}/mod-arith-grokking...")
    run_data = download_all_runs(entity)

    # Save data
    output_dir = Path("wandb_data")
    save_run_data(run_data, output_dir)

    # Find best comparison
    comparison = find_best_comparison(run_data)

    # Save comparison
    with open(output_dir / 'best_comparison.json', 'w') as f:
        # Remove history before saving
        grokked = [{k: v for k, v in r.items() if k != 'history'}
                   for r in comparison['grokked_runs'][:5]]
        almost = [{k: v for k, v in r.items() if k != 'history'}
                  for r in comparison['almost_grokked_runs'][:5]]
        json.dump({
            'grokked_runs': grokked,
            'almost_grokked_runs': almost,
        }, f, indent=2, default=str)

    print(f"\nData saved to {output_dir}/")
    print(f"  - runs_summary.json: Summary of all runs")
    print(f"  - history_*.csv: Full metrics history for each run")
    print(f"  - best_comparison.json: Best candidates for comparison")
