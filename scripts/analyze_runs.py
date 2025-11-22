"""Analyze downloaded W&B runs to find best grokking comparison."""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_run(run_id: str, data_dir: Path = Path("wandb_data")) -> dict:
    """Analyze a single run's history."""
    history_file = data_dir / f"history_{run_id}.csv"
    if not history_file.exists():
        return None

    df = pd.read_csv(history_file)

    # Check if test_acc exists
    if 'test_acc' not in df.columns:
        return {
            'run_id': run_id,
            'has_test_acc': False,
            'num_steps': len(df),
        }

    # Calculate grokking metrics
    max_test_acc = df['test_acc'].max()
    final_test_acc = df['test_acc'].iloc[-1] if len(df) > 0 else 0
    final_train_loss = df['train_loss'].iloc[-1] if 'train_loss' in df.columns and len(df) > 0 else None

    # Check if grokking occurred (test acc > 0.95 at some point)
    grokked = max_test_acc > 0.95

    # Find grokking step (when test acc first crosses 0.9)
    grok_steps = df[df['test_acc'] > 0.9]['_step'].values
    grok_step = int(grok_steps[0]) if len(grok_steps) > 0 else None

    # Find almost-grokking characteristics (sustained but incomplete performance)
    # Check for plateau at high accuracy
    if len(df) > 50:
        last_50_mean = df['test_acc'].iloc[-50:].mean()
        last_50_std = df['test_acc'].iloc[-50:].std()
        plateaued = last_50_std < 0.02  # Low variance = plateaued
    else:
        last_50_mean = final_test_acc
        plateaued = False

    return {
        'run_id': run_id,
        'has_test_acc': True,
        'num_steps': len(df),
        'max_test_acc': max_test_acc,
        'final_test_acc': final_test_acc,
        'final_train_loss': final_train_loss,
        'grokked': grokked,
        'grok_step': grok_step,
        'plateaued': plateaued,
        'last_50_mean_acc': last_50_mean,
        'history': df,
    }


def plot_comparison(run1_data: dict, run2_data: dict, output_file: Path):
    """Plot test accuracy comparison between two runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get runs metadata
    with open('wandb_data/runs_summary.json') as f:
        all_runs = json.load(f)
        run1_meta = next(r for r in all_runs if r['run_id'] == run1_data['run_id'])
        run2_meta = next(r for r in all_runs if r['run_id'] == run2_data['run_id'])

    df1 = run1_data['history']
    df2 = run2_data['history']

    # Test accuracy
    axes[0, 0].plot(df1['_step'], df1['test_acc'], label=f"{run1_meta['run_name']} (wd={run1_meta['config']['weight_decay']})")
    axes[0, 0].plot(df2['_step'], df2['test_acc'], label=f"{run2_meta['run_name']} (wd={run2_meta['config']['weight_decay']})")
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Grokking threshold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Test Accuracy Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Train loss
    axes[0, 1].plot(df1['_step'], df1['train_loss'], label=f"{run1_meta['run_name']}")
    axes[0, 1].plot(df2['_step'], df2['train_loss'], label=f"{run2_meta['run_name']}")
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Train Loss')
    axes[0, 1].set_title('Train Loss Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # Fourier sparsity (algorithmic structure indicator)
    if 'fourier_sparsity' in df1.columns and 'fourier_sparsity' in df2.columns:
        axes[1, 0].plot(df1['_step'], df1['fourier_sparsity'], label=f"{run1_meta['run_name']}")
        axes[1, 0].plot(df2['_step'], df2['fourier_sparsity'], label=f"{run2_meta['run_name']}")
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Fourier Sparsity')
        axes[1, 0].set_title('Fourier Sparsity (Algorithmic Structure)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Logit attribution (where computation happens)
    if 'logit_attribution.mlp' in df1.columns and 'logit_attribution.attn' in df1.columns:
        axes[1, 1].plot(df1['_step'], df1['logit_attribution.mlp'], label=f"{run1_meta['run_name']} MLP", linestyle='-')
        axes[1, 1].plot(df1['_step'], df1['logit_attribution.attn'], label=f"{run1_meta['run_name']} Attn", linestyle='--')
        axes[1, 1].plot(df2['_step'], df2['logit_attribution.mlp'], label=f"{run2_meta['run_name']} MLP", linestyle='-')
        axes[1, 1].plot(df2['_step'], df2['logit_attribution.attn'], label=f"{run2_meta['run_name']} Attn", linestyle='--')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Logit Attribution')
        axes[1, 1].set_title('MLP vs Attention Attribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved comparison plot to {output_file}")


if __name__ == "__main__":
    # Load runs summary
    with open('wandb_data/runs_summary.json') as f:
        runs_summary = json.load(f)

    # Analyze all finished runs
    finished_runs = [r for r in runs_summary if r['state'] == 'finished']
    print(f"Found {len(finished_runs)} finished runs\n")

    analyses = []
    for run in finished_runs:
        analysis = analyze_run(run['run_id'])
        if analysis and analysis['has_test_acc']:
            # Add config info
            analysis['config'] = run['config']
            analysis['run_name'] = run['run_name']
            analyses.append(analysis)

            # Print summary
            status = "GROKKED" if analysis['grokked'] else "ALMOST" if analysis['max_test_acc'] > 0.7 else "NO GROK"
            print(f"{status}: {run['run_name']} (id: {run['run_id']})")
            print(f"  Weight decay: {run['config']['weight_decay']}")
            print(f"  Steps: {analysis['num_steps']}")
            print(f"  Max test acc: {analysis['max_test_acc']:.4f}")
            print(f"  Final test acc: {analysis['final_test_acc']:.4f}")
            print(f"  Final train loss: {analysis['final_train_loss']:.6f}")
            if analysis['grokked']:
                print(f"  Grokked at step: {analysis['grok_step']}")
            if analysis['plateaued']:
                print(f"  Plateaued at: {analysis['last_50_mean_acc']:.4f}")
            print()

    # Find best grokking vs almost-grokking comparison
    print("\n" + "="*80)
    print("FINDING BEST COMPARISON")
    print("="*80 + "\n")

    grokked_runs = [a for a in analyses if a['grokked']]
    almost_grokked = [a for a in analyses if not a['grokked'] and a['max_test_acc'] > 0.7]

    print(f"Grokked runs: {len(grokked_runs)}")
    for r in grokked_runs:
        print(f"  - {r['run_name']}: max_acc={r['max_test_acc']:.4f}, wd={r['config']['weight_decay']}, steps={r['num_steps']}")

    print(f"\nAlmost-grokked runs: {len(almost_grokked)}")
    for r in almost_grokked:
        print(f"  - {r['run_name']}: max_acc={r['max_test_acc']:.4f}, wd={r['config']['weight_decay']}, steps={r['num_steps']}")

    # Find the most minimal models (smallest architecture)
    def model_size(run):
        cfg = run['config']
        return cfg.get('d_model', 32) * cfg.get('n_layers', 1) * cfg.get('n_heads', 3)

    if grokked_runs:
        best_grokked = min(grokked_runs, key=model_size)
        print(f"\n🎯 BEST GROKKED RUN: {best_grokked['run_name']} ({best_grokked['run_id']})")
        print(f"   Max test acc: {best_grokked['max_test_acc']:.4f}")
        print(f"   Grokked at step: {best_grokked['grok_step']}")
        print(f"   Weight decay: {best_grokked['config']['weight_decay']}")

    if almost_grokked:
        best_almost = min(almost_grokked, key=model_size)
        print(f"\n🎯 BEST ALMOST-GROKKED RUN: {best_almost['run_name']} ({best_almost['run_id']})")
        print(f"   Max test acc: {best_almost['max_test_acc']:.4f}")
        print(f"   Plateaued at: {best_almost['last_50_mean_acc']:.4f}")
        print(f"   Weight decay: {best_almost['config']['weight_decay']}")

    # Plot comparison if we have both
    if grokked_runs and almost_grokked:
        print("\n📊 Generating comparison plot...")
        plot_comparison(best_grokked, best_almost, Path('wandb_data/comparison.png'))

        # Save best runs for later use
        with open('wandb_data/best_runs.json', 'w') as f:
            json.dump({
                'grokked': {
                    'run_id': best_grokked['run_id'],
                    'run_name': best_grokked['run_name'],
                    'max_test_acc': best_grokked['max_test_acc'],
                    'grok_step': best_grokked['grok_step'],
                    'config': best_grokked['config'],
                },
                'almost_grokked': {
                    'run_id': best_almost['run_id'],
                    'run_name': best_almost['run_name'],
                    'max_test_acc': best_almost['max_test_acc'],
                    'last_50_mean_acc': best_almost['last_50_mean_acc'],
                    'config': best_almost['config'],
                }
            }, f, indent=2)
        print("✅ Saved best runs to wandb_data/best_runs.json")
