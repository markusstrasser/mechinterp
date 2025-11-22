"""Visualize the best grokking vs almost-grokking comparison."""

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path


# Best runs based on analysis:
# GROKKED: 15a8sppw (wd=3, grokked at step 39600, max_acc=0.9927)
# ALMOST: 5bon0t2j (wd=2, max_acc=0.2712) - actually didn't grok at all
# Alternative: zv7e09pl (wd=1, max_acc=0.3035) - also didn't grok

grokked_id = "15a8sppw"  # or "e332cujg"
almost_id = "5bon0t2j"   # wd=2

df_grokked = pd.read_csv(f"wandb_data/history_{grokked_id}.csv")
df_almost = pd.read_csv(f"wandb_data/history_{almost_id}.csv")

# Load metadata
with open('wandb_data/runs_summary.json') as f:
    all_runs = json.load(f)
    grokked_meta = next(r for r in all_runs if r['run_id'] == grokked_id)
    almost_meta = next(r for r in all_runs if r['run_id'] == almost_id)

print(f"GROKKED RUN: {grokked_meta['run_name']} (wd={grokked_meta['config']['weight_decay']})")
print(f"  Steps: {len(df_grokked)}")
print(f"  Max test acc: {df_grokked['test_acc'].max():.4f}")
print(f"  Final train loss: {df_grokked['train_loss'].iloc[-1]:.6f}")

print(f"\nALMOST RUN: {almost_meta['run_name']} (wd={almost_meta['config']['weight_decay']})")
print(f"  Steps: {len(df_almost)}")
print(f"  Max test acc: {df_almost['test_acc'].max():.4f}")
print(f"  Final train loss: {df_almost['train_loss'].iloc[-1]:.6f}")

# Create comprehensive comparison plot
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Row 1: Test acc and train loss
axes[0, 0].plot(df_grokked['_step'], df_grokked['test_acc'], label=f"Grokked (wd=3)", linewidth=2)
axes[0, 0].plot(df_almost['_step'], df_almost['test_acc'], label=f"Almost (wd=2)", linewidth=2)
axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Grokking threshold')
axes[0, 0].set_xlabel('Training Step')
axes[0, 0].set_ylabel('Test Accuracy')
axes[0, 0].set_title('Test Accuracy: Grokking vs Almost-Grokking')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df_grokked['_step'], df_grokked['train_loss'], label=f"Grokked (wd=3)", linewidth=2)
axes[0, 1].plot(df_almost['_step'], df_almost['train_loss'], label=f"Almost (wd=2)", linewidth=2)
axes[0, 1].set_xlabel('Training Step')
axes[0, 1].set_ylabel('Train Loss (log scale)')
axes[0, 1].set_title('Training Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Row 2: Fourier metrics (algorithmic structure)
if 'fourier_sparsity' in df_grokked.columns:
    axes[1, 0].plot(df_grokked['_step'], df_grokked['fourier_sparsity'], label='Grokked', linewidth=2)
    axes[1, 0].plot(df_almost['_step'], df_almost['fourier_sparsity'], label='Almost', linewidth=2)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Fourier Sparsity')
    axes[1, 0].set_title('Fourier Sparsity (Higher = More Periodic Structure)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

if 'circulant_score' in df_grokked.columns:
    axes[1, 1].plot(df_grokked['_step'], df_grokked['circulant_score'], label='Grokked', linewidth=2)
    axes[1, 1].plot(df_almost['_step'], df_almost['circulant_score'], label='Almost', linewidth=2)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Circulant Score')
    axes[1, 1].set_title('Circulant Score (Circular Convolution Structure)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

# Row 3: Logit attribution (mechanism)
if 'logit_attribution.mlp' in df_grokked.columns:
    axes[2, 0].plot(df_grokked['_step'], df_grokked['logit_attribution.mlp'],
                    label='Grokked MLP', linewidth=2, linestyle='-')
    axes[2, 0].plot(df_grokked['_step'], df_grokked['logit_attribution.attn'],
                    label='Grokked Attn', linewidth=2, linestyle='--')
    axes[2, 0].plot(df_almost['_step'], df_almost['logit_attribution.mlp'],
                    label='Almost MLP', linewidth=2, linestyle='-', alpha=0.7)
    axes[2, 0].plot(df_almost['_step'], df_almost['logit_attribution.attn'],
                    label='Almost Attn', linewidth=2, linestyle='--', alpha=0.7)
    axes[2, 0].set_xlabel('Training Step')
    axes[2, 0].set_ylabel('Logit Attribution')
    axes[2, 0].set_title('MLP vs Attention Attribution')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

if 'neuron_specialization' in df_grokked.columns:
    axes[2, 1].plot(df_grokked['_step'], df_grokked['neuron_specialization'],
                    label='Grokked', linewidth=2)
    axes[2, 1].plot(df_almost['_step'], df_almost['neuron_specialization'],
                    label='Almost', linewidth=2)
    axes[2, 1].set_xlabel('Training Step')
    axes[2, 1].set_ylabel('Neuron Specialization')
    axes[2, 1].set_title('Neuron Specialization (Lower = More Specialized)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wandb_data/grokking_comparison.png', dpi=200, bbox_inches='tight')
print(f"\n✅ Saved plot to wandb_data/grokking_comparison.png")

# Also create a focused plot on just the key grokking moment
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(df_grokked['_step'], df_grokked['test_acc'], label=f"Grokked (wd=3)", linewidth=3, color='green')
ax.plot(df_almost['_step'], df_almost['test_acc'], label=f"Didn't Grok (wd=2)", linewidth=3, color='red')
ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Grokking threshold (0.9)')

# Mark the grokking moment
grok_step = df_grokked[df_grokked['test_acc'] > 0.9]['_step'].iloc[0]
ax.axvline(x=grok_step, color='green', linestyle=':', alpha=0.5, linewidth=2)
ax.text(grok_step, 0.5, f'Grokking at step {grok_step}', rotation=90, va='bottom')

ax.set_xlabel('Training Step', fontsize=14)
ax.set_ylabel('Test Accuracy', fontsize=14)
ax.set_title('The Grokking Phenomenon: wd=3 vs wd=2', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('wandb_data/grokking_moment.png', dpi=200, bbox_inches='tight')
print(f"✅ Saved plot to wandb_data/grokking_moment.png")

# Save comparison metadata
with open('wandb_data/comparison_metadata.json', 'w') as f:
    json.dump({
        'grokked': {
            'run_id': grokked_id,
            'run_name': grokked_meta['run_name'],
            'weight_decay': grokked_meta['config']['weight_decay'],
            'max_test_acc': float(df_grokked['test_acc'].max()),
            'grok_step': int(grok_step),
            'final_train_loss': float(df_grokked['train_loss'].iloc[-1]),
            'config': grokked_meta['config'],
        },
        'almost': {
            'run_id': almost_id,
            'run_name': almost_meta['run_name'],
            'weight_decay': almost_meta['config']['weight_decay'],
            'max_test_acc': float(df_almost['test_acc'].max()),
            'final_train_loss': float(df_almost['train_loss'].iloc[-1]),
            'config': almost_meta['config'],
        }
    }, f, indent=2)
print(f"✅ Saved metadata to wandb_data/comparison_metadata.json")
