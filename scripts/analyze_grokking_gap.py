"""Analyze what happens in the 5000-step gap between v23 and v24."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the history
df = pd.read_csv('wandb_data/history_e332cujg.csv')

# Focus on the grokking region (step 30000 to 35000)
grok_region = df[(df['_step'] >= 28000) & (df['_step'] <= 37000)].copy()

print("="*80)
print("ANALYZING THE GROKKING GAP (v23 to v24)")
print("="*80)
print(f"\nv23 checkpoint: step 30000, test_acc = {df[df['_step'] == 30000]['test_acc'].values[0]:.4f}")
print(f"v24 checkpoint: step 35000, test_acc = {df[df['_step'] == 35000]['test_acc'].values[0]:.4f}")
print(f"\nBUT... the actual grokking happens INSIDE this gap:")

# Find the exact grokking moment
pre_grok = df[df['test_acc'] < 0.5].iloc[-1]
post_grok = df[df['test_acc'] > 0.9].iloc[0]

print(f"\n  Pre-grok:  step {int(pre_grok['_step'])}, acc={pre_grok['test_acc']:.4f}")
print(f"  Post-grok: step {int(post_grok['_step'])}, acc={post_grok['test_acc']:.4f}")
print(f"  Gap: {int(post_grok['_step'] - pre_grok['_step'])} steps")

# Show the trajectory through the gap
print(f"\n{'Step':>6} {'Test Acc':>10} {'Train Loss':>12} {'Fourier':>10} {'Circulant':>10}")
print("-"*60)
for _, row in grok_region.iterrows():
    if row['_step'] % 500 == 0 or (31000 <= row['_step'] <= 33000 and row['_step'] % 100 == 0):
        print(f"{int(row['_step']):6d} {row['test_acc']:10.4f} {row['train_loss']:12.6f} "
              f"{row.get('fourier_sparsity', 0):10.4f} {row.get('circulant_score', 0):10.6f}")

# Plot the detailed trajectory
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Test accuracy
axes[0, 0].plot(grok_region['_step'], grok_region['test_acc'], 'o-', markersize=3)
axes[0, 0].axvline(x=30000, color='red', linestyle='--', alpha=0.5, label='v23')
axes[0, 0].axvline(x=35000, color='green', linestyle='--', alpha=0.5, label='v24')
axes[0, 0].axvline(x=pre_grok['_step'], color='orange', linestyle=':', alpha=0.7, label='Actual grok start')
axes[0, 0].axvline(x=post_grok['_step'], color='purple', linestyle=':', alpha=0.7, label='Actual grok end')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Test Accuracy')
axes[0, 0].set_title('Test Accuracy (The Grokking Gap)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Train loss
axes[0, 1].plot(grok_region['_step'], grok_region['train_loss'], 'o-', markersize=3)
axes[0, 1].axvline(x=30000, color='red', linestyle='--', alpha=0.5)
axes[0, 1].axvline(x=35000, color='green', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Train Loss (log scale)')
axes[0, 1].set_title('Train Loss')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# Fourier sparsity
if 'fourier_sparsity' in grok_region.columns:
    axes[1, 0].plot(grok_region['_step'], grok_region['fourier_sparsity'], 'o-', markersize=3)
    axes[1, 0].axvline(x=30000, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=35000, color='green', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Fourier Sparsity')
    axes[1, 0].set_title('Fourier Sparsity (Algorithmic Structure)')
    axes[1, 0].grid(True, alpha=0.3)

# Circulant score
if 'circulant_score' in grok_region.columns:
    axes[1, 1].plot(grok_region['_step'], grok_region['circulant_score'], 'o-', markersize=3)
    axes[1, 1].axvline(x=30000, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=35000, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Circulant Score')
    axes[1, 1].set_title('Circulant Score (Modular Structure)')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'wandb_data/grokking_gap_detail.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"\n✅ Saved detailed plot to {output_file}")

# Analysis: what correlates with grokking?
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Compare metrics before and after grokking
pre_grok_window = df[(df['_step'] >= 28000) & (df['_step'] < 31600)]
post_grok_window = df[(df['_step'] > 32300) & (df['_step'] <= 36000)]

metrics = ['train_loss', 'fourier_sparsity', 'circulant_score', 'neuron_specialization']
print(f"\n{'Metric':<25} {'Pre-Grok Mean':>15} {'Post-Grok Mean':>15} {'Change':>15}")
print("-"*70)
for metric in metrics:
    if metric in df.columns:
        pre_mean = pre_grok_window[metric].mean()
        post_mean = post_grok_window[metric].mean()
        change = post_mean - pre_mean
        print(f"{metric:<25} {pre_mean:15.6f} {post_mean:15.6f} {change:+15.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The actual grokking transition happens between steps 31600-32300 (700 steps).
Our checkpoints v23 (step 30000) and v24 (step 35000) bracket this, but don't
capture the exact moment.

WHAT WE HAVE:
✅ Before state (v23): Model still memorizing (27% test acc)
✅ After state (v24): Model has generalized (98% test acc)
✅ Metrics every 100 steps showing the transition

WHAT WE'RE MISSING:
❌ Checkpoint at step 31600 (last moment before grokking)
❌ Checkpoint at step 32300 (first moment after grokking)
❌ Weight evolution during the 700-step transition

RECOMMENDATION:
1. Analyze v23 vs v24 for macro-level changes (what we can do now)
2. If you want the exact moment, rerun with denser checkpoints around step 32000
3. Consider multiple seeds to see if grokking is robust
""")
