"""Create focused, insightful circuit diagrams - remove bloat."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def load_checkpoint(checkpoint_path: Path):
    """Load a model checkpoint."""
    model_file = checkpoint_path / "model.pt"
    metadata_file = checkpoint_path / "metadata.json"

    model_state = torch.load(model_file, map_location='cpu')
    with open(metadata_file) as f:
        metadata = json.load(f)

    return model_state, metadata


def create_weight_comparison(before_path: Path, after_path: Path, output_path: Path):
    """Create a focused before/after weight comparison showing what actually changes."""

    model_before, meta_before = load_checkpoint(before_path)
    model_after, meta_after = load_checkpoint(after_path)

    # Extract key weights
    W_E_before = model_before['embed.W_E'].numpy()
    W_E_after = model_after['embed.W_E'].numpy()

    W_U_before = model_before['unembed.W_U'].numpy()
    W_U_after = model_after['unembed.W_U'].numpy()

    W_in_before = model_before['blocks.0.mlp.W_in'].numpy()
    W_in_after = model_after['blocks.0.mlp.W_in'].numpy()

    W_out_before = model_before['blocks.0.mlp.W_out'].numpy()
    W_out_after = model_after['blocks.0.mlp.W_out'].numpy()

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(f'Weight Changes During Grokking\nBefore (acc={meta_before["test_acc"]:.3f}) → After (acc={meta_after["test_acc"]:.3f})',
                 fontsize=16, fontweight='bold')

    # Row 1: Weight magnitudes
    # Embedding
    ax = axes[0, 0]
    im = ax.imshow(W_E_after.T, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_E_after).max(), vmax=np.abs(W_E_after).max())
    ax.set_title('Embedding After Grokking', fontsize=12, fontweight='bold')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax)

    # Unembedding
    ax = axes[0, 1]
    im = ax.imshow(W_U_after, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(W_U_after).max(), vmax=np.abs(W_U_after).max())
    ax.set_title('Unembedding After Grokking', fontsize=12, fontweight='bold')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax)

    # MLP neuron importance (product of in/out norms)
    ax = axes[0, 2]
    neuron_in_norms_before = np.linalg.norm(W_in_before, axis=0)
    neuron_out_norms_before = np.linalg.norm(W_out_before, axis=1)
    importance_before = neuron_in_norms_before * neuron_out_norms_before

    neuron_in_norms_after = np.linalg.norm(W_in_after, axis=0)
    neuron_out_norms_after = np.linalg.norm(W_out_after, axis=1)
    importance_after = neuron_in_norms_after * neuron_out_norms_after

    top_k = 30
    top_neurons = np.argsort(importance_after)[-top_k:]

    x = np.arange(top_k)
    width = 0.35
    ax.barh(x - width/2, importance_before[top_neurons], width, label='Before', alpha=0.7)
    ax.barh(x + width/2, importance_after[top_neurons], width, label='After', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f'N{i}' for i in top_neurons])
    ax.set_xlabel('Weight Norm Product')
    ax.set_title(f'Top {top_k} Neurons Before/After', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Row 2: Weight CHANGES (delta)
    # Embedding change
    ax = axes[1, 0]
    delta_E = W_E_after - W_E_before
    im = ax.imshow(delta_E.T, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(delta_E).max(), vmax=np.abs(delta_E).max())
    ax.set_title('Embedding Change (Δ)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax)

    # Unembedding change
    ax = axes[1, 1]
    delta_U = W_U_after - W_U_before
    im = ax.imshow(delta_U, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(delta_U).max(), vmax=np.abs(delta_U).max())
    ax.set_title('Unembedding Change (Δ)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax)

    # Metrics comparison
    ax = axes[1, 2]
    metrics = ['Test Acc', 'Train Loss', 'Fourier\nSparsity', 'Circulant\nScore']
    before_vals = [
        meta_before['test_acc'],
        meta_before['train_loss'],
        meta_before.get('fourier_sparsity', 0),
        meta_before.get('circulant_score', 0) * 100  # Scale for visibility
    ]
    after_vals = [
        meta_after['test_acc'],
        meta_after['train_loss'],
        meta_after.get('fourier_sparsity', 0),
        meta_after.get('circulant_score', 0) * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, before_vals, width, label='Before', alpha=0.7)
    ax.bar(x + width/2, after_vals, width, label='After', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Value')
    ax.set_title('Metrics Before/After', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'✅ Saved focused comparison to {output_path}')
    plt.close()


def create_embedding_structure(before_path: Path, after_path: Path, output_path: Path):
    """Visualize embedding structure changes - look for periodic patterns."""

    model_before, meta_before = load_checkpoint(before_path)
    model_after, meta_after = load_checkpoint(after_path)

    W_E_before = model_before['embed.W_E'].numpy()  # [114, 32]
    W_E_after = model_after['embed.W_E'].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    fig.suptitle('Embedding Structure: Search for Modular Arithmetic Patterns',
                 fontsize=16, fontweight='bold')

    # Top-left: Embedding similarity matrix (before)
    ax = axes[0, 0]
    # Show token-token similarity (cosine)
    E_norm_before = W_E_before / (np.linalg.norm(W_E_before, axis=1, keepdims=True) + 1e-8)
    sim_before = E_norm_before @ E_norm_before.T
    im = ax.imshow(sim_before[:50, :50], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Token Similarity (Before, acc={meta_before["test_acc"]:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Token ID')
    plt.colorbar(im, ax=ax)

    # Top-right: Embedding similarity matrix (after)
    ax = axes[0, 1]
    E_norm_after = W_E_after / (np.linalg.norm(W_E_after, axis=1, keepdims=True) + 1e-8)
    sim_after = E_norm_after @ E_norm_after.T
    im = ax.imshow(sim_after[:50, :50], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Token Similarity (After, acc={meta_after["test_acc"]:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Token ID')
    plt.colorbar(im, ax=ax)

    # Bottom-left: FFT of first embedding dimension (look for periodicity)
    ax = axes[1, 0]
    # FFT of embedding weights for first dimension
    fft_before = np.abs(np.fft.rfft(W_E_before[:113, 0]))  # Exclude special token
    fft_after = np.abs(np.fft.rfft(W_E_after[:113, 0]))
    freqs = np.fft.rfftfreq(113)

    ax.plot(freqs[1:20], fft_before[1:20], 'o-', label='Before', alpha=0.7)
    ax.plot(freqs[1:20], fft_after[1:20], 'o-', label='After', alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('FFT Magnitude')
    ax.set_title('FFT of Embedding Dim 0\n(Periodic structure = peaks)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Embedding norm per token (which tokens become important?)
    ax = axes[1, 1]
    norms_before = np.linalg.norm(W_E_before, axis=1)
    norms_after = np.linalg.norm(W_E_after, axis=1)

    ax.plot(norms_before[:50], 'o-', alpha=0.7, label='Before', markersize=4)
    ax.plot(norms_after[:50], 'o-', alpha=0.7, label='After', markersize=4)
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Embedding Norm')
    ax.set_title('Which Tokens Become Important?', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'✅ Saved embedding analysis to {output_path}')
    plt.close()


if __name__ == "__main__":
    # Load manifest
    manifest_file = Path("checkpoints/grokking_manifest.json")
    with open(manifest_file) as f:
        manifest = json.load(f)

    output_dir = Path("focused_diagrams")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("CREATING FOCUSED, INSIGHTFUL VISUALIZATIONS")
    print("="*80)

    before_path = Path(manifest['grokked_run']['checkpoints']['just_before_grok']['path'])
    after_path = Path(manifest['grokked_run']['checkpoints']['grokking_moment']['path'])

    # 1. Weight comparison (what changed?)
    create_weight_comparison(before_path, after_path,
                            output_dir / "weight_changes.png")

    # 2. Embedding structure (periodic patterns?)
    create_embedding_structure(before_path, after_path,
                               output_dir / "embedding_structure.png")

    print("\n" + "="*80)
    print("✅ FOCUSED DIAGRAMS COMPLETE!")
    print(f"📁 Saved to: {output_dir.absolute()}")
    print("\nRemoved bloat:")
    print("  ❌ Redundant attention pattern plots")
    print("  ❌ Overly detailed MLP composition matrices")
    print("  ❌ Uninformative singular value plots")
    print("\nKept insights:")
    print("  ✅ Weight magnitudes and changes (Δ)")
    print("  ✅ Neuron importance before/after")
    print("  ✅ Embedding periodic structure (FFT)")
    print("  ✅ Token similarity patterns")
    print("  ✅ Key metrics comparison")
    print("="*80)
