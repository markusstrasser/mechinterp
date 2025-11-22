"""Generate circuit diagrams for grokking analysis."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from matplotlib.gridspec import GridSpec


def load_checkpoint(checkpoint_path: Path):
    """Load a model checkpoint."""
    model_file = checkpoint_path / "model.pt"
    metadata_file = checkpoint_path / "metadata.json"

    model_state = torch.load(model_file, map_location='cpu')

    with open(metadata_file) as f:
        metadata = json.load(f)

    return model_state, metadata


def plot_weight_heatmap(ax, weight, title, vmin=None, vmax=None, cmap='RdBu_r'):
    """Plot a weight matrix as a heatmap."""
    if len(weight.shape) > 2:
        # Flatten extra dimensions
        weight = weight.reshape(weight.shape[0], -1)

    im = ax.imshow(weight, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(f'Dim {weight.shape[1]}', fontsize=8)
    ax.set_ylabel(f'Dim {weight.shape[0]}', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_attention_pattern(ax, model_state, head_idx=0, n_ctx=3):
    """Plot attention pattern for a specific head."""
    # For a 1-layer model, get W_Q and W_K for the head
    # Attention pattern (data-independent) = softmax(W_Q @ W_E.T @ W_E @ W_K.T / sqrt(d_head))

    W_E = model_state['embed.W_E'].numpy()  # [vocab_size, d_model]
    W_Q = model_state['blocks.0.attn.W_Q'].numpy()  # [n_heads, d_model, d_head]
    W_K = model_state['blocks.0.attn.W_K'].numpy()

    n_heads = W_Q.shape[0]
    d_head = W_Q.shape[2]

    # Get specific head
    W_Q_h = W_Q[head_idx]  # [d_model, d_head]
    W_K_h = W_K[head_idx]  # [d_model, d_head]

    # Compute QK circuit for embeddings (simplified - just first few tokens)
    # QK = W_Q @ W_E.T @ W_E @ W_K.T
    QK = W_Q_h.T @ W_E.T @ W_E @ W_K_h / np.sqrt(d_head)  # [vocab_size, vocab_size] approximation

    # For visualization, show a smaller subset
    subset_size = min(20, QK.shape[0])
    QK_subset = QK[:subset_size, :subset_size]

    im = ax.imshow(QK_subset, cmap='RdBu_r', vmin=-np.abs(QK_subset).max(), vmax=np.abs(QK_subset).max())
    ax.set_title(f'Attention Pattern (Head {head_idx})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Key Position', fontsize=8)
    ax.set_ylabel('Query Position', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_circuit_diagram(checkpoint_path: Path, output_path: Path, title: str):
    """Create a comprehensive circuit diagram for a checkpoint."""
    model_state, metadata = load_checkpoint(checkpoint_path)

    # Extract key weights
    W_E = model_state['embed.W_E'].numpy()  # [vocab_size, d_model]
    W_U = model_state['unembed.W_U'].numpy()  # [d_model, vocab_size]

    # Attention weights (1 layer)
    W_Q = model_state['blocks.0.attn.W_Q'].numpy()  # [n_heads, d_model, d_head]
    W_K = model_state['blocks.0.attn.W_K'].numpy()
    W_V = model_state['blocks.0.attn.W_V'].numpy()
    W_O = model_state['blocks.0.attn.W_O'].numpy()  # [n_heads, d_head, d_model]

    # MLP weights
    W_in = model_state['blocks.0.mlp.W_in'].numpy()  # [d_model, d_ffn]
    W_out = model_state['blocks.0.mlp.W_out'].numpy()  # [d_ffn, d_model]

    n_heads = W_Q.shape[0]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)

    # Title
    fig.suptitle(f"{title}\nTest Acc: {metadata.get('test_acc', 0):.4f} | Train Loss: {metadata.get('train_loss', 0):.4f}",
                 fontsize=16, fontweight='bold')

    # Row 1: Embedding and Unembedding
    ax1 = fig.add_subplot(gs[0, 0])
    plot_weight_heatmap(ax1, W_E.T, 'Embedding (W_E)', cmap='viridis')

    ax2 = fig.add_subplot(gs[0, 1])
    plot_weight_heatmap(ax2, W_U, 'Unembedding (W_U)', cmap='viridis')

    ax3 = fig.add_subplot(gs[0, 2])
    # Compute W_U @ W_E (direct path from input to output)
    W_logit = W_U.T @ W_E.T  # [vocab, vocab]
    plot_weight_heatmap(ax3, W_logit[:50, :50], 'W_U @ W_E (Direct Path)', vmin=-1, vmax=1)

    ax4 = fig.add_subplot(gs[0, 3])
    # Embedding singular values (structure in embedding space)
    U, S, Vh = np.linalg.svd(W_E.T, full_matrices=False)
    ax4.plot(S / S[0], 'o-', markersize=4)
    ax4.set_title('Embedding Spectrum', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Component', fontsize=8)
    ax4.set_ylabel('Normalized Singular Value', fontsize=8)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Row 2: Attention heads
    for i in range(min(n_heads, 3)):
        ax = fig.add_subplot(gs[1, i])
        plot_attention_pattern(ax, model_state, head_idx=i)

    # Show attention head specialization
    ax = fig.add_subplot(gs[1, 3])
    # Compute OV circuit for each head: W_V @ W_O
    ov_norms = []
    for i in range(n_heads):
        OV = W_O[i] @ W_V[i]  # [d_model, d_model]
        ov_norms.append(np.linalg.norm(OV, 'fro'))
    ax.bar(range(n_heads), ov_norms)
    ax.set_title('Attention Head Strength (OV norm)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Head', fontsize=8)
    ax.set_ylabel('Frobenius Norm', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3: MLP
    ax1 = fig.add_subplot(gs[2, 0])
    plot_weight_heatmap(ax1, W_in, 'MLP Input (W_in)', cmap='RdBu_r')

    ax2 = fig.add_subplot(gs[2, 1])
    plot_weight_heatmap(ax2, W_out, 'MLP Output (W_out)', cmap='RdBu_r')

    ax3 = fig.add_subplot(gs[2, 2])
    # Neuron importance: norm of input and output weights
    neuron_in_norms = np.linalg.norm(W_in, axis=0)
    neuron_out_norms = np.linalg.norm(W_out, axis=1)
    top_k = 20
    top_neurons = np.argsort(neuron_in_norms * neuron_out_norms)[-top_k:]
    ax3.barh(range(top_k), (neuron_in_norms * neuron_out_norms)[top_neurons])
    ax3.set_title(f'Top {top_k} Neurons (by weight norm)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Weight Norm Product', fontsize=8)
    ax3.set_ylabel('Neuron Index', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='x')

    ax4 = fig.add_subplot(gs[2, 3])
    # MLP composition: W_out @ W_in [d_model, d_model]
    MLP_composition = W_out.T @ W_in.T
    plot_weight_heatmap(ax4, MLP_composition, 'MLP Composition (W_out @ W_in)', cmap='RdBu_r')

    # Row 4: Interpretability metrics
    ax1 = fig.add_subplot(gs[3, :2])
    # Show key metrics
    metrics = {
        'Test Acc': metadata.get('test_acc', 0),
        'Train Loss': metadata.get('train_loss', 0),
        'Fourier Sparsity': metadata.get('fourier_sparsity', 0),
        'Circulant Score': metadata.get('circulant_score', 0),
        'Neuron Spec': metadata.get('neuron_specialization', 0),
        'L2 Norm': metadata.get('l2_norm', 0) / 1000,  # Scale for visibility
    }
    ax1.barh(list(metrics.keys()), list(metrics.values()))
    ax1.set_title('Interpretability Metrics', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='x')

    ax2 = fig.add_subplot(gs[3, 2:])
    # Logit attribution breakdown
    logit_attrs = {
        'Direct': metadata.get('logit_attribution.direct', 0),
        'MLP': metadata.get('logit_attribution.mlp', 0),
        'Attn': metadata.get('logit_attribution.attn', 0),
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax2.pie(list(logit_attrs.values()), labels=list(logit_attrs.keys()),
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Logit Attribution', fontsize=10, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved circuit diagram to {output_path}")
    plt.close()


def create_comparison_diagram(before_path: Path, after_path: Path, output_path: Path):
    """Create a side-by-side comparison of before and after grokking."""
    model_before, meta_before = load_checkpoint(before_path)
    model_after, meta_after = load_checkpoint(after_path)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Before vs After Grokking Comparison', fontsize=16, fontweight='bold')

    # Row 1: Before grokking
    W_E_before = model_before['embed.W_E'].numpy()
    W_U_before = model_before['unembed.W_U'].numpy()
    W_in_before = model_before['blocks.0.mlp.W_in'].numpy()
    W_out_before = model_before['blocks.0.mlp.W_out'].numpy()

    axes[0, 0].set_ylabel('BEFORE\nTest Acc: {:.3f}'.format(meta_before.get('test_acc', 0)),
                          fontsize=12, fontweight='bold')
    plot_weight_heatmap(axes[0, 0], W_E_before.T, 'Embedding', cmap='viridis')
    plot_weight_heatmap(axes[0, 1], W_U_before, 'Unembedding', cmap='viridis')
    plot_weight_heatmap(axes[0, 2], W_in_before, 'MLP Input', cmap='RdBu_r')
    plot_weight_heatmap(axes[0, 3], W_out_before, 'MLP Output', cmap='RdBu_r')

    # Row 2: After grokking
    W_E_after = model_after['embed.W_E'].numpy()
    W_U_after = model_after['unembed.W_U'].numpy()
    W_in_after = model_after['blocks.0.mlp.W_in'].numpy()
    W_out_after = model_after['blocks.0.mlp.W_out'].numpy()

    axes[1, 0].set_ylabel('AFTER\nTest Acc: {:.3f}'.format(meta_after.get('test_acc', 0)),
                          fontsize=12, fontweight='bold')
    plot_weight_heatmap(axes[1, 0], W_E_after.T, 'Embedding', cmap='viridis')
    plot_weight_heatmap(axes[1, 1], W_U_after, 'Unembedding', cmap='viridis')
    plot_weight_heatmap(axes[1, 2], W_in_after, 'MLP Input', cmap='RdBu_r')
    plot_weight_heatmap(axes[1, 3], W_out_after, 'MLP Output', cmap='RdBu_r')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison diagram to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Load manifest
    manifest_file = Path("checkpoints/grokking_manifest.json")
    with open(manifest_file) as f:
        manifest = json.load(f)

    output_dir = Path("circuit_diagrams")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("GENERATING CIRCUIT DIAGRAMS")
    print("="*80)

    # Generate individual circuit diagrams
    checkpoints_to_visualize = [
        ('just_before_grok', 'Just Before Grokking (v23, test_acc=0.27)'),
        ('grokking_moment', 'Grokking Moment (v24, test_acc=0.98)'),
        ('post_grok_stable', 'Post-Grokking Stable (v30)'),
    ]

    for ckpt_key, title in checkpoints_to_visualize:
        ckpt_path = Path(manifest['grokked_run']['checkpoints'][ckpt_key]['path'])
        output_path = output_dir / f"{ckpt_key}.png"
        print(f"\nGenerating diagram for: {title}")
        create_circuit_diagram(ckpt_path, output_path, title)

    # Almost-grokked final
    almost_path = Path(manifest['almost_run']['checkpoints']['final']['path'])
    create_circuit_diagram(almost_path, output_dir / "almost_grokked_final.png",
                          "Almost-Grokked (wd=2, test_acc=0.26)")

    # Create comparisons
    print("\n" + "="*80)
    print("GENERATING COMPARISON DIAGRAMS")
    print("="*80)

    before_path = Path(manifest['grokked_run']['checkpoints']['just_before_grok']['path'])
    after_path = Path(manifest['grokked_run']['checkpoints']['grokking_moment']['path'])
    create_comparison_diagram(before_path, after_path,
                            output_dir / "before_vs_after_grokking.png")

    print("\n" + "="*80)
    print("✅ ALL CIRCUIT DIAGRAMS GENERATED!")
    print(f"📁 Saved to: {output_dir.absolute()}")
    print("="*80)
