"""
Deep inspection of activations and weights across grokking transition.
This script extracts activations and computes interesting statistics.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model
from src.types import TrainConfig
import toml

# Load config to build model
config_dict = toml.load("configs/full_run2.toml")
config = TrainConfig(**config_dict)

# Key checkpoints to analyze
checkpoints = {
    "early": "checkpoints/e332cujg/early/model.pt",
    "mid_training": "checkpoints/e332cujg/mid_training/model.pt",
    "pre_grok_plateau": "checkpoints/e332cujg/pre_grok_plateau/model.pt",
    "just_before_grok": "checkpoints/e332cujg/just_before_grok/model.pt",
    "grokking_moment": "checkpoints/e332cujg/grokking_moment/model.pt",
    "post_grok_stable": "checkpoints/e332cujg/post_grok_stable/model.pt",
    "final": "checkpoints/e332cujg/final/model.pt",
    # Control (didn't grok)
    "almost_final": "checkpoints/5bon0t2j/final/model.pt",
}


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    # Build model using the create_model function
    model = create_model(config)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_test_samples(p, n_samples=100):
    """Generate random test samples for modular addition."""
    a = torch.randint(0, p, (n_samples,))
    b = torch.randint(0, p, (n_samples,))
    # Format: [a, b, =] where = is token p
    inputs = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    labels = (a + b) % p
    return inputs, labels


def extract_activations(model, inputs):
    """Extract activations at key points in the model using TransformerLens hooks."""
    activations = {}

    with torch.no_grad():
        # Run model with cache to get all activations
        logits, cache = model.run_with_cache(inputs)

        # Extract key activations
        activations['embeddings'] = cache['hook_embed'].cpu().numpy()
        activations['logits'] = logits.cpu().numpy()

        # For single-layer model (correct TransformerLens hook names)
        activations['block0_attn_pattern'] = cache['blocks.0.attn.hook_pattern'].cpu().numpy()
        activations['block0_attn_out'] = cache['blocks.0.hook_attn_out'].cpu().numpy()
        activations['block0_mlp_out'] = cache['blocks.0.hook_mlp_out'].cpu().numpy()
        activations['block0_residual'] = cache['blocks.0.hook_resid_post'].cpu().numpy()

    return activations, logits


def analyze_weight_matrices(checkpoint_name, model):
    """Analyze weight matrix properties."""
    print(f"\n{'='*80}")
    print(f"WEIGHT ANALYSIS: {checkpoint_name}")
    print(f"{'='*80}")

    # Embedding analysis (TransformerLens uses W_E)
    W_E = model.W_E.detach().cpu().numpy()  # [vocab_size, d_model]
    print(f"\nToken Embeddings (W_E): {W_E.shape}")
    print(f"  L2 norms per token: min={np.linalg.norm(W_E, axis=1).min():.3f}, "
          f"max={np.linalg.norm(W_E, axis=1).max():.3f}, "
          f"mean={np.linalg.norm(W_E, axis=1).mean():.3f}")

    # Attention weights (TransformerLens format)
    W_Q = model.blocks[0].attn.W_Q.detach().cpu().numpy()  # [n_heads, d_model, d_head]
    W_K = model.blocks[0].attn.W_K.detach().cpu().numpy()
    W_V = model.blocks[0].attn.W_V.detach().cpu().numpy()
    W_O = model.blocks[0].attn.W_O.detach().cpu().numpy()  # [n_heads, d_head, d_model]

    for head in range(3):
        # QK circuit for this head
        # W_Q[head]: [d_model, d_head], W_K[head]: [d_model, d_head]
        QK = W_Q[head].T @ W_K[head]  # [d_head, d_head]

        # OV circuit: W_O[head]: [d_head, d_model], W_V[head]: [d_model, d_head]
        OV = W_V[head] @ W_O[head]  # [d_model, d_model]

        print(f"\nHead {head}:")
        print(f"  QK circuit: rank={np.linalg.matrix_rank(QK)}, "
              f"||QK||_F={np.linalg.norm(QK, 'fro'):.3f}")
        print(f"  OV circuit: rank={np.linalg.matrix_rank(OV)}, "
              f"||OV||_F={np.linalg.norm(OV, 'fro'):.3f}")

    # MLP weights
    W_in = model.blocks[0].mlp.W_in.detach().cpu().numpy()  # [d_model, d_mlp]
    W_out = model.blocks[0].mlp.W_out.detach().cpu().numpy()  # [d_mlp, d_model]

    print(f"\nMLP:")
    print(f"  W_in: {W_in.shape}, ||W||_F={np.linalg.norm(W_in, 'fro'):.3f}")
    print(f"  W_out: {W_out.shape}, ||W||_F={np.linalg.norm(W_out, 'fro'):.3f}")
    print(f"  Effective rank (W_in): {np.linalg.matrix_rank(W_in)}")
    print(f"  Effective rank (W_out): {np.linalg.matrix_rank(W_out)}")

    # Compute singular values
    U_in, s_in, Vh_in = np.linalg.svd(W_in, full_matrices=False)
    U_out, s_out, Vh_out = np.linalg.svd(W_out, full_matrices=False)

    print(f"  Top 5 singular values (W_in): {s_in[:5]}")
    print(f"  Top 5 singular values (W_out): {s_out[:5]}")
    print(f"  Spectral norm ratio (s1/s2): W_in={s_in[0]/s_in[1]:.3f}, W_out={s_out[0]/s_out[1]:.3f}")


def analyze_activations(checkpoint_name, activations, labels):
    """Compute statistics on activations."""
    print(f"\n{'='*80}")
    print(f"ACTIVATION ANALYSIS: {checkpoint_name}")
    print(f"{'='*80}")

    # Analyze final layer representations
    final_repr = activations['block0_residual'][:, -1, :]  # [n_samples, d_model]

    print(f"\nFinal Layer Representations (before unembed):")
    print(f"  Shape: {final_repr.shape}")
    print(f"  Mean norm: {np.linalg.norm(final_repr, axis=1).mean():.3f}")
    print(f"  Std norm: {np.linalg.norm(final_repr, axis=1).std():.3f}")

    # Compute within-class variance (neural collapse metric)
    p = 113
    within_class_vars = []
    for class_idx in range(min(p, 20)):  # Just check first 20 classes
        mask = labels.numpy() == class_idx
        if mask.sum() > 1:
            class_repr = final_repr[mask]
            class_mean = class_repr.mean(axis=0)
            variance = ((class_repr - class_mean) ** 2).sum() / len(class_repr)
            within_class_vars.append(variance)

    if within_class_vars:
        print(f"\nNeural Collapse Metrics:")
        print(f"  Within-class variance (mean): {np.mean(within_class_vars):.6f}")
        print(f"  Within-class variance (std): {np.std(within_class_vars):.6f}")

    # Attention pattern analysis
    attn_patterns = activations['block0_attn_pattern']  # [n_samples, n_heads, seq_len, seq_len]
    print(f"\nAttention Patterns:")
    print(f"  Shape: {attn_patterns.shape}")

    # Average attention from position 2 (result) to positions 0,1 (operands)
    avg_attn = attn_patterns.mean(axis=0)  # [n_heads, seq_len, seq_len]
    for head in range(3):
        print(f"  Head {head}: pos2→pos0={avg_attn[head, 2, 0]:.3f}, "
              f"pos2→pos1={avg_attn[head, 2, 1]:.3f}, "
              f"pos2→pos2={avg_attn[head, 2, 2]:.3f}")

    # MLP activation magnitude
    mlp_out = activations['block0_mlp_out']
    print(f"\nMLP Output:")
    print(f"  Mean magnitude: {np.abs(mlp_out).mean():.3f}")
    print(f"  Max magnitude: {np.abs(mlp_out).max():.3f}")
    print(f"  Fraction > 0.1: {(np.abs(mlp_out) > 0.1).mean():.3f}")


def main():
    print("="*80)
    print("DEEP ACTIVATION AND WEIGHT INSPECTION")
    print("="*80)

    # Generate test samples
    print("\nGenerating test samples...")
    inputs, labels = generate_test_samples(config.p, n_samples=200)

    # Analyze key checkpoints
    key_checkpoints = [
        "early",
        "just_before_grok",
        "grokking_moment",
        "final",
        "almost_final",  # Control
    ]

    for ckpt_name in key_checkpoints:
        print(f"\n\n{'#'*80}")
        print(f"# {ckpt_name.upper()}")
        print(f"{'#'*80}")

        ckpt_path = checkpoints[ckpt_name]

        # Load model
        print(f"\nLoading model from {ckpt_path}...")
        model = load_model(ckpt_path)

        # Weight analysis
        analyze_weight_matrices(ckpt_name, model)

        # Activation analysis
        print(f"\nExtracting activations...")
        activations, logits = extract_activations(model, inputs)
        analyze_activations(ckpt_name, activations, labels)


if __name__ == "__main__":
    main()
