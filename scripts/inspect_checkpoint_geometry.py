"""
Compact, context-window-friendly checkpoint inspector.

Goal: take a single `model.pt` plus its config and print a *small*,
rounded summary of the key geometric features we care about:

- Embedding Fourier spectrum (top frequencies, Gini, conjugate pairs)
- MLP neuron frequency specialization
- Attention QK / OV circuit ranks and norms

This is intentionally minimal: everything is derived directly from the
full weights but reduced to a few lines of text so you can paste it
into a chat window without blowing the context budget.

Usage example (from repo root):

  python -m scripts.inspect_checkpoint_geometry \\
      --config configs/full_run2.toml \\
      --checkpoint checkpoints/e332cujg/grokking_moment/model.pt \\
      --label e332cujg_grokking_moment

  python -m scripts.inspect_checkpoint_geometry \\
      --config configs/ultra_dense_grokking.toml \\
      --checkpoint checkpoints/1z2q8rx3/v200/model.pt \\
      --label seed42_v200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import toml
import torch

from src.model import create_model
from src.types import TrainConfig


def load_config(path: str | Path) -> TrainConfig:
    config_dict = toml.load(str(path))
    return TrainConfig(**config_dict)


def load_model(config: TrainConfig, checkpoint_path: str | Path):
    model = create_model(config)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def gini(x: np.ndarray) -> float:
    """Standard Gini coefficient used elsewhere in this repo."""
    x = x.astype(np.float64)
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return (2.0 * np.sum((np.arange(1, n + 1) * x))) / (n * cum[-1]) - (n + 1) / n


def summarize_embedding_fourier(W_E: np.ndarray, p: int, top_k: int = 10) -> Dict:
    """
    Fourier analysis of embeddings.

    W_E: [vocab_size, d_model]; we only use tokens 0..p-1.
    Returns small numeric summaries with rounded floats.
    """
    W_E_numbers = W_E[:p, :]  # [P, d_model]

    # FFT along the token dimension
    fft_result = np.fft.fft(W_E_numbers, axis=0)  # [P, d_model]
    mag = np.abs(fft_result)

    # Total power per frequency k
    total_power = mag.sum(axis=1)
    total_power = total_power / total_power.sum()

    # Skip DC (k=0) for ranking
    freq_indices = np.arange(1, p)
    sorted_idx = freq_indices[np.argsort(total_power[1:])][::-1]
    top_freqs = sorted_idx[:top_k]

    # Conjugate pair power: (k, p-k), dedup by keeping k <= p-k
    pair_power: Dict[Tuple[int, int], float] = {}
    for k in range(1, p):
        partner = (p - k) % p
        if k > partner:
            continue
        pair_power[(k, partner)] = float(total_power[k] + total_power[partner])
    top_pairs = sorted(pair_power.items(), key=lambda kv: kv[1], reverse=True)[: min(5, len(pair_power))]

    return {
        "gini": float(gini(total_power)),
        "top_freqs": [(int(k), float(total_power[k])) for k in top_freqs],
        "top_conjugate_pairs": [((int(k1), int(k2)), float(pwr)) for (k1, k2), pwr in top_pairs],
    }


def summarize_neuron_specialization(
    W_E: np.ndarray, W_in: np.ndarray, p: int, d_mlp: int, top_k: int = 10
) -> Dict:
    """
    Neuron frequency specialization summary.

    For each neuron in the MLP (column of W_in), project embeddings
    through its input weights and FFT over tokens. The dominant
    frequency is argmax over |FFT| excluding DC.
    """
    W_E_numbers = W_E[:p, :]  # [P, d_model]
    neuron_freq_signatures = np.zeros((d_mlp, p), dtype=np.float64)

    for neuron_idx in range(d_mlp):
        w = W_in[:, neuron_idx]  # [d_model]
        projected = W_E_numbers @ w  # [P]
        fft_result = np.fft.fft(projected)
        neuron_freq_signatures[neuron_idx] = np.abs(fft_result)

    dominant_freqs = np.argmax(neuron_freq_signatures[:, 1:], axis=1) + 1  # skip DC

    # Count frequencies
    freq_counts: Dict[int, int] = {}
    for f in dominant_freqs:
        freq_counts[int(f)] = freq_counts.get(int(f), 0) + 1

    total_neurons = float(d_mlp)
    top_items = sorted(freq_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    top_summary = [
        (int(k), int(count), float(count / total_neurons)) for k, count in top_items
    ]

    return {
        "num_unique_freqs": int(len(freq_counts)),
        "top_freqs_by_neuron_count": top_summary,
    }


def summarize_attention_qk_ov(model, n_heads: int) -> Dict:
    """
    Basic QK/OV circuit summary per head (rank and Frobenius norm),
    matching the conventions in `scripts/inspect_activations.py`.
    """
    W_Q = model.blocks[0].attn.W_Q.detach().cpu().numpy()  # [n_heads, d_model, d_head]
    W_K = model.blocks[0].attn.W_K.detach().cpu().numpy()
    W_V = model.blocks[0].attn.W_V.detach().cpu().numpy()
    W_O = model.blocks[0].attn.W_O.detach().cpu().numpy()  # [n_heads, d_head, d_model]

    heads = []
    for h in range(n_heads):
        QK = W_Q[h].T @ W_K[h]  # [d_head, d_head]
        OV = W_V[h] @ W_O[h]  # [d_model, d_model]

        heads.append(
            {
                "head": int(h),
                "QK_rank": int(np.linalg.matrix_rank(QK)),
                "QK_fro": float(np.linalg.norm(QK, "fro")),
                "OV_rank": int(np.linalg.matrix_rank(OV)),
                "OV_fro": float(np.linalg.norm(OV, "fro")),
            }
        )

    return {"heads": heads}


def inspect_checkpoint(config_path: str | Path, checkpoint_path: str | Path, label: str | None) -> None:
    config = load_config(config_path)
    model = load_model(config, checkpoint_path)

    p = int(config.p)
    d_model = int(config.d_model)
    d_mlp = int(config.d_ffn)
    n_heads = int(config.n_heads)

    W_E = model.W_E.detach().cpu().numpy()  # [vocab, d_model]
    W_in = model.blocks[0].mlp.W_in.detach().cpu().numpy()  # [d_model, d_mlp]

    emb_fourier = summarize_embedding_fourier(W_E, p)
    neuron_spec = summarize_neuron_specialization(W_E, W_in, p, d_mlp)
    attn_summary = summarize_attention_qk_ov(model, n_heads)

    name = label or Path(checkpoint_path).stem

    print("=" * 80)
    print(f"CHECKPOINT GEOMETRY SUMMARY: {name}")
    print("=" * 80)
    print(f"config:    {config_path}")
    print(f"checkpoint:{checkpoint_path}")
    print(f"p={p}, d_model={d_model}, d_ffn={d_mlp}, n_heads={n_heads}")

    # Embedding Fourier summary
    print("\n[Embeddings: Fourier spectrum]")
    print(f"  Gini (sparsity): {emb_fourier['gini']:.6f}")
    print("  Top frequencies (k: power):")
    for k, power in emb_fourier["top_freqs"]:
        print(f"    k={k:3d}: power={power:.6f}")

    print("  Top conjugate pairs (k, p-k: pair_power):")
    for (k1, k2), pair_power in emb_fourier["top_conjugate_pairs"]:
        print(f"    ({k1:3d}, {k2:3d}): pair_power={pair_power:.6f}")

    # Neuron specialization summary
    print("\n[MLP neuron frequency specialization]")
    print(f"  Unique dominant frequencies: {neuron_spec['num_unique_freqs']}")
    print("  Top frequencies by neuron count (k: count, frac):")
    for k, count, frac in neuron_spec["top_freqs_by_neuron_count"]:
        print(f"    k={k:3d}: {count:3d} neurons ({100.0 * frac:5.1f}%)")

    # Attention QK/OV summary
    print("\n[Attention QK / OV circuits]")
    for h in attn_summary["heads"]:
        print(
            f"  Head {h['head']}: "
            f"QK_rank={h['QK_rank']}, QK||·||_F={h['QK_fro']:.3f}; "
            f"OV_rank={h['OV_rank']}, OV||·||_F={h['OV_fro']:.3f}"
        )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact checkpoint geometry inspector.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML config used to construct the model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model.pt checkpoint.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional human-readable label for this checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_checkpoint(args.config, args.checkpoint, args.label)


if __name__ == "__main__":
    main()
