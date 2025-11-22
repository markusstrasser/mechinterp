"""
Deep analysis following Nanda et al. 2023 methodology.
Check for Fourier structure, neuron specialization, and circuit formation.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import create_model
from src.types import TrainConfig
import toml

config_dict = toml.load("configs/full_run2.toml")
config = TrainConfig(**config_dict)

P = config.p  # Should be 113

checkpoints = {
    "early": "checkpoints/e332cujg/early/model.pt",
    "just_before_grok": "checkpoints/e332cujg/just_before_grok/model.pt",
    "grokking_moment": "checkpoints/e332cujg/grokking_moment/model.pt",
    "final": "checkpoints/e332cujg/final/model.pt",
    "almost_final": "checkpoints/5bon0t2j/final/model.pt",
}


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    model = create_model(config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def analyze_fourier_structure(W_E, checkpoint_name):
    """
    Analyze Fourier structure in embeddings.
    Following Nanda: check for specific frequency components.
    """
    print(f"\n{'='*80}")
    print(f"FOURIER ANALYSIS: {checkpoint_name}")
    print(f"{'='*80}")

    # W_E is [vocab_size, d_model] where vocab_size = P+1
    # We care about embeddings for 0...P-1 (ignore the '=' token)
    W_E_numbers = W_E[:P, :]  # [P, d_model]

    # For each dimension, compute FFT across the P number tokens
    fourier_components = np.zeros((config.d_model, P))

    for dim in range(config.d_model):
        # Get this dimension across all number embeddings
        signal = W_E_numbers[:, dim]

        # Compute FFT
        fft_result = fft(signal)
        fourier_components[dim, :] = np.abs(fft_result)

    # Sum across dimensions to get total power at each frequency
    total_power = fourier_components.sum(axis=0)

    # Normalize
    total_power = total_power / total_power.sum()

    # Find top frequencies (excluding DC component at k=0)
    top_k_indices = np.argsort(total_power[1:])[-10:] + 1  # +1 to skip DC
    top_k_indices = top_k_indices[::-1]  # Reverse to get descending order

    print(f"\nTop 10 Fourier frequencies (k):")
    for i, k in enumerate(top_k_indices):
        print(f"  k={k:3d}: power={total_power[k]:.6f}")

    # Check for Nanda's specific frequencies for p=113
    nanda_frequencies = [14, 35, 41, 42, 52]
    print(f"\nNanda's key frequencies for p=113: {nanda_frequencies}")
    print(f"Power at Nanda's frequencies:")
    for k in nanda_frequencies:
        print(f"  k={k:3d}: power={total_power[k]:.6f}")

    # Compute Gini coefficient (Nanda's sparsity measure)
    sorted_power = np.sort(total_power)
    n = len(sorted_power)
    cumsum = np.cumsum(sorted_power)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_power)) / (n * cumsum[-1]) - (n + 1) / n

    print(f"\nGini coefficient of Fourier components: {gini:.6f}")
    print(f"  (Higher = more sparse, concentrated in few frequencies)")

    # Compute concentration in top 5 frequencies
    top5_power = total_power[top_k_indices[:5]].sum()
    print(f"Power in top 5 frequencies: {top5_power:.6f}")

    return {
        'total_power': total_power,
        'top_k_indices': top_k_indices,
        'gini': gini,
        'top5_power': top5_power,
        'fourier_components': fourier_components,
    }


def analyze_neuron_specialization(model, checkpoint_name):
    """
    Check if MLP neurons specialize to specific frequencies.
    """
    print(f"\n{'='*80}")
    print(f"NEURON SPECIALIZATION: {checkpoint_name}")
    print(f"{'='*80}")

    W_in = model.blocks[0].mlp.W_in.detach().cpu().numpy()  # [d_model, d_mlp]
    W_out = model.blocks[0].mlp.W_out.detach().cpu().numpy()  # [d_mlp, d_model]

    # Get embedding matrix
    W_E = model.W_E.detach().cpu().numpy()[:P, :]  # [P, d_model]

    # For each neuron, compute its "frequency signature"
    # by looking at W_in weights in Fourier space
    neuron_freq_signatures = np.zeros((config.d_ffn, P))

    for neuron_idx in range(config.d_ffn):
        # Get input weights for this neuron
        w_in_neuron = W_in[:, neuron_idx]  # [d_model]

        # Project embeddings through this neuron's input weights
        # This tells us which frequencies this neuron "listens to"
        projected = W_E @ w_in_neuron  # [P]

        # FFT to see frequency response
        fft_result = fft(projected)
        neuron_freq_signatures[neuron_idx, :] = np.abs(fft_result)

    # For each neuron, find its dominant frequency
    dominant_freqs = np.argmax(neuron_freq_signatures[:, 1:], axis=1) + 1  # Skip DC

    print(f"\nNeuron frequency specialization:")
    print(f"  Unique frequencies used by neurons: {len(np.unique(dominant_freqs))}")

    # Show distribution
    freq_counts = {}
    for f in dominant_freqs:
        freq_counts[f] = freq_counts.get(f, 0) + 1

    # Top 10 most common frequencies
    top_freq_neurons = sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 frequencies by neuron count:")
    for freq, count in top_freq_neurons:
        print(f"  k={freq:3d}: {count:3d} neurons ({100*count/config.d_ffn:.1f}%)")

    # Check if Nanda's frequencies are represented
    nanda_frequencies = [14, 35, 41, 42, 52]
    print(f"\nNeurons tuned to Nanda's frequencies:")
    for k in nanda_frequencies:
        count = freq_counts.get(k, 0)
        print(f"  k={k:3d}: {count:3d} neurons ({100*count/config.d_ffn:.1f}%)")

    return {
        'dominant_freqs': dominant_freqs,
        'freq_counts': freq_counts,
        'neuron_freq_signatures': neuron_freq_signatures,
    }


def analyze_excluded_loss(model, checkpoint_name, key_frequencies=[14, 35, 41, 42, 52]):
    """
    Compute "excluded loss" - performance after removing key frequencies.
    This measures reliance on the algorithmic (Fourier) solution.
    """
    print(f"\n{'='*80}")
    print(f"EXCLUDED LOSS ANALYSIS: {checkpoint_name}")
    print(f"{'='*80}")

    # Generate all possible test examples
    all_a = []
    all_b = []
    for a in range(P):
        for b in range(P):
            all_a.append(a)
            all_b.append(b)

    inputs = torch.stack([
        torch.tensor(all_a),
        torch.tensor(all_b),
        torch.full((len(all_a),), P)  # '=' token
    ], dim=1)

    labels = torch.tensor([(a + b) % P for a, b in zip(all_a, all_b)])

    with torch.no_grad():
        full_logits = model(inputs)  # [P*P, vocab_size]
        # For sequence models, we might get [batch, seq_len, vocab] or [batch, vocab]
        if len(full_logits.shape) == 3:
            logits = full_logits[:, -1, :P]  # Last position, only number outputs
        else:
            logits = full_logits[:, :P]  # Only number outputs

        # Compute normal accuracy
        preds = logits.argmax(dim=1)
        normal_acc = (preds == labels).float().mean().item()

        # Compute normal loss
        normal_loss = torch.nn.functional.cross_entropy(logits, labels).item()

        print(f"\nNormal performance:")
        print(f"  Accuracy: {normal_acc:.6f}")
        print(f"  Loss: {normal_loss:.6f}")

        # Now ablate key frequencies in Fourier space
        # For each example, FFT the logits, zero out key frequencies, IFFT back
        logits_freq = fft(logits.numpy(), axis=1)  # [P*P, P]

        # Zero out key frequencies
        logits_freq_ablated = logits_freq.copy()
        for k in key_frequencies:
            logits_freq_ablated[:, k] = 0
            # Also zero out negative frequency
            logits_freq_ablated[:, P - k] = 0

        # IFFT back
        from scipy.fft import ifft
        logits_ablated = ifft(logits_freq_ablated, axis=1).real
        logits_ablated = torch.from_numpy(logits_ablated).float()

        # Compute excluded accuracy/loss
        preds_ablated = logits_ablated.argmax(dim=1)
        excluded_acc = (preds_ablated == labels).float().mean().item()
        excluded_loss = torch.nn.functional.cross_entropy(logits_ablated, labels).item()

        print(f"\nExcluded performance (key frequencies removed):")
        print(f"  Accuracy: {excluded_acc:.6f} (drop: {normal_acc - excluded_acc:.6f})")
        print(f"  Loss: {excluded_loss:.6f} (increase: {excluded_loss - normal_loss:.6f})")

        print(f"\nInterpretation:")
        if normal_acc - excluded_acc > 0.3:
            print(f"  STRONG reliance on Fourier algorithm (large accuracy drop)")
        elif normal_acc - excluded_acc > 0.1:
            print(f"  MODERATE reliance on Fourier algorithm")
        else:
            print(f"  WEAK reliance on Fourier algorithm (likely memorization)")

    return {
        'normal_acc': normal_acc,
        'normal_loss': normal_loss,
        'excluded_acc': excluded_acc,
        'excluded_loss': excluded_loss,
        'acc_drop': normal_acc - excluded_acc,
    }


def main():
    print("="*80)
    print("NANDA-STYLE DEEP ANALYSIS")
    print("Following 'Progress Measures for Grokking' (Nanda et al., ICLR 2023)")
    print("="*80)

    results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n\n{'#'*80}")
        print(f"# {ckpt_name.upper()}")
        print(f"{'#'*80}")

        model = load_model(ckpt_path)
        W_E = model.W_E.detach().cpu().numpy()

        # Fourier analysis
        fourier_results = analyze_fourier_structure(W_E, ckpt_name)

        # Neuron specialization
        neuron_results = analyze_neuron_specialization(model, ckpt_name)

        # Excluded loss
        excluded_results = analyze_excluded_loss(model, ckpt_name)

        results[ckpt_name] = {
            'fourier': fourier_results,
            'neurons': neuron_results,
            'excluded': excluded_results,
        }

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Checkpoint':<20} {'Gini':>8} {'Top5 Pwr':>10} {'Acc Drop':>10} {'Normal Acc':>11}")
    print("-" * 70)
    for ckpt_name in checkpoints.keys():
        r = results[ckpt_name]
        print(f"{ckpt_name:<20} {r['fourier']['gini']:8.4f} {r['fourier']['top5_power']:10.4f} "
              f"{r['excluded']['acc_drop']:10.4f} {r['excluded']['normal_acc']:11.4f}")

    print("\nKey insights:")
    print("- Gini ↑ means more sparse (concentrated) Fourier spectrum")
    print("- Top5 Pwr ↑ means more power in top 5 frequencies")
    print("- Acc Drop ↑ means stronger reliance on Fourier algorithm")


if __name__ == "__main__":
    main()
