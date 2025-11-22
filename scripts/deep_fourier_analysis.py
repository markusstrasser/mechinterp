"""
Deep dive into Fourier structure - verify and visualize findings.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft, fft2
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import create_model
from src.types import TrainConfig
import toml

config_dict = toml.load("configs/full_run2.toml")
config = TrainConfig(**config_dict)
P = config.p

def load_model(checkpoint_path):
    model = create_model(config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def visualize_embedding_structure(W_E, checkpoint_name, save_path):
    """
    Create detailed visualizations of embedding Fourier structure.
    """
    W_E_numbers = W_E[:P, :]  # [P, d_model]

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # 1. Heatmap of raw embeddings
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(W_E_numbers.T, aspect='auto', cmap='RdBu_r',
                     vmin=-W_E_numbers.std()*3, vmax=W_E_numbers.std()*3)
    ax1.set_xlabel('Token Index (0 to p-1)')
    ax1.set_ylabel('Embedding Dimension')
    ax1.set_title(f'Raw Embeddings - {checkpoint_name}')
    plt.colorbar(im1, ax=ax1)

    # 2. Fourier transform for each dimension
    fourier_components = np.zeros((config.d_model, P))
    for dim in range(config.d_model):
        signal = W_E_numbers[:, dim]
        fft_result = fft(signal)
        fourier_components[dim, :] = np.abs(fft_result)

    ax2 = fig.add_subplot(gs[0, 2])
    total_power = fourier_components.sum(axis=0)
    total_power = total_power / total_power.sum()
    ax2.bar(range(P), total_power, width=1.0, edgecolor='none')
    ax2.set_xlabel('Frequency k')
    ax2.set_ylabel('Normalized Power')
    ax2.set_title('Total Fourier Power Spectrum')
    ax2.set_xlim(0, P)
    ax2.axvline(53, color='red', linestyle='--', alpha=0.7, label='k=53')
    ax2.axvline(60, color='red', linestyle='--', alpha=0.7, label='k=60')
    ax2.legend()

    # 3. Heatmap of Fourier components
    ax3 = fig.add_subplot(gs[1, :2])
    im3 = ax3.imshow(fourier_components, aspect='auto', cmap='hot')
    ax3.set_xlabel('Frequency k')
    ax3.set_ylabel('Embedding Dimension')
    ax3.set_title('Fourier Components (All Dimensions)')
    plt.colorbar(im3, ax=ax3)
    ax3.axvline(53, color='cyan', linestyle='--', alpha=0.7)
    ax3.axvline(60, color='cyan', linestyle='--', alpha=0.7)

    # 4. Top frequencies closeup
    ax4 = fig.add_subplot(gs[1, 2])
    top_k_indices = np.argsort(total_power)[-20:][::-1]
    ax4.barh(range(20), total_power[top_k_indices])
    ax4.set_yticks(range(20))
    ax4.set_yticklabels([f'k={k}' for k in top_k_indices])
    ax4.set_xlabel('Power')
    ax4.set_title('Top 20 Frequencies')
    ax4.invert_yaxis()

    # 5. Check periodicity in embeddings - plot actual embedding values
    ax5 = fig.add_subplot(gs[2, 0])
    # Plot first few dimensions as functions of token index
    for dim in range(min(5, config.d_model)):
        ax5.plot(W_E_numbers[:, dim], alpha=0.6, label=f'dim {dim}')
    ax5.set_xlabel('Token Index')
    ax5.set_ylabel('Embedding Value')
    ax5.set_title('Embeddings vs Token Index (first 5 dims)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Phase plot for k=53
    ax6 = fig.add_subplot(gs[2, 1])
    # Project embeddings onto k=53 frequency
    freq_53_basis_cos = np.cos(2 * np.pi * 53 * np.arange(P) / P)
    freq_53_basis_sin = np.sin(2 * np.pi * 53 * np.arange(P) / P)

    proj_cos = W_E_numbers @ W_E_numbers.T @ freq_53_basis_cos
    proj_sin = W_E_numbers @ W_E_numbers.T @ freq_53_basis_sin

    ax6.scatter(proj_cos, proj_sin, c=range(P), cmap='hsv', s=10, alpha=0.6)
    ax6.set_xlabel('Projection onto cos(2πk·53/p)')
    ax6.set_ylabel('Projection onto sin(2πk·53/p)')
    ax6.set_title('Phase Structure (k=53)')
    ax6.axis('equal')
    ax6.grid(True, alpha=0.3)

    # 7. Phase plot for k=60
    ax7 = fig.add_subplot(gs[2, 2])
    freq_60_basis_cos = np.cos(2 * np.pi * 60 * np.arange(P) / P)
    freq_60_basis_sin = np.sin(2 * np.pi * 60 * np.arange(P) / P)

    proj_cos = W_E_numbers @ W_E_numbers.T @ freq_60_basis_cos
    proj_sin = W_E_numbers @ W_E_numbers.T @ freq_60_basis_sin

    ax7.scatter(proj_cos, proj_sin, c=range(P), cmap='hsv', s=10, alpha=0.6)
    ax7.set_xlabel('Projection onto cos(2πk·60/p)')
    ax7.set_ylabel('Projection onto sin(2πk·60/p)')
    ax7.set_title('Phase Structure (k=60)')
    ax7.axis('equal')
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()

    return {
        'fourier_components': fourier_components,
        'total_power': total_power,
        'top_k_indices': top_k_indices,
    }


def analyze_frequency_math(P=113):
    """
    Analyze mathematical properties of key frequencies.
    Why k=53, 60 specifically?
    """
    print(f"\n{'='*80}")
    print(f"MATHEMATICAL ANALYSIS OF FREQUENCIES (p={P})")
    print(f"{'='*80}")

    # Check if they're related
    print(f"\nKey frequencies found: k=53, k=60")
    print(f"Sum: 53 + 60 = {53+60} = p = {P} ✓ (Conjugate pair!)")
    print(f"Difference: 60 - 53 = {60-53}")

    # GCD with p
    import math
    print(f"\nGCD with p:")
    print(f"  gcd(53, {P}) = {math.gcd(53, P)}")
    print(f"  gcd(60, {P}) = {math.gcd(60, P)}")

    # Check if they're generators or have special multiplicative properties
    print(f"\nMultiplicative properties:")
    print(f"  53 * 60 mod {P} = {(53*60) % P}")
    print(f"  53^2 mod {P} = {(53**2) % P}")
    print(f"  60^2 mod {P} = {(60**2) % P}")

    # Check Nanda's frequencies too
    print(f"\nNanda's frequencies: [14, 35, 41, 42, 52]")
    nanda_freqs = [14, 35, 41, 42, 52]
    print(f"Sum of Nanda's frequencies: {sum(nanda_freqs)}")
    print(f"Checking conjugates:")
    for k in nanda_freqs:
        print(f"  {k} + {P-k} = {P} (conjugate: k={P-k})")

    # Hypothesis: Maybe the model picks frequencies that maximize coverage
    # with minimal overlap
    print(f"\n" + "="*80)
    print(f"HYPOTHESIS: Why k=53, 60?")
    print(f"="*80)
    print(f"If we use frequency k, we get 2 basis functions: cos(2πkn/p), sin(2πkn/p)")
    print(f"For k=53, this gives periodicity with period p/gcd(53,p) = {P//math.gcd(53,P)}")
    print(f"For k=60, this gives periodicity with period p/gcd(60,p) = {P//math.gcd(60,P)}")

    # Check if there's a pattern in Nanda's p=113 results
    print(f"\nNanda used p={P} (same as us!)")
    print(f"But found different frequencies. This suggests:")
    print(f"  1. Random initialization matters")
    print(f"  2. Multiple 'solutions' exist in frequency space")
    print(f"  3. Our model found a SIMPLER solution (2 freq vs 5)")


def compare_grokked_vs_ungrokked():
    """
    Direct comparison between grokked and ungrokked models.
    """
    print(f"\n{'='*80}")
    print(f"GROKKED VS UNGROKKED COMPARISON")
    print(f"{'='*80}")

    grokked = load_model("checkpoints/e332cujg/grokking_moment/model.pt")
    ungrokked = load_model("checkpoints/5bon0t2j/final/model.pt")

    W_E_grok = grokked.W_E.detach().cpu().numpy()[:P, :]
    W_E_ungrok = ungrokked.W_E.detach().cpu().numpy()[:P, :]

    # Compare Fourier spectra
    fourier_grok = np.zeros(P)
    fourier_ungrok = np.zeros(P)

    for dim in range(config.d_model):
        fourier_grok += np.abs(fft(W_E_grok[:, dim]))
        fourier_ungrok += np.abs(fft(W_E_ungrok[:, dim]))

    fourier_grok /= fourier_grok.sum()
    fourier_ungrok /= fourier_ungrok.sum()

    print(f"\nFourier spectrum comparison:")
    print(f"  Grokked - Top 5 power: {fourier_grok[np.argsort(fourier_grok)[-5:]].sum():.4f}")
    print(f"  Ungrokked - Top 5 power: {fourier_ungrok[np.argsort(fourier_ungrok)[-5:]].sum():.4f}")

    # Gini comparison
    def gini(x):
        sorted_x = np.sort(x)
        n = len(sorted_x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(n) + 1) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

    print(f"\nGini coefficient:")
    print(f"  Grokked: {gini(fourier_grok):.6f}")
    print(f"  Ungrokked: {gini(fourier_ungrok):.6f}")
    print(f"  Ratio: {gini(fourier_grok) / gini(fourier_ungrok):.2f}x more concentrated")

    # MLP comparison
    W_in_grok = grokked.blocks[0].mlp.W_in.detach().cpu().numpy()
    W_in_ungrok = ungrokked.blocks[0].mlp.W_in.detach().cpu().numpy()

    print(f"\nMLP weights (W_in):")
    print(f"  Grokked - Frobenius norm: {np.linalg.norm(W_in_grok, 'fro'):.3f}")
    print(f"  Ungrokked - Frobenius norm: {np.linalg.norm(W_in_ungrok, 'fro'):.3f}")

    # SVD comparison
    U_g, s_g, Vh_g = np.linalg.svd(W_in_grok, full_matrices=False)
    U_u, s_u, Vh_u = np.linalg.svd(W_in_ungrok, full_matrices=False)

    print(f"\nSingular value spectra:")
    print(f"  Grokked top 5: {s_g[:5]}")
    print(f"  Ungrokked top 5: {s_u[:5]}")
    print(f"  Ratio of top singular values: {s_g[0] / s_u[0]:.3f}")


def main():
    print("="*80)
    print("DEEP FOURIER ANALYSIS - VERIFICATION AND VISUALIZATION")
    print("="*80)

    # Analyze frequency mathematics
    analyze_frequency_math(P)

    # Compare grokked vs ungrokked
    compare_grokked_vs_ungrokked()

    # Create visualizations for key checkpoints
    checkpoints = {
        "early": "checkpoints/e332cujg/early/model.pt",
        "grokking_moment": "checkpoints/e332cujg/grokking_moment/model.pt",
        "almost_final": "checkpoints/5bon0t2j/final/model.pt",
    }

    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    for name, path in checkpoints.items():
        print(f"\nProcessing {name}...")
        model = load_model(path)
        W_E = model.W_E.detach().cpu().numpy()

        save_path = f"fourier_analysis_{name}.png"
        results = visualize_embedding_structure(W_E, name, save_path)


if __name__ == "__main__":
    main()
