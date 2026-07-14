"""
Literature validation: test recent theoretical predictions against our data.

1. Execution Manifold Analysis (Xu, Feb 2026):
   - Do parameter trajectories collapse to low-dimensional manifolds?
   - Does dimensionality match their 3-4 finding?
   - Does manifold orientation vary by seed?

2. O(log n) Frequency Prediction (McCracken et al., NeurIPS 2025):
   - How many frequencies does our model actually use?
   - Does it match O(log 113) ≈ 7?

3. Frequency Switching Characterization:
   - Precisely when does 38,75 → 12,101 happen?
   - What does the transition look like in parameter space?
"""

import torch
import numpy as np
from pathlib import Path
from scipy.fft import fft
from sklearn.decomposition import PCA
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import create_model
from src.types import TrainConfig

P = 113
CONFIG = TrainConfig(
    p=P, d_model=32, n_heads=3, n_layers=1, d_ffn=128,
    lr=0.001, weight_decay=3.0, steps=60000, seed=42,
    device="cpu", frac_train=0.3,
)

CKPT_DIR = Path("checkpoints/1z2q8rx3")


def load_checkpoint(version_dir: Path):
    """Load model and return flattened parameters."""
    model = create_model(CONFIG)
    state_dict = torch.load(version_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_version_step_map(ckpt_dir: Path) -> dict[str, int]:
    """Build mapping from version directory name to training step."""
    # Load checkpoint_intervals from any metadata file
    intervals = None
    for d in ckpt_dir.iterdir():
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            intervals = meta.get("checkpoint_intervals", [])
            if intervals:
                break

    if not intervals:
        return {}

    # Named checkpoints map to specific versions
    # v0 -> intervals[0], v1 -> intervals[1], etc.
    mapping = {}
    for i, step in enumerate(intervals):
        mapping[f"v{i}"] = step

    # Named dirs are aliases - try to find their step from inventory
    inv_path = ckpt_dir.parent / "checkpoint_inventory.json"
    if inv_path.exists():
        with open(inv_path) as f:
            inventory = json.load(f)
        for run in inventory.get("runs", []):
            if run.get("run_id") == ckpt_dir.name:
                for ckpt in run.get("checkpoints", []):
                    name = ckpt.get("name", "")
                    step = ckpt.get("step", -1)
                    if name and step >= 0:
                        mapping[name] = step

    return mapping


def flatten_params(model) -> np.ndarray:
    """Flatten all model parameters into a single vector."""
    params = []
    for p in model.parameters():
        params.append(p.detach().cpu().numpy().flatten())
    return np.concatenate(params)


def fourier_analysis(model) -> dict:
    """Analyze Fourier structure of embeddings."""
    W_E = model.W_E.detach().cpu().numpy()[:P, :]  # [P, d_model]

    # FFT per dimension, sum magnitudes
    total_power = np.zeros(P)
    for dim in range(W_E.shape[1]):
        fft_result = fft(W_E[:, dim])
        total_power += np.abs(fft_result)

    # Normalize (skip DC)
    power_no_dc = total_power.copy()
    power_no_dc[0] = 0
    power_no_dc = power_no_dc / power_no_dc.sum()

    # Top frequencies
    top_indices = np.argsort(power_no_dc)[::-1][:20]

    # Gini coefficient
    sorted_p = np.sort(power_no_dc)
    n = len(sorted_p)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_p)) / (n * sorted_p.sum()) - (n + 1) / n

    # Count "significant" frequencies (> 2x mean power)
    mean_power = power_no_dc[1:].mean()  # excluding DC
    significant = np.where(power_no_dc[1:] > 2 * mean_power)[0] + 1
    very_significant = np.where(power_no_dc[1:] > 5 * mean_power)[0] + 1
    dominant = np.where(power_no_dc[1:] > 10 * mean_power)[0] + 1

    return {
        "top_freqs": top_indices[:10].tolist(),
        "top_powers": power_no_dc[top_indices[:10]].tolist(),
        "gini": float(gini),
        "n_significant_2x": len(significant),
        "n_significant_5x": len(very_significant),
        "n_dominant_10x": len(dominant),
        "significant_freqs": significant.tolist(),
        "dominant_freqs": dominant.tolist(),
        "power_spectrum": power_no_dc,
    }


def neuron_frequency_analysis(model) -> dict:
    """Analyze which frequencies MLP neurons are tuned to."""
    W_in = model.blocks[0].mlp.W_in.detach().cpu().numpy()  # [d_model, d_ffn]
    W_E = model.W_E.detach().cpu().numpy()[:P, :]  # [P, d_model]

    dominant_freqs = []
    for neuron_idx in range(W_in.shape[1]):
        projected = W_E @ W_in[:, neuron_idx]  # [P]
        fft_result = np.abs(fft(projected))
        fft_result[0] = 0  # skip DC
        dominant_freqs.append(int(np.argmax(fft_result)))

    unique_freqs = sorted(set(dominant_freqs))
    return {
        "dominant_freqs": dominant_freqs,
        "n_unique": len(unique_freqs),
        "unique_freqs": unique_freqs,
    }


def main():
    # ── Discover checkpoints ──
    print("Discovering checkpoints...")
    step_map = build_version_step_map(CKPT_DIR)

    versions = []
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir() and (d / "model.pt").exists():
            step = step_map.get(d.name, -1)
            if step >= 0:
                versions.append((step, d))

    versions.sort(key=lambda x: x[0])
    # Deduplicate by step (named dirs may alias versioned dirs)
    seen_steps = set()
    unique_versions = []
    for step, d in versions:
        if step not in seen_steps:
            seen_steps.add(step)
            unique_versions.append((step, d))
    versions = unique_versions

    print(f"Found {len(versions)} checkpoints (steps {versions[0][0]} to {versions[-1][0]})")

    # ── 1. EXECUTION MANIFOLD ANALYSIS ──
    print("\n" + "=" * 80)
    print("ANALYSIS 1: EXECUTION MANIFOLD (cf. Xu, Feb 2026)")
    print("=" * 80)

    print("Loading all checkpoints and flattening parameters...")
    param_vectors = []
    steps = []
    for step, vdir in versions:
        model = load_checkpoint(vdir)
        params = flatten_params(model)
        param_vectors.append(params)
        steps.append(step)
        if len(param_vectors) % 50 == 0:
            print(f"  Loaded {len(param_vectors)}/{len(versions)} checkpoints...")

    param_matrix = np.stack(param_vectors)  # [n_checkpoints, n_params]
    steps = np.array(steps)
    print(f"Parameter matrix shape: {param_matrix.shape}")
    print(f"  {param_matrix.shape[0]} checkpoints x {param_matrix.shape[1]} parameters")

    # PCA
    print("\nRunning PCA...")
    pca = PCA()
    pca.fit(param_matrix)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\nExplained variance by component:")
    for i in range(min(20, len(explained))):
        print(f"  PC{i+1:2d}: {explained[i]:8.4f} (cumulative: {cumulative[i]:8.4f})")

    # Find effective dimensionality
    dim_90 = np.searchsorted(cumulative, 0.90) + 1
    dim_95 = np.searchsorted(cumulative, 0.95) + 1
    dim_99 = np.searchsorted(cumulative, 0.99) + 1

    print(f"\nEffective dimensionality:")
    print(f"  90% variance: {dim_90} components")
    print(f"  95% variance: {dim_95} components")
    print(f"  99% variance: {dim_99} components")
    print(f"\n  Xu (2026) predicts: 3-4 dimensions")
    print(f"  Our finding: {dim_90} dimensions (90% threshold)")

    # Project onto top PCs
    projected = pca.transform(param_matrix)  # [n_checkpoints, n_components]

    # Identify key moments
    grok_mask = (steps >= 3400) & (steps <= 4200)
    freq_switch_idx = np.argmin(np.abs(steps - 3440))

    print(f"\n  Frequency switch at step 3440 -> PC coordinates:")
    print(f"    PC1={projected[freq_switch_idx, 0]:.3f}, PC2={projected[freq_switch_idx, 1]:.3f}, PC3={projected[freq_switch_idx, 2]:.3f}")

    # ── 2. O(log n) FREQUENCY PREDICTION ──
    print("\n" + "=" * 80)
    print("ANALYSIS 2: O(log n) FREQUENCY PREDICTION (cf. McCracken, NeurIPS 2025)")
    print(f"Prediction: O(log₂ {P}) = {np.log2(P):.1f} frequencies")
    print("=" * 80)

    # Analyze key checkpoints
    key_steps = [100, 500, 1000, 2000, 3000, 3400, 3440, 3500, 3600, 3800,
                 3900, 4000, 5000, 8000, 10000, 20000, 40000, 60000]

    print(f"\n{'Step':>6} {'TestAcc':>8} {'Gini':>6} {'#Sig(2x)':>8} {'#Sig(5x)':>8} "
          f"{'#Dom(10x)':>9} {'Top Freqs':>30} {'#Neuron Freqs':>14}")
    print("-" * 110)

    for target_step in key_steps:
        # Find closest checkpoint
        idx = np.argmin(np.abs(steps - target_step))
        actual_step = steps[idx]

        model = load_checkpoint(versions[idx][1])

        # Fourier analysis
        fa = fourier_analysis(model)

        # Neuron analysis
        na = neuron_frequency_analysis(model)

        # Test accuracy
        all_inputs = []
        all_labels = []
        for a in range(P):
            for b in range(P):
                all_inputs.append([a, b, P])
                all_labels.append((a + b) % P)
        inputs_t = torch.tensor(all_inputs)
        labels_t = torch.tensor(all_labels)

        with torch.no_grad():
            logits = model(inputs_t)[:, -1, :P]
            preds = logits.argmax(dim=1)
            acc = (preds == labels_t).float().mean().item()

        top_3_str = ", ".join(f"{k}" for k in fa["top_freqs"][:3])

        print(f"{actual_step:6d} {acc:8.1%} {fa['gini']:6.3f} "
              f"{fa['n_significant_2x']:8d} {fa['n_significant_5x']:8d} "
              f"{fa['n_dominant_10x']:9d} {top_3_str:>30} {na['n_unique']:14d}")

    # ── 3. DETAILED FREQUENCY SWITCH ANALYSIS ──
    print("\n" + "=" * 80)
    print("ANALYSIS 3: FREQUENCY SWITCHING (steps 3000-4500)")
    print("=" * 80)

    switch_steps = [s for s in steps if 3000 <= s <= 4500]
    print(f"\n{'Step':>6} {'Top1':>5} {'Top2':>5} {'Top3':>5} {'Top4':>5} "
          f"{'Pair Sum':>9} {'Conjugate?':>11} {'Gini':>6}")
    print("-" * 70)

    for target_step in switch_steps:
        idx = np.argmin(np.abs(steps - target_step))
        model = load_checkpoint(versions[idx][1])
        fa = fourier_analysis(model)

        top4 = fa["top_freqs"][:4]
        # Check if top 2 are conjugate pair
        pair_sum = top4[0] + top4[1]
        is_conjugate = "YES" if pair_sum == P else "no"

        print(f"{steps[idx]:6d} {top4[0]:5d} {top4[1]:5d} {top4[2]:5d} {top4[3]:5d} "
              f"{pair_sum:9d} {is_conjugate:>11} {fa['gini']:6.3f}")

    # ── 4. FINAL MODEL DEEP FREQUENCY COUNT ──
    print("\n" + "=" * 80)
    print("ANALYSIS 4: PRECISE FREQUENCY COUNT (Final Model)")
    print("=" * 80)

    # Load final model
    final_idx = len(versions) - 1
    model = load_checkpoint(versions[final_idx][1])
    fa = fourier_analysis(model)
    na = neuron_frequency_analysis(model)

    print(f"\nFinal model (step {steps[final_idx]}):")
    print(f"\nEmbedding Fourier spectrum:")
    print(f"  Frequencies > 2x mean: {fa['n_significant_2x']} ({fa['significant_freqs'][:15]}...)")
    print(f"  Frequencies > 5x mean: {fa['n_significant_5x']} ({fa['dominant_freqs'][:15] if fa['n_significant_5x'] > 0 else 'none'})")
    print(f"  Frequencies > 10x mean: {fa['n_dominant_10x']} ({fa['dominant_freqs'][:15] if fa['n_dominant_10x'] > 0 else 'none'})")
    print(f"  Gini: {fa['gini']:.4f}")

    print(f"\nNeuron specialization:")
    print(f"  Unique frequencies across {CONFIG.d_ffn} neurons: {na['n_unique']}")
    print(f"  Frequencies: {na['unique_freqs']}")

    print(f"\nConjugate pair check:")
    for k in fa["top_freqs"][:5]:
        conj = P - k
        print(f"  k={k}, p-k={conj}, sum={k+conj}")

    print(f"\n  McCracken prediction: O(log₂ {P}) ≈ {np.log2(P):.1f} frequencies")
    print(f"  Our dominant (>10x): {fa['n_dominant_10x']}")
    print(f"  Our significant (>5x): {fa['n_significant_5x']}")
    print(f"  Our significant (>2x): {fa['n_significant_2x']}")

    # ── 5. EXCLUDED LOSS WITH VARYING FREQUENCY COUNTS ──
    print("\n" + "=" * 80)
    print("ANALYSIS 5: HOW MANY FREQUENCIES ARE ACTUALLY NEEDED?")
    print("=" * 80)

    # Test accuracy when keeping only top-N frequencies
    all_inputs = []
    all_labels = []
    for a in range(P):
        for b in range(P):
            all_inputs.append([a, b, P])
            all_labels.append((a + b) % P)
    inputs_t = torch.tensor(all_inputs)
    labels_t = torch.tensor(all_labels)

    with torch.no_grad():
        logits = model(inputs_t)[:, -1, :P]
        full_acc = (logits.argmax(dim=1) == labels_t).float().mean().item()

    print(f"\nFull model accuracy: {full_acc:.4f}")
    print(f"\nAblation: zeroing out frequencies and checking accuracy drop")

    top_freqs_ordered = fa["top_freqs"]

    # Test: remove top-N frequencies one by one
    print(f"\n{'Removed':>20} {'Remaining Acc':>14} {'Acc Drop':>10}")
    print("-" * 50)

    logits_np = logits.numpy()
    logits_fft = fft(logits_np, axis=1)

    for n_remove in range(1, min(8, len(top_freqs_ordered))):
        freqs_to_remove = top_freqs_ordered[:n_remove]
        ablated = logits_fft.copy()
        for k in freqs_to_remove:
            ablated[:, k] = 0
            if P - k != k:
                ablated[:, P - k] = 0

        from scipy.fft import ifft
        logits_ablated = ifft(ablated, axis=1).real
        preds = np.argmax(logits_ablated, axis=1)
        acc = (preds == labels_t.numpy()).mean()
        removed_str = ",".join(str(k) for k in freqs_to_remove)
        print(f"  k={removed_str:>16} {acc:14.4f} {full_acc - acc:10.4f}")

    # Test: keep ONLY top-N frequencies
    print(f"\n{'Kept Only':>20} {'Accuracy':>14}")
    print("-" * 40)

    for n_keep in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        freqs_to_keep = set(top_freqs_ordered[:n_keep])
        # Also keep conjugates
        freqs_with_conj = set()
        for k in freqs_to_keep:
            freqs_with_conj.add(k)
            freqs_with_conj.add(P - k)
        freqs_with_conj.add(0)  # Keep DC

        ablated = np.zeros_like(logits_fft)
        for k in freqs_with_conj:
            if k < P:
                ablated[:, k] = logits_fft[:, k]

        logits_ablated = ifft(ablated, axis=1).real
        preds = np.argmax(logits_ablated, axis=1)
        acc = (preds == labels_t.numpy()).mean()
        kept_str = ",".join(str(k) for k in sorted(freqs_to_keep)[:5])
        if n_keep > 5:
            kept_str += "..."
        print(f"  top {n_keep:2d} ({kept_str:>12}) {acc:14.4f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
