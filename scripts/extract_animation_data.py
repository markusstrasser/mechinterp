"""
Extract animation data from grokking checkpoints.

Processes all checkpoints from run 1z2q8rx3 and saves metrics
needed for Manim animations to a single .npz file.

Usage:
    uv run python scripts/extract_animation_data.py
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
D_MODEL = 32
D_FFN = 128

CONFIG = TrainConfig(
    p=P, d_model=D_MODEL, n_heads=3, n_layers=1, d_ffn=D_FFN,
    lr=0.001, weight_decay=3.0, steps=60000, seed=42,
    device="cpu", frac_train=0.3,
)

CKPT_DIR = Path("checkpoints/1z2q8rx3")
OUTPUT_DIR = Path("animations/data")


def build_version_step_map(ckpt_dir: Path) -> dict[str, int]:
    """Build mapping from version directory name to training step."""
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

    mapping = {}
    for i, step in enumerate(intervals):
        mapping[f"v{i}"] = step

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


def load_checkpoint(version_dir: Path):
    model = create_model(CONFIG)
    state_dict = torch.load(version_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_fourier_power(model) -> np.ndarray:
    """Return FFT power spectrum of embeddings, shape [P]."""
    W_E = model.W_E.detach().cpu().numpy()[:P, :]
    total_power = np.zeros(P)
    for dim in range(W_E.shape[1]):
        total_power += np.abs(fft(W_E[:, dim]))
    total_power[0] = 0  # drop DC
    return total_power


def compute_neuron_freqs(model) -> np.ndarray:
    """Return dominant frequency per MLP neuron, shape [D_FFN]."""
    W_in = model.blocks[0].mlp.W_in.detach().cpu().numpy()  # [d_model, d_ffn]
    W_E = model.W_E.detach().cpu().numpy()[:P, :]
    freqs = np.zeros(D_FFN, dtype=np.int32)
    for i in range(D_FFN):
        projected = W_E @ W_in[:, i]
        fft_mag = np.abs(fft(projected))
        fft_mag[0] = 0
        freqs[i] = np.argmax(fft_mag)
    return freqs


def get_embedding_matrix(model) -> np.ndarray:
    """Return embedding matrix for numbers 0..P-1, shape [P, d_model]."""
    return model.W_E.detach().cpu().numpy()[:P, :]


def main():
    print("Discovering checkpoints...")
    step_map = build_version_step_map(CKPT_DIR)

    versions = []
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir() and (d / "model.pt").exists():
            step = step_map.get(d.name, -1)
            if step >= 0:
                versions.append((step, d))

    versions.sort(key=lambda x: x[0])
    seen_steps = set()
    unique_versions = []
    for step, d in versions:
        if step not in seen_steps:
            seen_steps.add(step)
            unique_versions.append((step, d))
    versions = unique_versions

    n = len(versions)
    print(f"Found {n} checkpoints (steps {versions[0][0]} to {versions[-1][0]})")

    # Pre-allocate arrays
    steps = np.zeros(n, dtype=np.int32)
    fourier_power = np.zeros((n, P))
    neuron_freqs = np.zeros((n, D_FFN), dtype=np.int32)
    embeddings_raw = np.zeros((n, P, D_MODEL))

    # Load pre-computed scalar metrics
    dynamics_path = Path("notebooks/data/grokking_dynamics.json")
    if dynamics_path.exists():
        with open(dynamics_path) as f:
            dynamics = json.load(f)
        dynamics_by_step = {d["step"]: d for d in dynamics}
    else:
        dynamics_by_step = {}

    accuracy = np.zeros(n)
    gini = np.zeros(n)
    norm = np.zeros(n)
    rnc1 = np.zeros(n)

    # Process each checkpoint
    for i, (step, vdir) in enumerate(versions):
        if i % 25 == 0:
            print(f"  Processing {i+1}/{n} (step {step})...")
        model = load_checkpoint(vdir)
        steps[i] = step
        fourier_power[i] = compute_fourier_power(model)
        neuron_freqs[i] = compute_neuron_freqs(model)
        embeddings_raw[i] = get_embedding_matrix(model)

        # Scalar metrics from pre-computed data
        if step in dynamics_by_step:
            d = dynamics_by_step[step]
            accuracy[i] = d["acc"]
            gini[i] = d["gini"]
            norm[i] = d["norm"]
            rnc1[i] = d["rnc1"]

    # Compute PCA of embeddings using final checkpoint's basis
    print("Computing PCA of embeddings (fixed basis from final checkpoint)...")
    final_emb = embeddings_raw[-1]  # [P, d_model]
    pca = PCA(n_components=2)
    pca.fit(final_emb)

    embedding_pca = np.zeros((n, P, 2))
    for i in range(n):
        embedding_pca[i] = pca.transform(embeddings_raw[i])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "grokking_animation.npz"
    np.savez_compressed(
        output_path,
        steps=steps,
        accuracy=accuracy,
        gini=gini,
        norm=norm,
        rnc1=rnc1,
        fourier_power=fourier_power,
        embedding_pca=embedding_pca,
        neuron_freqs=neuron_freqs,
    )
    print(f"\nSaved animation data to {output_path}")
    print(f"  steps: {steps.shape}")
    print(f"  fourier_power: {fourier_power.shape}")
    print(f"  embedding_pca: {embedding_pca.shape}")
    print(f"  neuron_freqs: {neuron_freqs.shape}")


if __name__ == "__main__":
    main()
