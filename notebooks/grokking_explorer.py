import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Grokking Explorer

    Interactive analysis of modular arithmetic grokking across 3 runs:
    - No grok (wd=2.0, 27% accuracy)
    - Almost grok (wd=3.0, 98.4% accuracy)
    - Full grok (wd=3.0, 100% accuracy, ultra-dense checkpoints)
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from numpy.fft import fft

    from src.types import TrainConfig
    from src.model import create_model
    return Path, TrainConfig, create_model, fft, json, np, plt, torch


@app.cell
def _(Path, json, mo):
    # Load checkpoint inventory
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        project_root = cwd.parent
    else:
        project_root = cwd

    inventory_path = project_root / "checkpoints" / "checkpoint_inventory.json"

    with inventory_path.open("r") as f:
        checkpoint_inventory = json.load(f)

    # Run selector
    run_labels = {
        "No grok (wd=2.0, 5bon0t2j)": "5bon0t2j",
        "Almost grok (wd=3.0, e332cujg)": "e332cujg",
        "Full grok (wd=3.0, 1z2q8rx3)": "1z2q8rx3",
    }

    run_select = mo.ui.dropdown(
        options=list(run_labels.keys()),
        value="No grok (wd=2.0, 5bon0t2j)",
        label="Run",
    )
    return checkpoint_inventory, project_root, run_labels, run_select


@app.cell
def _(checkpoint_inventory, mo, run_labels, run_select):
    # Get checkpoints for selected run
    run_id = run_labels[run_select.value]
    run_info = checkpoint_inventory[run_id]
    ckpts = run_info["checkpoints"]

    # Sort by step
    ckpts_sorted = sorted(ckpts, key=lambda c: c.get("step", 0))

    # Checkpoint selector
    def _fmt(c):
        step = c.get("step", "NA")
        acc = c.get("test_acc", None)
        if acc is None:
            return f"{c['label']} @ step {step}"
        return f"{c['label']} @ step {step} (acc={acc:.3f})"

    options = [_fmt(c) for c in ckpts_sorted]
    ckpt_dropdown = mo.ui.dropdown(
        options=options,
        value=options[0],
        label="Checkpoint",
    )

    mo.vstack([
        mo.md(f"**Run:** {run_id} ({len(ckpts_sorted)} checkpoints)"),
        ckpt_dropdown
    ])
    return ckpt_dropdown, ckpts_sorted, run_id


@app.cell
def _(Path, ckpt_dropdown, ckpts_sorted):
    # Parse selected checkpoint
    selected_label = ckpt_dropdown.value.split("@", 1)[0].strip()
    chosen_ckpt = next(
        (c for c in ckpts_sorted if c["label"] == selected_label),
        ckpts_sorted[0]
    )
    ckpt_path = Path(chosen_ckpt["path"])
    return chosen_ckpt, ckpt_path


@app.cell
def _(TrainConfig, ckpt_path, create_model, project_root, run_id, torch):
    # Load model
    import tomllib

    if run_id == "1z2q8rx3":
        cfg_path = project_root / "configs/ultra_dense_grokking.toml"
    else:
        cfg_path = project_root / "configs/grokking_100pct.toml"

    with open(cfg_path, "rb") as f2:
        cfg_dict = tomllib.load(f2)

    config = TrainConfig(**cfg_dict)
    model = create_model(config)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return config, model


@app.cell
def _(chosen_ckpt, config, fft, mo, model, np, plt):
    # Fourier analysis of embeddings
    p = config.p
    W_E = model.W_E.detach().cpu().numpy()[:p, :]

    fft_components = np.abs(fft(W_E, axis=0))
    total_power = fft_components.sum(axis=1)
    total_power = total_power / total_power.sum()

    # Gini coefficient
    def gini(x):
        x = np.abs(x.flatten())
        if x.sum() == 0:
            return 0.0
        sorted_x = np.sort(x)
        n = len(sorted_x)
        cumx = np.cumsum(sorted_x)
        return (n + 1 - 2 * (cumx / cumx[-1]).sum()) / n

    gini_power = gini(total_power)

    # Top frequencies
    topk_idx = np.argsort(total_power)[-10:][::-1]
    topk_power = total_power[topk_idx]
    top5_frac = float(topk_power[:5].sum())

    k53_power = float(total_power[53]) if 53 < len(total_power) else 0.0
    k60_power = float(total_power[60]) if 60 < len(total_power) else 0.0

    # Plot
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.bar(np.arange(len(total_power)), total_power, width=1.0)
    ax1.set_xlabel("Frequency k")
    ax1.set_ylabel("Normalized power")
    ax1.set_title("Embedding Fourier spectrum")
    ax1.axvline(53, color="red", linestyle="--", alpha=0.7, label="k=53")
    ax1.axvline(60, color="orange", linestyle="--", alpha=0.7, label="k=60")
    ax1.legend()
    plt.close()

    mo.vstack([
        mo.md(f"""
        ### Fourier structure – embeddings

        **Checkpoint:** `{chosen_ckpt['label']}` (step {chosen_ckpt.get('step', 'NA')})

        - Gini (sparsity): `{gini_power:.3f}`
        - Top 5 fraction: `{top5_frac:.3f}`
        - k=53 power: `{k53_power:.4f}`
        - k=60 power: `{k60_power:.4f}`
        """),
        mo.as_html(fig1)
    ])
    return


@app.cell
def _(chosen_ckpt, config, fft, mo, model, np):
    # Neuron frequency specialization
    p2 = config.p
    W_in = model.blocks[0].mlp.W_in.detach().cpu().numpy()
    W_E2 = model.W_E.detach().cpu().numpy()[:p2, :]

    d_mlp = W_in.shape[1]
    neuron_dom_freq = np.zeros(d_mlp, dtype=int)

    for i in range(d_mlp):
        w_in_neuron = W_in[:, i]
        projected = W_E2 @ w_in_neuron
        fft_neuron = np.abs(fft(projected))
        k = int(np.argmax(fft_neuron[1:]) + 1)
        neuron_dom_freq[i] = k

    unique, counts = np.unique(neuron_dom_freq, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique = unique[order]
    counts = counts[order]

    # Build table
    rows = ["| k | #neurons | frac |", "|---|----------|------|"]
    total = counts.sum()
    for k, c in zip(unique[:10], counts[:10]):
        rows.append(f"| {int(k)} | {int(c)} | {c/total:.3f} |")

    table_md = "\n".join(rows)

    mo.md(f"""
    ### Neuron frequency specialization

    **Checkpoint:** `{chosen_ckpt['label']}`
    **Distinct frequencies:** {len(unique)} / {len(neuron_dom_freq)}

    Top 10:

    {table_md}
    """)
    return


@app.cell
def _(chosen_ckpt, config, mo, model, plt, torch):
    # Full-grid exactness
    p3 = config.p

    # Get model's device
    device = next(model.parameters()).device

    a = torch.arange(p3, device=device).repeat_interleave(p3)
    b = torch.arange(p3, device=device).repeat(p3)
    prompts = torch.stack([a, b, torch.full_like(a, p3)], dim=1)
    labels = (a + b) % p3

    with torch.no_grad():
        logits = model(prompts)[:, -1, :]

    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()

    target_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    tmp = logits.clone()
    tmp.scatter_(1, labels.view(-1, 1), float("-inf"))
    best_other = tmp.max(dim=1).values
    margins = (target_logits - best_other).cpu()

    min_margin = float(margins.min())
    mean_margin = float(margins.mean())

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.hist(margins.numpy(), bins=50)
    ax2.set_xlabel("Logit margin (true − best other)")
    ax2.set_ylabel("Count")
    ax2.set_title("Full-grid logit margins")
    plt.close()

    mo.vstack([
        mo.md(f"""
        ### Exactness on full (a, b) grid

        **Checkpoint:** `{chosen_ckpt['label']}`
        - Accuracy: `{acc:.4f}`
        - Min margin: `{min_margin:.4f}`
        - Mean margin: `{mean_margin:.4f}`
        """),
        mo.as_html(fig2)
    ])
    return


if __name__ == "__main__":
    app.run()
