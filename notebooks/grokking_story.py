import marimo

__generated_with = "0.18.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # The Grokking Story: From Memorization to Generalization

    We trained transformers on modular arithmetic (a + b mod 113) with different weight decay values.
    This notebook tells the story of what happens inside the model as it transitions from
    memorization (27% test accuracy) to perfect generalization (100% accuracy).

    ## Our Discovery: The 2-Frequency Solution

    While Nanda et al. found a 5-frequency Fourier solution, we discovered the model can grok
    using just **2 key frequencies: k=53 and k=60**. This notebook explores how this simpler
    solution emerges during training.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path
    import tomllib

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from numpy.fft import fft
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from src.types import TrainConfig
    from src.model import create_model
    return (
        Path,
        TrainConfig,
        create_model,
        fft,
        go,
        json,
        make_subplots,
        np,
        tomllib,
        torch,
    )


@app.cell
def _(Path, json):
    # Load checkpoint inventory
    _cwd = Path.cwd()
    if _cwd.name == "notebooks":
        project_root = _cwd.parent
    else:
        project_root = _cwd

    _inventory_path = project_root / "checkpoints" / "checkpoint_inventory.json"
    with _inventory_path.open("r") as _f:
        checkpoint_inventory = json.load(_f)

    # Select key checkpoints: final from each run
    checkpoints_to_analyze = {
        "No Grok (27% acc)": {
            "run_id": "5bon0t2j",
            "label": "final",
            "description": "Weight decay 2.0, memorization only",
        },
        "Almost Grok (98% acc)": {
            "run_id": "e332cujg",
            "label": "final",
            "description": "Weight decay 3.0, nearly perfect but not quite",
        },
        "Full Grok (100% acc)": {
            "run_id": "1z2q8rx3",
            "label": "final",
            "description": "Weight decay 3.0, perfect generalization",
        },
    }
    return checkpoint_inventory, checkpoints_to_analyze, project_root


@app.cell
def _(
    Path,
    TrainConfig,
    checkpoint_inventory,
    checkpoints_to_analyze,
    create_model,
    mo,
    project_root,
    tomllib,
    torch,
):
    mo.md("## Loading Models...")

    # Load all three models
    _models = {}
    _configs = {}

    for _stage_name, _ckpt_info in checkpoints_to_analyze.items():
        _run_id = _ckpt_info["run_id"]
        _label = _ckpt_info["label"]

        # Find checkpoint
        _run_ckpts = checkpoint_inventory[_run_id]["checkpoints"]
        _ckpt = next((_c for _c in _run_ckpts if _c["label"] == _label), None)

        if _ckpt is None:
            continue

        _ckpt_path = Path(_ckpt["path"])

        # Load config
        if _run_id == "1z2q8rx3":
            _cfg_path = project_root / "configs/ultra_dense_grokking.toml"
        else:
            _cfg_path = project_root / "configs/grokking_100pct.toml"

        with open(_cfg_path, "rb") as _f2:
            _cfg_dict = tomllib.load(_f2)

        _config = TrainConfig(**_cfg_dict)
        _model = create_model(_config)
        _state_dict = torch.load(_ckpt_path, map_location="cpu")
        _model.load_state_dict(_state_dict)
        _model.eval()

        _models[_stage_name] = _model
        _configs[_stage_name] = _config

    models = _models
    configs = _configs

    mo.md(f"✓ Loaded {len(models)} models")
    return configs, models


@app.cell
def _(configs, fft, go, make_subplots, mo, models, np):
    mo.md("""
    ## Act I: The Fourier Structure Emerges

    As the model groks, it learns to represent numbers using Fourier features.
    We can see this by taking the FFT of the embedding matrix and measuring
    how concentrated the power is across different frequencies.
    """)

    # Analyze Fourier structure for available stages
    _fourier_results = {}

    _stage_candidates = [
        "No Grok (27% acc)",
        "Almost Grok (98% acc)",
        "Full Grok (100% acc)",
    ]
    _stage_order = [s for s in _stage_candidates if s in models]
    for _s in models.keys():
        if _s not in _stage_order:
            _stage_order.append(_s)

    for _stage_name in _stage_order:
        _model = models[_stage_name]
        _config = configs[_stage_name]
        _p = _config.p

        # Get embeddings
        _W_E = _model.W_E.detach().cpu().numpy()[:_p, :]

        # FFT analysis
        _fft_components = np.abs(fft(_W_E, axis=0))
        _total_power = _fft_components.sum(axis=1)
        _denom = _total_power.sum()
        if _denom == 0:
            _total_power = np.zeros_like(_total_power)
        else:
            _total_power = _total_power / _denom

        # Compute metrics
        def _gini(x):
            x = np.abs(x.flatten())
            if x.sum() == 0:
                return 0.0
            sorted_x = np.sort(x)
            n = len(sorted_x)
            cumx = np.cumsum(sorted_x)
            return (n + 1 - 2 * (cumx / cumx[-1]).sum()) / n

        _gini_power = _gini(_total_power)
        _topk = min(5, len(_total_power))
        _topk_power = float(np.sort(_total_power)[-(_topk):].sum())
        _k53_power = float(_total_power[53]) if 53 < len(_total_power) else 0.0
        _k60_power = float(_total_power[60]) if 60 < len(_total_power) else 0.0

        _fourier_results[_stage_name] = {
            "total_power": _total_power,
            "gini": _gini_power,
            "top5": _topk_power,
            "k53": _k53_power,
            "k60": _k60_power,
        }

    fourier_results = _fourier_results

    # Create comparison plot
    if len(fourier_results) == 0:
        _fig = go.Figure()
        _fig.update_layout(
            height=400, title_text="Fourier Spectrum Evolution (no models loaded)"
        )
    else:
        _fig = make_subplots(
            rows=1,
            cols=len(fourier_results),
            subplot_titles=list(fourier_results.keys()),
            horizontal_spacing=0.08,
        )

        for _idx, (_stage_name, _results) in enumerate(fourier_results.items(), 1):
            _fig.add_trace(
                go.Bar(
                    x=np.arange(len(_results["total_power"])),
                    y=_results["total_power"],
                    name=_stage_name,
                    showlegend=False,
                    marker_color="#1f77b4",
                    hovertemplate="k=%{x}<br>Power=%{y:.4f}<extra></extra>",
                ),
                row=1,
                col=_idx,
            )
            # Highlight k=53 and k=60
            _fig.add_vline(
                x=53,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                row=1,
                col=_idx,
            )
            _fig.add_vline(
                x=60,
                line_dash="dash",
                line_color="orange",
                opacity=0.7,
                row=1,
                col=_idx,
            )

        for _c in range(1, len(fourier_results) + 1):
            _fig.update_xaxes(title_text="Frequency k", row=1, col=_c)
        _fig.update_yaxes(title_text="Normalized Power", row=1, col=1)
        _fig.update_layout(height=400, title_text="Fourier Spectrum Evolution")

    # Build dynamic table
    _table_rows = [
        "### Key Observations:",
        "",
        "| Stage | Gini (sparsity) | Top-5 Power | k=53 Power | k=60 Power |",
        "|-------|------------------|-------------|------------|------------|",
    ]
    for _stage_name, _res in fourier_results.items():
        _table_rows.append(
            f"| {_stage_name} | {_res['gini']:.3f} | {_res['top5']:.3f} | {_res['k53']:.4f} | {_res['k60']:.4f} |"
        )

    _table_rows.extend(
        [
            "",
            "**Interpretation:**",
            "- **Gini coefficient** measures sparsity (higher = more concentrated in few frequencies)",
            "- As the model groks, power concentrates in k=53 and k=60 (shown by red/orange dashed lines)",
            "- The no-grok model has diffuse power across many frequencies (memorization)",
            "- The grokked model has sharp peaks at specific frequencies (algorithmic solution)",
        ]
    )

    mo.vstack([_fig, mo.md("\n".join(_table_rows))])
    return


@app.cell
def _(configs, fft, mo, models, np):
    mo.md("""
    ## Act II: Neurons Specialize by Frequency

    The MLP neurons learn to detect specific frequencies. We can see this by
    projecting each neuron's input weights through the embeddings and finding
    which frequency dominates its activation pattern.
    """)

    # Analyze neuron specialization
    _neuron_results = {}

    for _stage_name, _model in models.items():
        _config = configs[_stage_name]
        _p = _config.p

        _W_in = _model.blocks[0].mlp.W_in.detach().cpu().numpy()
        _W_E = _model.W_E.detach().cpu().numpy()[:_p, :]

        _d_mlp = _W_in.shape[1]
        _neuron_dom_freq = np.zeros(_d_mlp, dtype=int)

        for _i in range(_d_mlp):
            _w_in_neuron = _W_in[:, _i]
            _projected = _W_E @ _w_in_neuron
            _fft_neuron = np.abs(fft(_projected))
            _k = int(np.argmax(_fft_neuron[1:]) + 1)
            _neuron_dom_freq[_i] = _k

        _unique, _counts = np.unique(_neuron_dom_freq, return_counts=True)
        _order = np.argsort(_counts)[::-1]

        _neuron_results[_stage_name] = {
            "unique_freqs": _unique[_order],
            "counts": _counts[_order],
            "n_distinct": len(_unique),
            "total_neurons": _d_mlp,
        }

    neuron_results = _neuron_results

    # Build comparison table
    _rows = [
        "| Stage | Distinct Frequencies | Top Frequency | # Neurons at Top Freq |",
        "|-------|---------------------|---------------|----------------------|",
    ]

    for _stage_name, _results in neuron_results.items():
        _top_freq = int(_results["unique_freqs"][0])
        _top_count = int(_results["counts"][0])
        _n_distinct = _results["n_distinct"]
        _rows.append(
            f"| {_stage_name} | {_n_distinct} / 128 | k={_top_freq} | {_top_count} |"
        )

    mo.md(f"""
    {chr(10).join(_rows)}

    **Interpretation:**
    - As grokking occurs, neurons increasingly specialize on fewer frequencies
    - The grokked model has many neurons tuned to k=53 and k=60
    - This specialization enables the model to compute modular arithmetic algorithmically
    """)
    return


@app.cell
def _(configs, mo, models, torch):
    mo.md("""
    ## Act III: Perfect Generalization

    The ultimate test: does the model get *every* possible input correct?
    We evaluate on the full grid of all p² = 12,769 possible (a, b) pairs.
    """)

    # Test all three models on full grid
    _acc_results = {}

    for _stage_name, _model in models.items():
        _config = configs[_stage_name]
        _p = _config.p
        _device = next(_model.parameters()).device

        _a = torch.arange(_p, device=_device).repeat_interleave(_p)
        _b = torch.arange(_p, device=_device).repeat(_p)
        _prompts = torch.stack([_a, _b, torch.full_like(_a, _p)], dim=1)
        _labels = (_a + _b) % _p

        with torch.no_grad():
            _logits = _model(_prompts)[:, -1, :]

        _preds = _logits.argmax(dim=-1)
        _acc = (_preds == _labels).float().mean().item()

        # Logit margins
        _target_logits = _logits.gather(1, _labels.view(-1, 1)).squeeze(1)
        _tmp = _logits.clone()
        _tmp.scatter_(1, _labels.view(-1, 1), float("-inf"))
        _best_other = _tmp.max(dim=1).values
        _margins = (_target_logits - _best_other).cpu()

        _acc_results[_stage_name] = {
            "accuracy": _acc,
            "min_margin": float(_margins.min()),
            "mean_margin": float(_margins.mean()),
            "margins": _margins.numpy(),
        }

    acc_results = _acc_results

    # Summary table
    _summary_rows = [
        "| Stage | Accuracy | Min Margin | Mean Margin |",
        "|-------|----------|------------|-------------|",
    ]

    for _stage_name, _results in acc_results.items():
        _summary_rows.append(
            f"| {_stage_name} | {_results['accuracy']:.4f} | "
            f"{_results['min_margin']:.2f} | {_results['mean_margin']:.2f} |"
        )

    mo.md(f"""
    {chr(10).join(_summary_rows)}

    **Interpretation:**
    - **Accuracy**: Fraction of correct predictions on all 12,769 inputs
    - **Logit margin**: How confident the model is (true logit - best wrong logit)
    - The grokked model achieves 100% accuracy with large positive margins
    - The no-grok model is essentially guessing (27% ≈ random for p=113)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Conclusion: The Path to Grokking

    By increasing weight decay from 2.0 to 3.0, we observe a phase transition:

    1. **Fourier Structure**: Power concentrates in k=53 and k=60
    2. **Neuron Specialization**: MLPs learn frequency-specific detectors
    3. **Perfect Generalization**: 100% accuracy on all possible inputs

    This happens because weight decay prevents memorization and forces the model
    to find the simpler algorithmic solution based on Fourier features.

    ### The 2-Frequency Solution

    Our key finding: the model can grok with just 2 frequencies (k=53, k=60),
    simpler than Nanda et al.'s 5-frequency solution. This suggests there are
    multiple algorithmic solutions to modular arithmetic, and weight decay
    biases the model toward finding sparser ones.
    """)
    return


if __name__ == "__main__":
    app.run()
