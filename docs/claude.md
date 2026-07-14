# Claude Code Development Guide

Documentation for AI assistants working on this project.

## Project Overview

This project analyzes grokking in transformer models trained on modular arithmetic (p=113).

Key findings:
- Different seeds discover different Fourier frequency bases
  - Seed 42: k=12, k=38 (2 independent freqs) → 100% accuracy
  - Seed 43: k=48 (1 independent freq) → 94-98% accuracy
  - Note: FFT symmetry on real data means |FFT[k]| = |FFT[p-k]|, so each freq appears as a conjugate pair of peaks
- QK circuit simplifies to rank-1 BEFORE grokking (by step 500)
- Frequency crossover (not switch) occurs during training — both k=12 and k=38 present throughout, relative power shifts
- Post-grokking cleanup continues (embedding rank drops even after 100% accuracy)
- Compared to Nanda et al.'s 5-frequency solution (k=14, 35, 41, 42, 52)
- Analyzed 4 training runs: no grok (27%), almost grok MPS (98.4%), almost grok CPU (94.5%), full grok (100%)
- See FINDINGS.md for literature context (Ding, Li, McCracken, Tian, Zhang, Xu, Prakash & Martin)

## Project Structure

```
.
├── configs/           # Training configurations
├── checkpoints/       # Model checkpoints and inventory
├── docs/             # Documentation
├── notebooks/        # Marimo notebooks for analysis
├── scripts/          # Utility scripts
└── src/              # Source code
    ├── data.py       # Data loading
    ├── model.py      # Model architecture
    ├── types.py      # Type definitions
    └── train.py      # Training logic
```

## Key Files

- `notebooks/grokking_story.py` - Narrative marimo notebook that tells the grokking story (recommended)
- `notebooks/grokking_explorer.py` - Interactive marimo notebook for exploring checkpoints
- `checkpoints/checkpoint_inventory.json` - Manifest of all available checkpoints
- `CHECKPOINT_STATUS.md` - Human-readable checkpoint summary
- `scripts/literature_validation.py` - Execution manifold, frequency analysis, ablation experiments
- `pyproject.toml` - Project dependencies and marimo configuration

## Working with Marimo Notebooks

See **[marimo-gotchas.md](marimo-gotchas.md)** for common pitfalls and solutions.

Key points:
- Variable names must be unique across all cells
- Consolidate imports in a single cell
- Project uses `pythonpath = ["."]` in pyproject.toml for imports

## Running the Project

View the grokking story (narrative, no interaction needed):
```bash
uv run marimo edit notebooks/grokking_story.py
```

Explore checkpoints interactively:
```bash
uv run marimo edit notebooks/grokking_explorer.py
```

Run training:
```bash
uv run python -m src.train
```

## Development Setup

1. Install dependencies: `uv sync`
2. Configure marimo pythonpath (already in pyproject.toml)
3. Launch notebooks from project root

## Important Notes

- Checkpoints are stored in WandB artifacts
- Model uses p=113 modular arithmetic
- Training config specifies device, batch size, etc.
- Marimo configuration requires restart after pyproject.toml changes
