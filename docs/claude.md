# Claude Code Development Guide

Documentation for AI assistants working on this project.

## Project Overview

This project analyzes grokking in transformer models trained on modular arithmetic (p=113).

Key findings:
- Discovered a 2-frequency solution (k=53, k=60)
- Compared to Nanda et al.'s 5-frequency solution
- Analyzed 3 training runs: no grok (27% acc), almost grok (98.4% acc), full grok (100% acc)

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
