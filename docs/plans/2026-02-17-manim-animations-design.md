# Manim Animations: Seeing Cognition Happen

**Date:** 2026-02-17

## Overview

Animate the grokking transition in a transformer trained on (a+b) mod 113, using 449 ultra-dense checkpoints (every 20 steps). Two-phase pipeline: extract data first, then animate with Manim.

## Pipeline

```
checkpoints/1z2q8rx3/v0-v448/  →  scripts/extract_animation_data.py  →  animations/data/grokking_animation.npz  →  animations/grokking.py  →  media/videos/
```

## Data Extraction (`scripts/extract_animation_data.py`)

Processes all 449 checkpoints, saves to `.npz`:

| Array | Shape | Description |
|-------|-------|-------------|
| `steps` | `[449]` | Training step for each checkpoint |
| `accuracy` | `[449]` | Test accuracy |
| `gini` | `[449]` | Fourier Gini coefficient |
| `norm` | `[449]` | L2 weight norm |
| `rnc1` | `[449]` | Within-class variance |
| `fourier_power` | `[449, 113]` | FFT power spectrum of W_E per step |
| `embedding_pca` | `[449, 113, 2]` | W_E[:113,:] projected to 2D (fixed PCA basis from final checkpoint) |
| `neuron_freqs` | `[449, 128]` | Dominant frequency per MLP neuron per step |

## Manim Scenes (`animations/grokking.py`)

### Scene 1: `FourierSpectrumRace` (~30s)
- Top 15 frequencies as horizontal bars, ranked by power
- Bars reorder as rankings change
- Conjugate pairs share colors (k=12/101 blue, k=38/75 orange)
- Step counter + accuracy overlay
- Highlight crossover at step ~3440

### Scene 2: `EmbeddingCircleFormation` (~30s)
- 113 dots in 2D (PCA projection)
- Color = number value (0-112), smooth colormap
- Random cloud → circles
- Gini + accuracy text overlay

### Scene 3: `NeuronCollapse` (~20s)
- Histogram: frequency bins × neuron count
- 128 neurons across ~43 frequencies → collapse to 2-3
- Smooth bar height transitions

### Scene 4: `GrokkingDashboard` (~45s)
- 2×2 panels synchronized via time cursor:
  - Top-left: accuracy curve
  - Top-right: Fourier heatmap
  - Bottom-left: weight norm curve
  - Bottom-right: neuron frequency histogram
- Phase labels appear at transitions

## Dependencies
- `manim` added to pyproject.toml
