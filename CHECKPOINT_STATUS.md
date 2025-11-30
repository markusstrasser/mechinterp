# Checkpoint Availability Summary

**Status:** All required checkpoints are downloaded and available locally
**Last Updated:** November 29, 2025

## 4-Way Comparison Runs

### Run 1: No Grokking (Control)
- **Run ID:** `5bon0t2j`
- **Config:** WD=2.0, seed=43, CPU
- **Final Accuracy:** 27%
- **Dominant Frequencies:** k=48, 65 (same as seed=43 grokked runs!)
- **Gini Coefficient:** 0.07 (diffuse - no concentration)
- **Checkpoints:** 27 total
  - Named: early, mid, mid_training, final
  - Version artifacts: v0-v22
- **Path:** `checkpoints/5bon0t2j/`

### Run 2: Almost Grokked (MPS)
- **Run ID:** `e332cujg`
- **Config:** WD=3.0, seed=43, MPS (Apple Silicon)
- **Final Accuracy:** 98.4%
- **Dominant Frequencies:** k=48, 65 (conjugate pair: 48+65=113)
- **Gini Coefficient:** 0.50
- **Grokking:** Occurred at steps 42,000-44,000
- **Checkpoints:** 51 total
  - Named: early, pre_grok, pre_grok_plateau, just_before_grok, grokking_moment, post_grok_stable, final, mid_training
  - Version artifacts: v0-v42
- **Path:** `checkpoints/e332cujg/`

### Run 3: Full Grokking (Ultra-Dense)
- **Run ID:** `1z2q8rx3`
- **Config:** WD=3.0, seed=42, CPU, ultra-dense checkpointing
- **Final Accuracy:** 100%
- **Dominant Frequencies:** k=12, 101 (conjugate pair: 12+101=113)
- **Gini Coefficient:** 0.68
- **Grokking:** Occurred at step ~3,900 (VERY EARLY!)
- **Checkpoints:** 449 total
  - Named: early, pre_grok_1, pre_grok_2, grokking_moment, post_grok_1, post_grok_2, stable, final
  - Version artifacts: v0-v440
  - **Checkpoint Schedule:** (from configs/ultra_dense_grokking.toml)
    - Every 20 steps from 20 to 8,000
    - Every 100 steps from 8,100 to 10,000
    - Every 500 steps from 10,500 to 20,000
    - Sparse after 20k (25k, 30k, 35k, 40k, 45k, 50k, 55k, 60k)
- **Path:** `checkpoints/1z2q8rx3/`

### Run 4: CPU Replication of seed=43
- **Run ID:** `l3wye1yc`
- **Config:** WD=3.0, seed=43, CPU
- **Final Accuracy:** 94.5%
- **Dominant Frequencies:** k=48, 65 (same as e332cujg)
- **Gini Coefficient:** ~0.48
- **Grokking:** Occurred at step ~1,000 (much earlier than MPS!)
- **Checkpoints:** ~19 (sparse)
- **Path:** W&B artifact (not downloaded locally)
- **Purpose:** Test if CPU vs MPS explains accuracy differences

## Summary Table

| Run | Seed | WD | Device | Final Acc | Dominant Freqs | Gini | Grok Step |
|-----|------|-----|--------|-----------|----------------|------|-----------|
| 5bon0t2j | 43 | 2.0 | CPU | **27%** | k=48, 65 | 0.07 | Never |
| e332cujg | 43 | 3.0 | MPS | **98.4%** | k=48, 65 | 0.50 | ~42k |
| l3wye1yc | 43 | 3.0 | CPU | **94.5%** | k=48, 65 | ~0.48 | ~1k |
| 1z2q8rx3 | 42 | 3.0 | CPU | **100%** | k=12, 101 | 0.68 | ~3.9k |

## Critical Observations (Updated)

### 1. Seed Determines Frequency Basis
Different seeds discover different conjugate frequency pairs:
- **Seed 42:** k=12, 101 (12 + 101 = 113)
- **Seed 43:** k=48, 65 (48 + 65 = 113)

Both are valid Fourier bases for computing (a+b) mod 113!

### 2. Device Does NOT Explain Accuracy Differences
Original hypothesis: MPS (98.4%) vs CPU might explain why some models don't reach 100%.

**Result:** CPU with seed=43 achieved only **94.5%** - actually WORSE than MPS!

**Conclusion:** The seed (and resulting frequency basis) determines final accuracy, not the device.

### 3. Same Seed, Same Frequencies, Different Accuracies
| Run | Seed | Device | Frequencies | Accuracy |
|-----|------|--------|-------------|----------|
| e332cujg | 43 | MPS | k=48, 65 | 98.4% |
| l3wye1yc | 43 | CPU | k=48, 65 | 94.5% |
| 5bon0t2j | 43 | CPU | k=48, 65 | 27% |

All three seed=43 runs found the SAME frequency basis, but achieved vastly different accuracies. This suggests:
- Weight decay strength is critical (WD=2.0 → 27%, WD=3.0 → 94-98%)
- Training trajectory matters even with same seed/frequencies

### 4. Grokking Time Varies Wildly
- Seed 42 (1z2q8rx3): Grokked at 3.9k steps
- Seed 43 (l3wye1yc): Grokked at ~1k steps (but plateaued at 94.5%)
- Seed 43 (e332cujg): Grokked at ~42k steps (reached 98.4%)

## Geometric Analysis Results

Using the 449 ultra-dense checkpoints from 1z2q8rx3, we discovered:

1. **QK circuit simplifies FIRST** - becomes rank-1 by step 500, long before grokking
2. **Frequency switching occurs** - model tried k=38,75 before settling on k=12,101
3. **Neuron collapse is dramatic** - from 43 unique frequencies → 2-3 frequencies
4. **Post-grokking cleanup** - embedding rank drops 21 → 13 even after 100% accuracy

See FINDINGS.md for complete analysis.

## Inventory File

Full checkpoint inventory with paths: `checkpoints/checkpoint_inventory.json`
