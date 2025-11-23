# Checkpoint Availability Summary

**Status:** ✅ All required checkpoints are downloaded and available locally

## 3-Way Comparison Runs

### Run 1: No Grokking (Control)
- **Run ID:** `5bon0t2j`
- **Config:** WD=2.0, seed=43, CPU
- **Final Accuracy:** 27%
- **Checkpoints:** 27 total
  - Named: early, mid, mid_training, final
  - Version artifacts: v0-v22
- **Path:** `checkpoints/5bon0t2j/`

### Run 2: Almost Grokked
- **Run ID:** `e332cujg`
- **Config:** WD=3.0, seed=43, MPS (Apple Silicon)
- **Final Accuracy:** 98.4%
- **Grokking:** Occurred at steps 42,000-44,000
- **Checkpoints:** 51 total
  - Named: early, pre_grok, pre_grok_plateau, just_before_grok, grokking_moment, post_grok_stable, final, mid_training
  - Version artifacts: v0-v42
- **Path:** `checkpoints/e332cujg/`

### Run 3: Full Grokking (Ultra-Dense)
- **Run ID:** `1z2q8rx3`
- **Config:** WD=3.0, seed=42, CPU, ultra-dense checkpointing
- **Final Accuracy:** 100%
- **Grokking:** Occurred at step ~3,900 (VERY EARLY!)
- **Checkpoints:** 449 total (!!!)
  - Named: early, pre_grok_1, pre_grok_2, grokking_moment, post_grok_1, post_grok_2, stable, final
  - Version artifacts: v0-v440
  - **Checkpoint Schedule:** (from configs/ultra_dense_grokking.toml)
    - Every 20 steps from 20 to 8,000
    - Every 100 steps from 8,100 to 10,000
    - Every 500 steps from 10,500 to 20,000
    - Sparse after 20k (25k, 30k, 35k, 40k, 45k, 50k, 55k, 60k)
- **Path:** `checkpoints/1z2q8rx3/`

## Key Differences

| Run | Device | Seed | WD | Grok Step | Final Acc | Checkpoint Density |
|-----|--------|------|----|-----------|-----------|--------------------|
| 5bon0t2j | CPU | 43 | 2.0 | N/A | 27% | Sparse (27) |
| e332cujg | MPS | 43 | 3.0 | 42-44k | 98.4% | Medium (51) |
| 1z2q8rx3 | CPU | 42 | 3.0 | ~3.9k | 100% | **Ultra-dense (449)** |

## Critical Observations

1. **Device Matters:** Run e332cujg on MPS reached only 98.4%, while CPU runs reached 100%
2. **Seed Variability:** Different seeds (42 vs 43) produced WILDLY different grokking times:
   - Seed 42 (1z2q8rx3): Grokked at 3.9k steps
   - Seed 43 (um7dxpfz, not in this set): Grokked at 19.2k steps
   - That's a 4.9x difference!
3. **Ultra-Dense Coverage:** Run 1z2q8rx3 has checkpoints every 20 steps during the critical grokking period (0-8k), perfect for analyzing the phase transition

## Next Steps: Fourier Analysis

We can now run deep Fourier analysis on all 3 runs to compare:

1. **Frequency discovery:** Do all 3 runs discover the same frequencies (k=53, 60)?
2. **No-grok control:** Does 5bon0t2j stay diffuse across all frequencies?
3. **Incomplete grok:** Does e332cujg (98.4%) have partial frequency concentration?
4. **Phase transition dynamics:** Using ultra-dense 1z2q8rx3 checkpoints, can we see the exact moment neurons specialize?

### Analysis Commands

```bash
# Analyze all 3 runs
uv run python scripts/deep_fourier_analysis.py \
  --run-ids 5bon0t2j e332cujg 1z2q8rx3 \
  --checkpoints early grokking_moment final

# Ultra-dense timeline for 1z2q8rx3
uv run python scripts/deep_fourier_analysis.py \
  --run-id 1z2q8rx3 \
  --all-checkpoints \
  --output-dir analysis/ultra_dense_timeline
```

## Inventory File

Full checkpoint inventory with paths: `checkpoints/checkpoint_inventory.json`
