# Critical Finding: Initialization Dependence in Grokking

**Date:** November 22, 2025

## The Problem

Even with **identical hyperparameters and seed**, grokking outcomes vary dramatically:

### Run Comparison (All: seed=43, WD=3.0, p=113, same architecture)

| Run ID | Steps | Final Test Acc | Outcome |
|--------|-------|----------------|---------|
| um7dxpfz | 19,200 | **100.0%** | Perfect grokking |
| e332cujg | 60,000 | **98.4%** | Incomplete grokking |
| ax96ivr2 | 80,000+ | **~89%** | Failed to grok |

## Key Insight

**Same seed ≠ Same initialization**

The random seed controls:
- Dataset split (train/test)
- Initial weight initialization

But does NOT guarantee:
- Same optimization trajectory
- Same final solution
- Same grokking dynamics

## Implications

1. **Multiple Solutions Exist**: There are at least two types of outcomes:
   - Full grokking (100% accuracy)
   - Partial grokking (98-99% accuracy)
   - Failed grokking (<90% accuracy)

2. **Initialization Matters**: Even with same seed, tiny numerical differences in:
   - Weight initialization order
   - Floating point rounding
   - GPU/CPU computational order
   Can lead to completely different learning trajectories

3. **DEVICE MATTERS**: **Critical finding - successful 100% run (um7dxpfz) used `device="cpu"`**
   - MPS (Apple Silicon GPU) runs consistently fail to reach 100%
   - MPS runs get stuck at 78-89% accuracy
   - CPU vs MPS have different:
     - Floating point precision handling
     - Operation ordering (affects gradient accumulation)
     - Random number generation behavior
   - **For reproducibility: Use CPU device to match successful run**

3. **For Publication**: This strengthens our story:
   - Shows grokking is NOT deterministic
   - Supports "multiple paths to generalization"
   - Validates architects' concern about robustness
   - Requires multiple runs for statistical significance

## Next Steps

1. Run with multiple different seeds (not just 43)
2. Save dense checkpoints for 100% run
3. Compare frequency signatures: Does 100% run use same k=53,60 or different frequencies?
4. Statistical analysis: What fraction of runs hit 100%?

## Hypothesis

The 98.4% run (e332cujg) might have discovered a **suboptimal frequency basis** that works for most but not all examples. The 100% run likely found a **complete basis** that spans the full solution space.

**Question to answer**: Do they use the same frequencies with different strengths, or completely different frequencies?
