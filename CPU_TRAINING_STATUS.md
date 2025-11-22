# CPU Training Status - Seed 42

**Start Time:** 2025-11-22 14:55 (2:55 PM)
**Run ID:** apwl1xw2
**Config:** configs/perfect_grokking_seed42.toml
**Device:** CPU (to match successful um7dxpfz run)

## Key Difference from Failed Runs

**Previous runs (FAILED to reach 100%):**
- device = "mps" (Apple Silicon GPU)
- Stuck at 78-89% accuracy
- Seeds: 42, 43

**Successful reference run (um7dxpfz):**
- device = "cpu"
- Reached 100% accuracy at step 19,200
- seed = 43

**Current run:**
- device = "cpu" ✓ (matches successful run)
- seed = 42 (different from reference, testing generality)
- All other params identical to um7dxpfz

## Hypothesis

MPS (Apple Silicon GPU) has different numerical behavior vs CPU:
- Different floating point precision handling
- Different operation ordering (affects gradient accumulation)
- Different random number generation
- These tiny differences lead to different optimization trajectories

**Prediction:** CPU training should reach 100% accuracy, validating the device hypothesis.

## Monitoring

Check progress:
```bash
tail -f train_cpu_seed42.log
```

wandb: https://wandb.ai/discoelysium-neuromatch/mod-arith-grokking/runs/apwl1xw2

## Expected Timeline

Based on um7dxpfz (CPU):
- Step 19,200: Reached 100% test accuracy
- Grokking should occur between 10k-20k steps
- Dense checkpoints every 500 steps in grokking zone (20k-40k)
- Total run: 60,000 steps

CPU is slower than MPS (~3-5x), so expect:
- ~30-60 minutes for first checkpoint (1000 steps)
- ~2-4 hours to reach grokking zone
- ~6-10 hours total runtime
