# Grokking Analysis Findings
**Date:** November 29, 2025 (Updated February 2026)
**Analysis of:** Modular arithmetic transformer (p=113) trained with weight decay 3.0

---

## Executive Summary

We replicated and extended Nanda et al.'s (ICLR 2023) mechanistic interpretability analysis of grokking. Our analysis reveals that **different random seeds discover different Fourier frequency bases**, and we tracked geometric changes across 449 ultra-dense checkpoints to understand the exact structural evolution during grokking.

**Epistemic note (Feb 2026):** Many findings initially reported as novel have since been explained by recent theoretical work. We have updated this document to distinguish confirmed observations from overclaims, and to contextualize our results within the current literature.

### Key Observations
1. **Seed determines frequency basis:** Different seeds find different frequencies (k=12,38 for seed 42 vs k=48 for seed 43) — explained by Tian (2025) as semi-ring algebraic structure of solution space
2. **QK circuit simplifies FIRST:** Attention becomes rank-1 by step 500, long before grokking — consistent with Nanda et al.
3. **Frequency crossover during training:** Relative power between competing frequencies shifts during transition — consistent with Ding et al. (2024) Lotka-Volterra dynamics
4. **Post-grokking cleanup:** Network continues compressing representations after reaching 100% accuracy — consistent with Yunis et al. (2024) rank minimization framework

---

## 1. Model Comparison

We trained multiple models with varying seeds, weight decay, and devices:

| Run ID | Seed | WD | Device | Final Acc | Independent Freqs | FFT Peaks (with mirrors) | Gini | Grokking Step |
|--------|------|-----|--------|-----------|-------------------|--------------------------|------|---------------|
| **1z2q8rx3** | 42 | 3.0 | CPU | **100%** | k=12, 38 | k=12, 38, 75, 101 | 0.68 | ~3,900 |
| **e332cujg** | 43 | 3.0 | MPS | **98.4%** | k=48 | k=48, 65 | 0.50 | ~42,000 |
| **l3wye1yc** | 43 | 3.0 | CPU | **94.5%** | k=48 | k=48, 65 | ~0.48 | ~1,000 |
| **5bon0t2j** | 43 | 2.0 | CPU | **27%** | (diffuse) | (diffuse) | 0.07 | Never |

**Note on conjugate symmetry:** For real-valued embedding matrices, the FFT satisfies |FFT[k]| = |FFT[p-k]|, so each independent frequency k appears as a pair of peaks at k and p-k. The frequencies k=12 and k=101=113-12 carry identical information, as do k=38 and k=75=113-38. The model uses **2 independent frequencies** (seed 42) or **1 independent frequency** (seed 43), not 4 or 2 respectively.

**Key Insight:** The specific frequencies depend on the random seed. This is explained by Tian (NeurIPS 2025) who shows the solution space has semi-ring algebraic structure — weight decay selects the simplest solution reachable from a given initialization.

---

## 2. Geometric Changes During Grokking

Using 449 ultra-dense checkpoints from run 1z2q8rx3 (every 20 steps), we tracked structural evolution:

### 2.1 Summary Table

| Metric | Step 100 | Step 3000 | Step 3900 | Step 5000 | Step 20000 |
|--------|----------|-----------|-----------|-----------|------------|
| Test Accuracy | 1.2% | 42.2% | 97.1% | 100% | 100% |
| Fourier Gini | 0.058 | 0.203 | 0.340 | 0.500 | 0.680 |
| Embedding Rank | 29.1 | 26.5 | 21.2 | 17.5 | 12.7 |
| #Unique Neuron Freqs | 43 | 3 | 3 | 2 | 3 |
| QK Circuit Rank | 4.55 | 1.00 | 1.00 | 1.00 | 1.00 |
| Total Weight Norm | 43.5 | 75.1 | 64.1 | 60.3 | 76.8 |
| Strongest FFT peak | k=38 | k=38 | k=12 | k=12 | k=12 |

**Note:** Both independent frequencies (k=12 and k=38) are present throughout training. What changes is which frequency has the highest power. See Section 2.2.2 for the corrected "crossover" interpretation.

### 2.2 Five Key Findings

**1. QK Circuit Simplifies FIRST (by step 500)**
- QK effective rank drops from 4.55 → 1.00 (essentially rank-1)
- This happens LONG BEFORE test accuracy improves
- The attention mechanism learns to uniformly attend to both operands
- This is a **prerequisite** for grokking, not the cause

**2. ~~Frequency Switching~~ → Frequency Crossover During Transition (step ~3440)**
- **Corrected (Feb 2026):** Ultra-dense analysis shows both k=12 and k=38 are present throughout the transition. What changes is their relative power ranking — k=38 initially dominates, then k=12 overtakes it around step 3440.
- This is a **crossover/rebalancing**, not a discrete switch.
- Both frequency pairs (k=12/101 and k=38/75) are conjugate mirrors (sum to p=113).
- Consistent with Ding et al. (2024) "Survival of the Fittest Representation" which models frequency competition as Lotka-Volterra dynamics — frequencies compete for representational capacity, with the "fittest" eventually dominating.

**3. Neuron Collapse is Dramatic**
- Starts: 43 unique frequencies across 128 neurons
- During transition: collapses to 3-4 frequencies
- After grokking: stable at 2-3 frequencies
- Almost ALL neurons specialize to the same frequencies

**4. Low-Rank Compression Continues After Grokking**
- Embedding rank: 29 → 21 (during grokking) → 13 (post-grokking)
- The network keeps compressing even at 100% accuracy
- Gini keeps increasing: 0.34 → 0.50 → 0.68
- This is the "cleanup" phase - optimizing the solution

**5. Weight Norm U-Curve**
- Initially increases (43 → 81) during memorization
- Drops during grokking (81 → 60)
- Slowly increases again post-grokking (60 → 77)
- Weight decay drives compression, then the model "relaxes"

---

## 3. Three-Phase Dynamics

### Phase 1: Memorization (steps 0-2000)
- Test accuracy: 1% → 23%
- QK circuit simplifies (rank 4.5 → 1.0) - **happens early!**
- Fourier spectrum stays diffuse (Gini ~0.1)
- Neurons spread across 40+ frequencies
- Model is memorizing, not generalizing

### Phase 2: Circuit Formation (steps 2000-4000)
- Test accuracy: 23% → 100% (the actual grokking)
- **Frequency crossover at step ~3440:** k=12 overtakes k=38 as dominant frequency (both present throughout)
- Fourier Gini increases 3x (0.14 → 0.34)
- Neurons collapse to 3-4 frequencies
- Embedding rank drops from 28 → 21
- Weight norm drops from 75 → 60
- The algorithmic circuit overtakes memorization

### Phase 3: Cleanup (steps 4000+)
- Test accuracy: stable at 100%
- Fourier Gini keeps increasing (0.34 → 0.68)
- Embedding rank keeps dropping (21 → 13)
- Network continues optimizing its representation
- Weight norm slowly increases again (60 → 77)
- The circuit is "polished" but not fundamentally changed

---

## 4. Detailed Transition Analysis

Fine-grained view of the critical transition period (steps 3000-4200):

| Step | TestAcc | Gini | Strongest Peak | #Freqs | EmbRank | Norm | Notes |
|------|---------|------|----------------|--------|---------|------|-------|
| 3000 | 42.2% | 0.203 | k=38 > k=12 | 3 | 26.5 | 75.1 | |
| 3200 | 47.5% | 0.220 | k=38 > k=12 | 3 | 25.8 | 71.9 | |
| 3400 | 57.2% | 0.241 | k=38 > k=12 | 4 | 25.0 | 69.6 | |
| 3440 | 59.6% | 0.245 | k=12 > k=38 | 4 | 24.8 | 69.2 | **CROSSOVER** |
| 3600 | 73.3% | 0.268 | k=12 > k=38 | 2 | 23.8 | 67.6 | |
| 3800 | 92.3% | 0.314 | k=12 > k=38 | 3 | 22.1 | 65.3 | |
| 3900 | 97.1% | 0.340 | k=12 >> k=38 | 3 | 21.2 | 64.1 | **GROKKED** |
| 4000 | 98.7% | 0.366 | k=12 >> k=38 | 3 | 20.3 | 63.1 | |
| 4200 | 99.2% | 0.413 | k=12 >> k=38 | 2 | 19.0 | 61.9 | |

**Note:** The "TopFreq" column in the original version of this table listed conjugate mirrors (e.g., "38, 75" or "101, 12") as if they were separate frequencies. Due to FFT symmetry on real data, these are the same independent frequency. The table above shows the actual independent frequency ordering.

---

## 5. Why Different Seeds Find Different Frequencies

### 5.1 The Fourier Algorithm
For modular addition (a + b) mod p, the MLP can compute the output using the trig identity:
```
cos(2πk(a+b)/p) = cos(2πka/p)cos(2πkb/p) - sin(2πka/p)sin(2πkb/p)
```
This works for ANY frequency k coprime to p. The model discovers whichever basis is easiest to reach from its random initialization.

**Our observations:**
- Seed 42: 2 independent frequencies (k=12, k=38)
- Seed 43: 1 independent frequency (k=48)
- Nanda et al.: 5 frequencies (k=14, 35, 41, 42, 52)

### 5.2 Literature Context

**Why different numbers of frequencies?** Recent theory provides several explanations:

- **McCracken et al. (NeurIPS 2025)** show a "universal algorithm" (approximate CRT) that unifies different solution types. For deep networks, O(log n) features suffice. Our 1-layer model operates in a different regime where the Fourier/trig mechanism dominates.
- **Li et al. (AIStats 2025)** prove Fourier circuits are the max-margin solution. The width threshold for full margin is m ≥ 2^(2k-2) × (p-1) neurons — for k=2, p=113, that's 448 neurons. Our model has only 128, so it hasn't reached the max-margin solution.
- **Ding et al. (2024)** model frequency selection as Lotka-Volterra competition — frequencies with the highest initial amplitude survive while others die out. The number of survivors depends on embedding dimension.
- **Tian (submitted ICLR 2026)** proposes the Li₂ framework with three stages (lazy, independent, interactive feature learning). An energy function predicts which frequencies emerge as local maxima.

### 5.3 Why Do Some Bases Achieve Higher Accuracy?
| Model | Seed | Independent Freqs | Final Acc | Notes |
|-------|------|-------------------|-----------|-------|
| 1z2q8rx3 | 42 | k=12, 38 (2 freqs) | 100% | More frequencies → better coverage |
| e332cujg | 43 | k=48 (1 freq) | 98.4% | Fewer frequencies → incomplete |
| l3wye1yc | 43 | k=48 (1 freq) | 94.5% | Same basis, weaker convergence |

**Updated interpretation:** The seed 42 model achieves 100% likely because it has **2 independent frequencies** providing better coverage of the output space. The seed 43 models have only **1 independent frequency** (k=48 with mirror k=65), which is insufficient for perfect accuracy. This is consistent with Li et al.'s finding that more frequencies → closer to max-margin optimum.

---

## 6. Attention & MLP Structure Evolution

### 6.1 QK Circuit
| Step | QK_Rank | QK_Top% | Interpretation |
|------|---------|---------|----------------|
| 100 | 4.55 | 59.9% | Random, diffuse |
| 500 | 1.09 | 98.9% | **Nearly rank-1** |
| 1000+ | 1.00 | 100% | Fully rank-1 |

**Key insight:** QK circuit goes rank-1 (QK_Top% → 100%) BEFORE grokking. This matches Nanda's observation that attention simplifies first.

### 6.2 MLP Structure
| Step | MLP_In Rank | MLP_Out Rank | Active Neurons | FreqSpec |
|------|-------------|--------------|----------------|----------|
| 100 | 29.9 | 29.8 | 128 | 0.021 |
| 3900 | 23.1 | 22.7 | 128 | 0.096 |
| 8000 | 18.6 | 13.8 | 128 | 0.143 |

The MLP becomes increasingly low-rank during and after grokking, compressing the computation.

---

## 7. Comparison with Nanda et al. (ICLR 2023) and Recent Literature

| Aspect | Our Analysis | Nanda et al. (2023) | Recent Theory |
|--------|--------------|---------------------|---------------|
| **#Independent Freqs** | 1-2 (seed dependent) | 5 | O(log p) ≈ 7 (McCracken 2025), up to 56 at max-margin (Li 2024) |
| **Gini coefficient** | 0.5-0.68 | ~0.4-0.5 | — |
| **QK rank-1** | Confirmed | Confirmed | Rank minimization (Yunis 2024) |
| **Three phases** | Confirmed | Confirmed | Li₂ framework (Tian 2025) |
| **Neuron specialization** | More extreme (2-3 freqs) | Yes | — |
| **Freq crossover** | Observed (k=38→k=12 dominance shift) | Not reported | Lotka-Volterra dynamics (Ding 2024) |
| **Post-grok cleanup** | Observed (rank drops after 100%) | Not reported | Glass relaxation (Zhang, NeurIPS 2025 Spotlight) |

---

## 8. Device Effect (CPU vs MPS)

We tested whether device (CPU vs MPS) affects final accuracy:

| Run | Seed | Device | Final Acc |
|-----|------|--------|-----------|
| e332cujg | 43 | MPS | 98.4% |
| l3wye1yc | 43 | CPU | 94.5% |

**Conclusion:** CPU actually performed WORSE than MPS for seed=43. Device does not explain why some models reach 100% - it's the **seed** (and resulting frequency basis) that matters.

---

## 9. Mechanistic Interpretation

The network learns modular addition through a Fourier algorithm:

```
(a + b) mod p  →  Embed in Fourier basis  →  MLP computes  →  Decode
```

Specifically:
1. **Embeddings** encode numbers as cos(2πka/p), sin(2πka/p) for frequency k
2. **Attention** (rank-1 QK) uniformly aggregates both operands
3. **MLP** computes: cos(2πk(a+b)/p) using the trig identity
4. **Output** uses conjugate frequency pair where k₁ + k₂ = p

---

## 10. Conclusions

### 10.1 Confirmed Observations
1. **Different seeds → different frequency bases** — seed 42 finds k=12,38 (2 independent freqs, 100% acc), seed 43 finds k=48 (1 independent freq, 94-98% acc). More frequencies correlates with higher accuracy.
2. **QK circuit simplifies FIRST** (step 500), long before accuracy improves — consistent with Nanda et al. and Yunis et al.'s rank minimization framework.
3. **Post-grokking cleanup continues** — embedding rank drops 21 → 13 even after 100% accuracy. Consistent with Zhang et al.'s glass relaxation interpretation.
4. **Weight norm follows a U-curve** — increases during memorization, drops during grokking, increases again during cleanup.

### 10.2 Corrected Claims (Feb 2026)
1. **~~Frequency switching~~ → Frequency crossover** — originally reported as a discrete switch from k=38,75 to k=12,101 at step 3440. Ultra-dense analysis shows all frequencies are present throughout; what changes is relative power. This is a competitive crossover consistent with Ding et al.'s Lotka-Volterra dynamics.
2. **~~2 frequencies vs Nanda's 5~~ → Counting was confused by conjugate symmetry** — We have 2 independent frequencies (k=12, k=38) for seed 42 and 1 for seed 43. Nanda et al.'s 5-frequency solution likely reflects a different training regime (different weight decay, longer training). McCracken et al. predict O(log p) ≈ 7 as the optimal number; Li et al. show (p-1)/2 = 56 at max-margin.
3. **Device effect was overstated** — CPU vs MPS differences exist but are secondary to seed choice and frequency count.

### 10.3 What's Genuinely Interesting in Our Data
1. **Ultra-dense checkpoint resolution** (every 20 steps) — most published analyses use much coarser checkpoints. Our 449-checkpoint trajectory provides unusually high-resolution dynamics.
2. **Frequency crossover dynamics** — direct empirical evidence of frequency competition during grokking, with continuous rebalancing rather than a discrete switch. This provides a concrete test case for Ding et al.'s Lotka-Volterra model.
3. **1-frequency vs 2-frequency accuracy gap** — seed 43's single-frequency solution plateaus at 94-98%, while seed 42's two-frequency solution reaches 100%. This is a clean demonstration of how frequency count affects accuracy.

---

## 11. Data Availability

All analysis scripts and checkpoints:
- `checkpoints/1z2q8rx3/` - 100% model (449 ultra-dense checkpoints)
- `checkpoints/e332cujg/` - 98.4% model (51 checkpoints)
- `checkpoints/5bon0t2j/` - 27% control (27 checkpoints)
- `configs/ultra_dense_grokking.toml` - Configuration for 100% model
- `configs/seed43_cpu_wd3.toml` - Configuration for CPU replication
- `scripts/nanda_analysis.py` - Fourier and excluded loss analysis
- `scripts/literature_validation.py` - Execution manifold and frequency analysis

---

## References

1. Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

2. Ding, X.D., Guo, Z.C., Michaud, E.J., Liu, Z., & Tegmark, M. (2024). Survival of the Fittest Representation: A Case Study with Modular Addition. [arXiv:2405.17420](https://arxiv.org/abs/2405.17420)

3. Li, C., Liang, Y., Shi, Z., Song, Z., & Zhou, T. (2024). Fourier Circuits in Neural Networks and Transformers: A Case Study of Modular Arithmetic with Multiple Inputs. *AIStats 2025*. [arXiv:2402.09469](https://arxiv.org/abs/2402.09469)

4. Yunis, D., Patel, K.K., Wheeler, S., Savarese, P., Vardi, G., Livescu, K., Maire, M., & Walter, M. (2024). Approaching Deep Learning through the Spectral Dynamics of Weights. *ICML 2024 MI Workshop / submitted to ICLR 2025*. [arXiv:2408.11804](https://arxiv.org/abs/2408.11804)

5. McCracken, G., Moisescu-Pareja, G., Letourneau, V., Precup, D., & Love, J. (2025). Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks. *NeurIPS 2025*. [arXiv:2505.18266](https://arxiv.org/abs/2505.18266)

6. Zhang, X., Shang, Y., Yang, E., & Zhang, G. (2025). Is Grokking a Computational Glass Relaxation? *NeurIPS 2025 Spotlight*. [arXiv:2505.11411](https://arxiv.org/abs/2505.11411)

7. Tian, Y. (2025). Li₂: A Framework on Dynamics of Feature Emergence and Delayed Generalization. *Submitted to ICLR 2026*. [arXiv:2509.21519](https://arxiv.org/abs/2509.21519)

8. Xu, Y. (2026). Low-Dimensional Execution Manifolds in Transformer Learning Dynamics. [arXiv:2602.10496](https://arxiv.org/abs/2602.10496)

9. Prakash, H.K. & Martin, C.H. (2026). Late-Stage Generalization Collapse in Grokking: Detecting Anti-Grokking with WeightWatcher. [arXiv:2602.02859](https://arxiv.org/abs/2602.02859)

10. Boursier, E., Pesme, S., & Dragomir, R.-A. (2025). A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation. *NeurIPS 2025*. [arXiv:2505.20172](https://arxiv.org/abs/2505.20172)

11. Sakamoto, K. & Sato, I. (2025). Explaining Grokking and Information Bottleneck through Neural Collapse Emergence. *ICLR 2026*. [arXiv:2509.20829](https://arxiv.org/abs/2509.20829)

12. Mallinar, N., et al. (2024). Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product. *ICML 2025 Oral*. [arXiv:2407.20199](https://arxiv.org/abs/2407.20199)

---

**Analysis conducted by:** Claude (Anthropic)
**Original date:** November 29, 2025
**Updated:** February 17, 2026
**Model:** claude-opus-4-6
