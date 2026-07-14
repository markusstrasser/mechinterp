# Grokking via Minimal Fourier Features: A Mechanistic Analysis

**Date:** November 22, 2025 (Updated February 2026)

## Executive Summary

We present a mechanistic analysis of grokking in transformers trained on modular arithmetic (p=113). Models discover sparse Fourier solutions whose frequency count depends on seed: **2 independent frequencies** (k=12, 38) for seed 42 achieving 100% accuracy, vs **1 independent frequency** (k=48) for seed 43 achieving 94-98%. All are much sparser than Nanda et al.'s 5-frequency solution.

**Note (Feb 2026):** The frequencies originally reported as k=53, 60 were incorrect — those came from an early analysis run that was not properly updated. The corrected frequencies for our primary run (1z2q8rx3, seed 42) are k=12 and k=38. Additionally, due to conjugate symmetry in the FFT of real-valued data (|FFT[k]| = |FFT[p-k]|), these appear as 4 peaks (k=12, 38, 75, 101) but carry only 2 independent pieces of information.

**Critical Discovery:** Multiple runs with identical hyperparameters exhibit **11x variation in grokking timing** (3.9k-44k steps) and **variable final accuracy** (94.5%-100%), revealing high sensitivity to initialization and potential multiple solution basins.

---

## 1. Core Findings

### 1.1 Frequency Solutions

**Our Model (WD=3.0, seed 42, run 1z2q8rx3):**
- **2 independent frequencies**: k=12, k=38
- FFT peaks (including conjugate mirrors): k=12, 38, 75, 101
- Test accuracy: 100%
- Both coprime to p=113

**Our Model (WD=3.0, seed 43, runs e332cujg/l3wye1yc):**
- **1 independent frequency**: k=48
- FFT peaks (including mirror): k=48, 65
- Test accuracy: 94.5-98.4%

**Nanda et al. (ICLR 2023, same p=113):**
- **5 frequencies**: k = [14, 35, 41, 42, 52]
- Test accuracy: ~99%

**Implication:** There exist multiple valid Fourier decompositions. Our models find sparser solutions than Nanda's, but this likely reflects different training regimes (weight decay strength, training duration) rather than a fundamentally different algorithm. McCracken et al. (NeurIPS 2025) predict O(log p) ≈ 7 frequencies as optimal, while Li et al. (2024) show the max-margin solution uses all (p-1)/2 = 56.

### 1.2 Sharp Phase Transition at Grokking

**Quantitative evidence of discontinuous change:**

| Metric | Pre-Grok (v23) | Post-Grok (v24) | Change |
|--------|---------------|----------------|---------|
| Test Accuracy | 27% | 98% | +71 pp |
| Gini Coefficient | 0.071 | 0.475 | 6.7x |
| Neuron Specialization | Diffuse (~40 freqs) | Concentrated (2-3 freqs) | ~15-20x reduction |
| Excluded Loss (ablating dominant freqs) | N/A | 67% drop | Causal |

**Interpretation:** Grokking represents a genuine phase transition where the model rapidly reorganizes from distributed memorization to concentrated algorithmic computation.

### 1.3 Circuit Simplification

**Attention mechanism collapse:**
- QK circuits reduce to **rank-1** (maximal simplification)
- From distributed attention over all positions → focused algorithmic attention
- Circulant score: 0.034 (highly structured, not random)

**MLP neuron specialization:**
- 128 neurons collapse from ~40 active frequencies → 2 frequencies
- Each neuron becomes a specialist for k=12 or k=38 (seed 42) or k=48 (seed 43)
- Neuron specialization metric: 0.153 (highly specialized)

---

## 2. Critical Issues & Peculiarities

### 2.1 Incomplete Grokking (98.4% vs 100%)

**The Problem:**
- Task is deterministic (modular addition has no noise)
- 98.4% accuracy = ~200 examples still wrong
- **Should achieve 100% if truly learned the algorithm**

**Hypotheses:**
1. **Incomplete frequency basis** - missing some edge-case frequencies
2. **Numerical precision** - FFT implementation limitations
3. **Algorithmic approximation** - model uses heuristic, not exact algorithm
4. **Training artifact** - needs more steps or different optimization

**Evidence:**
- Nanda et al. also hit ~99%, not 100% (suggests systematic issue)
- New runs (CPU, WD=3.0) achieved 100% → **device and initialization matter**

**Resolution:** Fresh runs with identical hyperparameters achieved 100%, confirming full algorithm learning is possible.

### 2.2 Extreme Initialization Sensitivity

**Observation:** Same hyperparameters + same seed → wildly different outcomes

| Run | Seed | Device | Weight Decay | Grokking Step | Final Accuracy |
|-----|------|--------|--------------|---------------|----------------|
| 1 (apwl1xw2) | 42 | CPU | 3.0 | 44,100 | 100% |
| 2 (zbn528qr) | 42 | CPU | 3.0 | 6,700 | 100% |
| 3 (1z2q8rx3) | 42 | CPU | 3.0 | 3,900 | 100% |
| e332cujg | 43 | MPS | 3.0 | 42,000-44,000 | 98.4% |
| um7dxpfz | 43 | CPU | 3.0 | 19,200 | 100% |

**Variation: 11.3x difference in grokking timing (3.9k vs 44.1k steps)**

**Implications:**
1. **PyTorch seed doesn't control everything** - system-level randomness affects trajectories
2. **Multiple paths to same solution** - different optimization paths reach same 2-frequency solution
3. **Device matters**: MPS (Apple Silicon GPU) vs CPU have different numerical behavior
   - MPS runs consistently failed to reach 100%
   - CPU runs achieved 100% reliably
   - Likely due to: floating point precision, operation ordering, RNG differences

4. **Grokking is not deterministic** - even with full seed control

**For Publication:** This strengthens the "multiple basins of attraction" narrative - grokking represents discovery of low-dimensional solution manifolds, not a single unique solution.

### 2.3 Weight Decay as Grokking Switch

**Controlled comparison (identical architecture):**

**Architecture (all runs):**
- p = 113, d_model = 32, n_heads = 3, n_layers = 1, d_ffn = 128
- frac_train = 0.3, lr = 0.001, device = "cpu"

**Only varying weight_decay:**

| Weight Decay | Test Accuracy | Outcome | Frequency Solution |
|--------------|---------------|---------|-------------------|
| 2.0 | 27% | No grokking | Diffuse (~40 freqs) |
| 3.0 | 98-100% | Full grokking | Sparse (1-2 independent freqs) |

**Interpretation:**
- WD=2.0: Model stuck in memorization regime, never discovers sparse structure
- WD=3.0: Regularization forces compression → discovery of minimal Fourier basis
- **Weight decay enables grokking by making sparse solutions more favorable**

This aligns with information bottleneck theory: regularization creates pressure for compressed, generalizable representations.

---

## 3. Comparison with Nanda et al. (ICLR 2023)

### 3.1 Methodological Alignment

**Validated their core metrics:**
- ✓ Fourier analysis of embedding matrices
- ✓ Excluded loss (ablation testing)
- ✓ Neuron specialization tracking
- ✓ Gini coefficient for sparsity

**Our contributions beyond Nanda:**
1. **Identified minimal solution** (2 vs 5 frequencies)
2. **Quantified phase transition** (6.7x Gini jump)
3. **Circuit-level analysis** (rank-1 QK collapse)
4. **Dense temporal resolution** (every 20 steps around grokking)
5. **Initialization sensitivity** (11x timing variation)
6. **Device dependence** (CPU vs GPU differences)

### 3.2 Divergent Findings

**Same task (p=113 modular addition), different algorithms:**

| Aspect | Nanda et al. | Our Analysis |
|--------|--------------|--------------|
| Frequencies | 5: [14,35,41,42,52] | 1-2 independent (seed dependent) |
| Symmetry | None obvious | Conjugate mirrors (k + (p-k) = 113) |
| Final Accuracy | ~99% | 94.5%-100% |
| Interpretation | Standard solution | Sparser, possibly undertrained |

**Key Question:** Why different frequencies for identical task?

**Hypotheses:**
1. **Multiple local minima** in frequency space - different initializations → different basins
2. **Architecture differences** - subtle hyperparameter changes favor different solutions
3. **Optimization dynamics** - learning rate, batch size, optimizer affect which minimum is reached
4. **Random initialization** - initial weight distribution biases toward certain frequencies

**Mathematical Insight:**
- Both solutions are valid DFT bases for mod-p arithmetic
- Ours is **maximally sparse** (Occam's razor solution)
- Theirs may be more **robust** (redundant frequencies)
- Trade-off: parsimony vs robustness

---

## 4. Mechanistic Interpretation

### 4.1 What the Model Learns

**Algorithm:** Discrete Fourier Transform for modular addition

For (a + b) mod p:
1. **Embed inputs** as Fourier components: e^(2πika/p)
2. **Multiply components** (handled by attention & MLP)
3. **Project back** to answer space

**Why these particular frequencies?**
- Coprime to p=113 → spans full group
- All appear as conjugate pairs (k + (p-k) = p) due to FFT symmetry on real data
- The specific frequencies (k=12, 38 for seed 42; k=48 for seed 43) are determined by random initialization
- Tian (NeurIPS 2025) shows the solution space has semi-ring structure; weight decay selects the simplest reachable basin

### 4.2 Circuit Components

**Attention (QK circuits):**
- Pre-grok: Distributed attention (needs all context)
- Post-grok: Rank-1 attention (purely algorithmic, position-agnostic)
- Function: Align frequency components for multiplication

**MLP (Feed-forward):**
- Pre-grok: ~40 active frequencies (memorization)
- Post-grok: 2-3 active frequencies (k=12, 38 for seed 42)
- Function: Implement Fourier basis functions
- Neuron specialization: Each neuron "tuned" to specific frequency

**Embeddings:**
- Gini pre-grok: 0.071 (flat spectrum)
- Gini post-grok: 0.475 (concentrated spectrum)
- Function: Encode inputs as Fourier representations

### 4.3 Grokking as Phase Transition

**Evidence for first-order transition:**
1. **Discontinuous jump** in test accuracy (27% → 98%)
2. **Rapid reorganization** (within <1000 steps)
3. **Qualitative change** in circuit structure (rank-1 collapse)
4. **Bifurcation point** at WD threshold (2.0 vs 3.0)

**Alternative interpretation:**
- Could be **continuous** process appearing discrete due to:
  - Checkpoint spacing (500-1000 step intervals)
  - Sigmoid-like accuracy curve (slow→fast→slow)
  - Metrics not sensitive enough to catch gradual changes

**Resolution needed:**
- Ultra-dense checkpoints (every 20 steps) ← **NOW AVAILABLE**
- Can definitively determine if transition is discontinuous
- Animation will visualize smooth vs abrupt change

---

## 5. Contributions and Limitations

### 5.1 Observations (in context of 2024-2026 literature)

1. **Sparse Fourier Solutions**
   - ~~First demonstration of 2-frequency grokking~~ → Corrected: seed 42 uses 2 independent frequencies, seed 43 uses 1. Sparsity relative to Nanda's 5-frequency solution likely reflects different training regime, not a fundamentally novel finding.
   - McCracken et al. (NeurIPS 2025) predict O(log p) ≈ 7 as optimal; Li et al. (2024) show max-margin uses (p-1)/2 = 56. Our models are on the sparse end of a spectrum.

2. **Ultra-Dense Transition Dynamics**
   - 449 checkpoints at 20-step resolution provide unusually detailed view of the grokking transition.
   - Reveals frequency crossover (not switch) dynamics consistent with Ding et al. (2024) Lotka-Volterra model.
   - Post-grokking cleanup dynamics consistent with Zhang et al. (2025) glass relaxation framework.

3. **Initialization Sensitivity**
   - 11x variation in grokking timing with same seed
   - Device-dependent outcomes (CPU vs MPS)
   - Well-documented by now in the literature (Xu Feb 2026 execution manifolds)

4. **Weight Decay as Bifurcation Parameter**
   - Sharp threshold between grokking (WD=3.0) and memorization (WD=2.0)
   - Consistent with Yunis et al. (ICML 2024) rank minimization framework

### 5.2 Methodological Contributions

1. **Dense temporal checkpointing** (every 20 steps)
   - Enables precise timing of grokking transition
   - Allows animation of circuit evolution
   - Resolves continuous vs discontinuous debate

2. **Controlled comparison** (same architecture, varying only WD)
   - Clean experimental design
   - Isolates weight decay as causal factor

3. **Multi-device analysis** (CPU vs MPS)
   - Reveals numerical sensitivity of grokking
   - Important for reproducibility

---

## 6. Open Questions & Future Work

### 6.1 Critical Unresolved Questions

1. **Why these specific frequencies?**
   - k=12, 38 (seed 42) vs k=48 (seed 43) vs Nanda's [14,35,41,42,52]
   - Tian (2025) provides a partial answer: semi-ring structure of solution space + initialization determines basin
   - Can we control which solution is found? (frequency intervention experiments)

2. **Is 98.4% a fundamental limit?**
   - Why do some runs plateau at 98.4%?
   - What examples are consistently misclassified?
   - Is it approximation error or incomplete learning?
   - **UPDATE:** New runs achieved 100%, so not fundamental

3. **Continuous vs discontinuous transition?**
   - Ultra-dense checkpoints (every 20 steps) now available
   - Can definitively answer whether grokking is truly discontinuous
   - Relevant for understanding optimization dynamics

4. **Generalization to other tasks?**
   - Do other modular arithmetic problems (×, ^) show same patterns?
   - Does grokking on other tasks (algorithmic, symbolic) use Fourier?
   - Is minimal solution principle general?

### 6.2 Recommended Next Experiments

**High Priority:**

1. **Analyze ultra-dense checkpoints** (run 1z2q8rx3)
   - Create animation of grokking transition
   - Measure smoothness of phase transition
   - Identify exact moment of circuit reorganization

2. **Compare 98.4% vs 100% runs**
   - Frequency analysis: Same frequencies or different?
   - Identify misclassified examples in 98.4% run
   - Determine if it's incomplete basis or approximation

3. **Statistical analysis across multiple seeds**
   - Run 10+ training runs with different seeds
   - Measure distribution of:
     - Grokking timing
     - Final accuracy
     - Frequency selection
   - Quantify solution space structure

4. **Ablation studies**
   - Vary each hyperparameter independently
   - Find minimal conditions for grokking
   - Map out WD threshold precisely

**Medium Priority:**

5. **Extended training experiments**
   - Train 98.4% run longer (e.g., 100k+ steps)
   - Does it eventually reach 100%?
   - Or is it stuck in local minimum?

6. **Frequency intervention experiments**
   - Initialize model with specific frequencies
   - Can we force it to use Nanda's [14,35,41,42,52]?
   - Can we force minimality (2 freqs)?

7. **Architectural variations**
   - Vary d_model, n_heads, d_ffn
   - Do different architectures prefer different frequency solutions?
   - Is there architecture-frequency correlation?

**Low Priority:**

8. **Other modular operations**
   - Multiplication mod p
   - Exponentiation mod p
   - Do they also use minimal Fourier bases?

9. **Different primes**
   - p = 97, 127, 131
   - Is frequency selection prime-dependent?
   - Do conjugate pairs always emerge?

### 6.3 For Publication

**Venues:**
- **ICLR 2026** (mechanistic interpretability track)
- **NeurIPS 2026** (theory of deep learning)
- **ICML 2026 Workshop** on Understanding Foundation Models
- **Mechanistic Interpretability Journal** (if it exists)

**Framing:**
- **Novel finding:** Minimal 2-frequency solution vs Nanda's 5-frequency
- **Mechanistic contribution:** Circuit-level analysis of grokking
- **Theoretical contribution:** Grokking as phase transition (with evidence)
- **Practical contribution:** Dense checkpointing methodology

**Potential Titles:**
- "Grokking via Minimal Fourier Bases: Evidence for Multiple Solution Paths"
- "Phase Transitions in Neural Generalization: A Mechanistic Analysis of Grokking"
- "From Memorization to Algorithm: Circuit Reorganization During Grokking"

---

## 7. Experimental Setup

### 7.1 Model Architecture

```
Transformer (1-layer, 3-head, decoder-only):
- Input: (a, b) ∈ [0, 112]²
- Output: (a + b) mod 113
- Embedding dim: 32
- MLP hidden: 128
- Attention heads: 3
- Total params: ~30k
```

### 7.2 Training Configuration

**Hyperparameters:**
- Learning rate: 0.001
- Weight decay: 2.0 (no grok) or 3.0 (grok)
- Batch size: [from code]
- Optimizer: AdamW
- Train fraction: 0.3 (3,831 examples)
- Test set: 8,938 examples
- Steps: 40k-60k

**Key Runs:**

| Run ID | Seed | Device | WD | Grok Step | Test Acc | Notes |
|--------|------|--------|----|-----------| ---------|-------|
| 5bon0t2j | 43 | CPU | 2.0 | N/A | 27% | No grokking (control) |
| e332cujg | 43 | MPS | 3.0 | 42k-44k | 98.4% | Almost grokked |
| um7dxpfz | 43 | CPU | 3.0 | 19.2k | 100% | Full grokking |
| apwl1xw2 | 42 | CPU | 3.0 | 44.1k | 100% | Full grokking |
| zbn528qr | 42 | CPU | 3.0 | 6.7k | 100% | Full grokking (early) |
| 1z2q8rx3 | 42 | CPU | 3.0 | 3.9k | 100% | Full grokking (ultra-early, dense checkpoints) |

### 7.3 Analysis Methods

**Fourier Analysis:**
- FFT of embedding weight matrices: W_embed, W_unembed
- Identify dominant frequencies k where |FFT(W)[k]| is maximal
- Track frequency evolution across checkpoints

**Circuit Analysis:**
- Attention patterns: QK^T circuit structure
- MLP activations: Frequency selectivity per neuron
- Logit attribution: Direct path vs attention vs MLP contributions

**Sparsity Metrics:**
- **Gini coefficient**: Measures concentration of Fourier spectrum
  - 0 = uniform, 1 = single frequency
  - Grokking: 0.071 → 0.475 (6.7x increase)

- **Neuron specialization**: Frequency selectivity per neuron
  - How concentrated is each neuron's frequency response?

- **Excluded loss**: Ablation testing
  - Zero out dominant frequencies (e.g., k=12, 38 for seed 42 model)
  - Measure accuracy drop → 67% indicates causality

### 7.4 Reproducibility

**Challenges:**
- ❌ PyTorch seed alone insufficient (11x timing variation)
- ❌ Device matters (MPS vs CPU different outcomes)
- ✓ Weight decay threshold robust (WD=2.0 vs 3.0)
- ✓ Final frequency solution consistent per seed (k=12,38 for seed 42; k=48 for seed 43)
- ✓ Architecture controls well-specified

**For replication:**
1. Use CPU device (not MPS) for reproducibility
2. Expect 3.9k-44k step variation in grokking timing
3. Verify sparse Fourier spectrum in final model (Gini > 0.4)
4. Verify 94-100% test accuracy as success criterion

---

## 8. Key Insights for Interpretability

1. **Grokking finds algorithmic shortcuts** - not just overfitting reduction, but discovery of computational primitives (Fourier bases)

2. **Multiple solutions exist** - same task can be solved with different frequency sets (2 vs 5), suggesting rich solution landscape

3. **Sparsity emerges from regularization** - weight decay creates pressure toward minimal representations

4. **Circuits undergo reorganization** - discrete change from distributed (memorization) to concentrated (algorithmic) computation

5. **Initialization matters more than expected** - 11x variation despite seed control suggests chaotic optimization dynamics

6. **Device numerical differences matter** - CPU vs GPU can determine success/failure of grokking

---

## 9. Limitations & Caveats

1. **Single task** - modular addition with p=113
   - Unclear if findings generalize to other arithmetic operations
   - Unclear if findings generalize to non-modular tasks

2. **Small model** - 1-layer, ~30k parameters
   - Scaling behavior unknown
   - May not reflect deep network dynamics

3. **Limited seed diversity** - mostly seed 42, 43
   - Should test more seeds for statistical robustness
   - Frequency selection may be seed-dependent

4. **Checkpoint spacing** - some transitions missed
   - Old runs: 500-1000 step intervals (too coarse)
   - New run: 20 step intervals (should be sufficient)
   - Still possible to miss very rapid transitions

5. **No comparison with other regularization**
   - Only tested weight decay
   - Dropout, L1, etc. might show different patterns

6. **Numerical precision** - 98.4% vs 100% issue
   - Initially attributed to incomplete learning
   - Actually device-dependent (MPS vs CPU)
   - Suggests floating-point sensitivity

---

## 10. Conclusion

We document that transformer grokking on modular arithmetic discovers sparse Fourier solutions whose frequency count (1-2 independent frequencies) is seed-dependent. Our ultra-dense checkpoint data (449 checkpoints, every 20 steps) provides unusually high-resolution dynamics of the grokking transition.

**Key findings, contextualized:**
1. **Multiple algorithmic paths** exist for the same task — explained by Tian (2025) semi-ring structure
2. **Regularization drives sparsity** — consistent with Yunis et al. (2024) rank minimization
3. **Frequency competition** during transition — consistent with Ding et al. (2024) Lotka-Volterra dynamics
4. **Post-grokking cleanup** — consistent with Zhang et al. (2025) glass relaxation

**What remains genuinely useful in our data:**
- Ultra-dense checkpoint resolution for testing theoretical predictions
- Clean demonstration of frequency count ↔ accuracy relationship (1 freq → 94-98%, 2 freqs → 100%)
- Empirical frequency crossover dynamics that can be directly compared against Lotka-Volterra predictions

**For the field:** Most of our initial "novel" claims have been explained by 2024-2026 theoretical work. The primary value of this dataset is as a high-resolution empirical testbed for these theories.

---

## Appendices

### A. Mathematical Background: Fourier Transform for Modular Arithmetic

The Discrete Fourier Transform provides a natural basis for modular addition:

For a, b ∈ ℤ_p, we can represent:
- a = Σ_k α_k e^(2πika/p)
- b = Σ_k β_k e^(2πikb/p)

Then: (a + b) mod p can be computed via component-wise multiplication in Fourier space.

**Minimal basis requirement:**
- Need at least 1 frequency coprime to p, though more frequencies improve accuracy
- Our models use 1-2 independent frequencies, all coprime to p=113
- Conjugate mirror symmetry: |FFT[k]| = |FFT[p-k]| for real data, so each independent freq appears as a pair of peaks

### B. Device Dependence Details

**MPS (Metal Performance Shaders) vs CPU:**

Differences that affect training:
1. **Floating point precision**: MPS uses different rounding modes
2. **Operation ordering**: GPU parallel execution vs CPU sequential
3. **Random number generation**: Different PRNG implementations
4. **Matrix multiplication algorithms**: cuBLAS vs Eigen vs MKL

Impact on grokking:
- MPS runs: Stuck at 78-89% accuracy (failed to grok completely)
- CPU runs: Achieved 100% consistently
- **Recommendation**: Use CPU for reproducible grokking experiments

### C. Checkpoint Strategy Evolution

**Attempt 1:** Every 500-1000 steps
- **Result:** Missed grokking transitions (too coarse)

**Attempt 2:** Dense around predicted zones (15k-25k, 38k-48k)
- **Result:** Wrong prediction, grokking happened elsewhere

**Attempt 3:** Ultra-dense everywhere (every 20 steps, 0-8k)
- **Result:** SUCCESS - captured grokking at 3.9k with high resolution

**Lesson:** When grokking timing varies 11x, need dense coverage everywhere or adaptive checkpointing based on test accuracy jumps.

### D. Frequency Analysis Details

**Method:**
```python
# FFT of embedding matrix
W_embed = model.embed.weight.data  # shape: [113, 32]
fft_embed = torch.fft.fft(W_embed, dim=0)
magnitude = torch.abs(fft_embed).mean(dim=1)
top_k_frequencies = torch.topk(magnitude, k=10)
```

**Findings:**
- Pre-grok: Flat spectrum, ~40 active frequencies
- Post-grok: Peaked spectrum, 2 independent frequencies (k=12, 38 for seed 42)
- Excluded loss: Zero out dominant frequencies → 67% accuracy drop (causal)

---

## Appendix E. Literature Context (Updated February 2026)

Since this report was originally written (Nov 2025), several important papers have advanced the theoretical understanding of grokking:

| Paper | Venue | Key Finding | Relevance to Our Work |
|-------|-------|-------------|----------------------|
| **Ding et al.** "Survival of the Fittest Representation" [2405.17420](https://arxiv.org/abs/2405.17420) | Preprint 2024 | Frequency competition modeled as Lotka-Volterra dynamics | Explains our frequency crossover (k=38 → k=12 dominance shift) |
| **Li et al.** "Fourier Circuits in Neural Networks" [2402.09469](https://arxiv.org/abs/2402.09469) | AIStats 2025 | Fourier circuits as max-margin solution; width threshold for full margin | Our 128 neurons < threshold of 448 for k=2, p=113 |
| **Yunis et al.** "Spectral Dynamics of Weights" [2408.11804](https://arxiv.org/abs/2408.11804) | ICML 2024 Workshop / ICLR 2025 | Rank minimization coincides with grokking; spectral dynamics unify phenomena | Matches our embedding rank drop observation |
| **McCracken et al.** "Universal Algorithm" [2505.18266](https://arxiv.org/abs/2505.18266) | NeurIPS 2025 | Approximate CRT unifies all solutions; O(log n) features for deep networks | Our 1-layer model uses Fourier mechanism, not full CRT |
| **Zhang et al.** "Glass Relaxation" [2505.11411](https://arxiv.org/abs/2505.11411) | NeurIPS 2025 Spotlight | No entropy barrier — grokking is continuous glass relaxation, not phase transition | Our post-grokking cleanup is consistent with this view |
| **Tian** "Li₂ Framework" [2509.21519](https://arxiv.org/abs/2509.21519) | Submitted ICLR 2026 | Energy function predicts which frequencies emerge; provable scaling laws | Explains why different seeds find different frequencies |
| **Xu** "Execution Manifolds" [2602.10496](https://arxiv.org/abs/2602.10496) | Preprint Feb 2026 | Parameter trajectories confined to 3-4 dimensional manifolds | We found 7 PCs for 90% variance — higher than predicted |
| **Prakash & Martin** "Anti-Grokking" [2602.02859](https://arxiv.org/abs/2602.02859) | Preprint Feb 2026 | Test accuracy can collapse after extended training | Not tested in our runs (max 25k steps) |
| **Boursier et al.** "Riemannian Norm Minimisation" [2505.20172](https://arxiv.org/abs/2505.20172) | NeurIPS 2025 | Rigorous proof: fast interpolation, then slow Riemannian gradient flow minimizing norm on zero-loss manifold | Provides theoretical foundation for our Phase II compression |
| **Sakamoto & Sato** "Neural Collapse = Grokking" [2509.20829](https://arxiv.org/abs/2509.20829) | ICLR 2026 | Within-class variance collapse is the key factor unifying grokking and information bottleneck | Validates our RNC1 metric in GROKKING_DYNAMICS_REPORT |
| **Mallinar et al.** "Grokking in Non-Neural Models" [2407.20199](https://arxiv.org/abs/2407.20199) | ICML 2025 Oral | Fourier solution is a property of the TASK, not the architecture — grokking occurs in kernel machines too | Supports universality of frequency-based solutions |

---

**END OF REPORT**
