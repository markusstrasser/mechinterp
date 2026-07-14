# Analysis of Grokking Dynamics: Compression and Phase Transitions

**Date:** November 22, 2025 (Updated February 2026)
**Subject:** Run `1z2q8rx3` (Modular Addition, p=113)
**Methodology:** High-resolution checkpoint analysis ($N=441$, $\Delta t=20$ steps)

---

## 1. Abstract

This report analyzes the dynamics of grokking in a transformer trained on modular arithmetic. By observing internal metrics at high temporal resolution, we identify a distinct "compression" phase preceding generalization. The data supports a mechanistic hypothesis where weight decay induces a norm constraint that destabilizes high-complexity memorization circuits, forcing a transition to a sparse, low-norm algorithmic solution (Fourier features).

## 2. Metric Definitions

To characterize the phase transition, we monitored the following metrics:

*   **Test Accuracy:** Fraction of correct predictions on the held-out test set (complement of training data).
*   **L2 Weight Norm:** The Euclidean norm of all trainable parameters $\theta$: $\|\theta\|_2 = \sqrt{\sum \theta_i^2}$. This serves as a proxy for model complexity.
*   **Within-Class Variance (RNC1):** A geometric measure of representation quality. For a given class $c$ (target output), it measures how tightly clustered the residual stream representations are:
    $$ \text{RNC1} \propto \sum_c \sum_{x \in X_c} \| f(x) - \mu_c \|^2 $$
    High variance implies unstructured/memorized representations; low variance implies collapsed/structured representations.
*   **Sparsity (Gini Coefficient):** Applied to the Fourier spectrum of the embedding matrix to measure the concentration of spectral energy. High Gini indicates a sparse solution using few frequencies.

## 3. Dynamics Analysis

The training trajectory exhibits three distinct phases characterized by the interplay between regularization (weight decay) and the loss landscape.

### Phase I: Unregularized Memorization ($t < 2000$)
*   **Behavior:** The model minimizes training loss via brute-force memorization.
*   **Signatures:**
    *   **Norm Expansion:** L2 Norm increases significantly ($17.0 \to 37.0$). The model utilizes available capacity without constraint.
    *   **High Variance:** RNC1 is maximal ($>400$). Representations for inputs mapping to the same output (e.g., $5+7$ and $1+11$) are orthogonal or uncorrelated.
    *   **Accuracy:** Remains near random baseline or improves slowly.

### Phase II: Regularization-Induced Compression ($2000 < t < 3500$)
*   **Behavior:** The weight decay term in the loss function ($\lambda \|\theta\|^2$) becomes dominant relative to the gradient of the cross-entropy loss on memorized examples.
*   **Signatures:**
    *   **Norm Contraction:** The L2 Norm decreases monotonically ($37.0 \to 32.0$).
    *   **Variance Collapse:** Within-Class Variance drops precipitously ($400 \to 129$). The regularization pressure forces the model to compress its representations, effectively removing the capacity required to maintain independent memorized entries.
    *   **Transition Point:** The minimum of the Within-Class Variance ($t \approx 3500$) coincides exactly with the inflection point of the Test Accuracy curve.

### Phase III: Algorithmic Crystallization ($t > 3500$)
*   **Behavior:** The model converges to a global minimum corresponding to the generalizable Fourier algorithm.
*   **Signatures:**
    *   **Accuracy Saturation:** Test accuracy rapidly converges to 100%.
    *   **Spectral Sparsity:** The Gini coefficient of the Fourier spectrum increases ($0.25 \to 0.60$), indicating the selection of specific frequencies. The model converges on 2 independent frequencies ($k=12$ and $k=38$), which appear as 4 peaks in the FFT due to conjugate symmetry ($|FFT[k]| = |FFT[p-k]|$ for real-valued data, giving mirrors at $k=101$ and $k=75$ respectively).
    *   **Structured Expansion:** Interestingly, RNC1 increases again post-grokking. This is consistent with a Fourier solution where representations lie on a structured manifold (e.g., a circle in high-dimensional space), which inherently has non-zero variance, distinct from the unstructured noise of Phase I.

## 4. Conclusion

Grokking in this setting is driven by the competition between the primary loss objective and the regularization constraint. The "delay" in generalization corresponds to the time required for weight decay to compress the high-norm memorization solution to a point of instability. The collapse of within-class variance serves as a precursor signal for the phase transition, marking the point where the algorithmic solution becomes the energetically favorable state.

These findings align with the "Norm Minimization" (Boursier et al.) and "Neural Collapse" (Papyan et al.) theoretical frameworks. The frequency competition dynamics (where $k=38$ initially dominates before $k=12$ overtakes it) are consistent with the Lotka-Volterra competition model proposed by Ding et al. (2024, "Survival of the Fittest Representation"), while the compression-driven transition aligns with Yunis et al. (ICML 2024) who show rank minimization coincides with grokking onset.
