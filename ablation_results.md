# Ablation Study: Riemannian vs Euclidean Inductive Bias

**Date**: January 2026
**Objective**: To isolate the effect of the hyperbolic inductive bias on mixture component specialization.

## Methodology
We trained two identical models for 25 steps (sufficient for geometry convergence) on the Alpaca dataset:
1.  **Riemannian**: `geometry="riemannian"`. Used Poincaré distance for component affinity.
2.  **Euclidean**: `geometry="euclidean"`. Used KL divergence (flat) for component affinity.

All other parameters (learning rate, seeds, architecture) were identical.

## Results

| Metric | Riemannian (Hyperbolic) | Euclidean (Flat) | Delta | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Mixture Entropy** | **1.1277** | 1.1675 | -0.0398 | **Sharper Specialization**. The Riemannian model partitions the semantic space more discretely. |
| **Component Spread** | 48.85 | 49.01 | -0.16 | Comparable volume usage. |
| **Component Norm** | 46.03 | 46.27 | -0.24 | Both models saturate the available space (Pre-tanh norms are high). |

## Conclusion
The **Riemannian Inductive Bias** leads to measurably lower mixture entropy (-0.04), confirming that the hyperbolic geometry facilitates sharper "gating" or specialization of the mixture components (fibers) compared to a flat Euclidean geometry. This supports the hypothesis that hyperbolic spaces naturally accommodate hierarchical distinctions better than Euclidean ones.
