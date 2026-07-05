# ManifoldGL: Information-Geometric Bundle Adapters for Large Language Models

## Abstract
(To be written after Falsification Experiment Results)

## 1. Introduction

The geometry of representations in Large Language Models (LLMs) fundamentally dictates their expressive capacity and reasoning dynamics. While standard fine-tuning approaches (such as LoRA) operate via linear projections in Euclidean parameter spaces, recent theoretical work suggests that complex hierarchical and semantic relationships are better represented on manifolds with constant negative curvature (hyperbolic space). 

In this work, we introduce the **Information-Geometric Bundle Adapter (IGBundle)**, a novel architecture that embeds a continuous Poincaré ball model ($H^n$) within the discrete transformer blocks of a frozen 7B-parameter language model. Our framework couples a Riemannian base manifold with categorical fiber sections ($\Delta^{K-1}$) routed via a mixture-of-experts (MoE) mechanism. We enforce geometric consistency via a suite of custom regularization losses—including Sheaf Consensus, Bundle Structure, and a GENERIC (General Equation for Non-Equilibrium Reversible-Irreversible Coupling) thermodynamic constraint.

Our initial hypothesis posited that enforcing rigorous, mathematically faithful hyperbolic geometry within the adapter parameters would directly translate into measurable downstream task performance and distinct behavioral regimes. However, exhaustive empirical investigation revealed a profound architectural decoupling, which we now state at two distinct levels. First, at the level of *geometric operations*: the fixed Poincaré kernels (Möbius addition, exponential/logarithmic maps, geodesic distance) are mathematically correct and pass 100% of faithfulness tests — but these tests verify the *fixed* kernels, not learned geometry (Tier: EMPIRICAL). Second, at the level of *geometric learning*: the single genuinely trainable geometric parameter (the metric Cholesky factor) converged to the identity — flat — and, as we prove in §2.6, the curvature telemetry could not have registered the difference even if it had not (Tier: FALSIFIED at the metric level, independently of the transfer-level failure). The discrete softmax routing mechanism additionally collapsed into a degenerate attractor, severing the continuous geometric manifold from the model's predictive token-generation mechanics (Tier: HYPOTHESIS, with supporting diagnostics).

Rather than obscuring this negative result, this paper presents a rigorous mechanistic diagnosis of *why* local geometric interventions fail to transfer to global task performance when bottlenecked by layer count and projection topologies. We establish an explicit falsification framework, comparing a geometrically rigorous single-layer adapter against standard multi-layer Euclidean LoRA baselines under strict compute-parity constraints.

## 2. Methodology

### 2.1 The IGBundle Architecture
The IGBundle architecture operates at a single intermediate layer (Layer 12) of a frozen Qwen2.5-7B base model. The adapter projects the Euclidean hidden states $h \in \mathbb{R}^d$ into a parameterized bottleneck $h_{bot} \in \mathbb{R}^{d_{bot}}$. From this bottleneck, the representation is split into:
1. **Base Coordinates ($q \in H^n$):** Projected onto the Poincaré ball via a norm-based scaling to ensure strictly bounded hyperbolic geometry ($\|q\| < 1$).
2. **Fiber Sections ($\theta \in \Delta^{K-1}$):** A categorical probability simplex over $K$ experts, intended to act as the continuous routing mechanism across the fiber bundle $E = H^n \times \Delta^{K-1}$.

### 2.2 Geometric Regularization and Collapse Diagnosis
We applied strict geometric regularization, attempting to steer the manifold via:
- **Curvature (K) Regularization:** Enforcing $K = -1$ (hyperbolic).
- **Entropy (S) Regularization:** Attempting to maintain high diversity across the fiber sections $\theta$.

However, our telemetry revealed a critical structural failure, whose mechanism was subsequently identified at the source level (§2.6): the reported curvature was not saturated by clamping — it was *weight-invariant by construction*, a constant of the hardcoded conformal factor evaluated at the coordinate-norm distribution. Independently, the fiber sections experienced severe "Bundle Lock": the routing mechanism collapsed, rendering the manifold entropy (S) a measurement of a degenerate distribution.

### 2.3 The Falsification Experiment Design
To empirically isolate the causal variable of task performance (Geometry vs. Layer Count), we designed a strict falsification protocol. We established the following null hypothesis:

> **$H_0$:** A standard, multi-layer Euclidean LoRA (Layers 8–20) matches or exceeds the performance of a geometrically rigorous single-layer (Layer 12) IGBundle adapter on standard downstream benchmarks.

To ensure computational parity and eliminate parameter-count confounding, we constructed two primary baseline configurations:
1. **Parameter-Matched Baseline:** A multi-layer (layers 8–20) LoRA tuned to Rank 2, mathematically equating the total number of trainable parameters to our Rank-8 single-layer IGBundle.
2. **Layer-Matched Baseline:** A multi-layer (layers 8–20) LoRA at Rank 8, providing the standard industry upper-bound for this architecture scale.

### 2.4 Evaluation Protocol
All checkpoints are evaluated using an identical, rigorous protocol to ensure statistical significance:
- **Datasets:** Established academic benchmarks including GSM8K, MMLU, and ARC-Easy.
- **Sampling:** Each prompt is evaluated 10 times at a consistent temperature of $T=0.6$.
- **Metrics:** Results are reported as the mean accuracy $\pm$ standard deviation. 

By demanding a $p < 0.05$ threshold to reject the null hypothesis, this framework cleanly separates the mathematical elegance of the geometry from its actual empirical utility in the context of frozen LLMs.

### 2.5 Telemetry Reconciliation: One Label, Four Instruments

Published artifacts of this project quote mutually inconsistent "curvature" values: $K = -5.63$ and $-5.72$ (training telemetry, model cards), $K = -0.98$ (an earlier evaluation), and `avg_curvature` $= +1.71$ (`geometric_validation.json`). These are **not** contradictory measurements of one quantity; they are outputs of four different instruments that shared a label:

| Reported value | Instrument | What it actually measures | Sign convention |
|---|---|---|---|
| $-5.63$, $-5.72$ | Adapter-internal stochastic estimator (`estimate_sectional_curvature_stochastic`) | $-\tfrac{D}{2}\,\partial_k^2 \log\lambda(x)$ for the *hardcoded* conformal factor $\lambda(x)=1+0.5\tanh\lVert x\rVert$, evaluated at the coordinate-norm distribution. Weight-invariant (§2.6). The magnitude tracks where coordinates sit relative to the projection boundary ($r \approx 0.95$), not learned geometry. | Signed; negative by construction of $\lambda$ |
| $-0.98$ | Same estimator, different evaluation run | Same architecture constant, read at a different coordinate-norm distribution (closer to the origin). The spread across the three values is the norm distribution shifting between evals, not geometry evolving. | Signed |
| $+1.71$ | `validate_geometry.py::compute_sectional_curvature` | A *turning-rate proxy*: mean norm of the change in normalized token-difference directions over the final hidden layer, $\lVert \hat{t}_{i+1} - \hat{t}_i \rVert$. Non-negative by construction ($\in [0,2]$); it cannot express hyperbolicity and is not commensurable with any sectional $\kappa$. | Unsigned |
| $\sigma = 2.2$ (retracted) | Weight-statistics probe (early thesis) | A standard deviation of parameter values, mislabeled as curvature. Retracted in the Corrected Thesis as error #1. | N/A |

**Resolution:** no single $K$ exists to quote. The only defensible statements are: (i) the fixed Poincaré kernels implement $K=-1$ geometry exactly, by construction; (ii) the learned metric factor converged to identity (flat); (iii) all telemetry previously reported as evidence of learned hyperbolicity originates in fixed components. Prior documents quoting "$K=-5.63$, strongly hyperbolic" as a training outcome are superseded by this section.

### 2.6 A Negative Methodological Result: Curvature Telemetry Unfalsifiable by Construction

We report a result we believe has value beyond this project. The adapter's conformal metric is $g(x) = \lambda(x)\,(LL^\top + \epsilon I)$ with $\lambda(x) = 1 + a\tanh\lVert x\rVert$ and $a = 0.5$ **hardcoded**. Therefore

$$\log\det g(x) = D\,\log\lambda(x) + \log\det(LL^\top + \epsilon I),$$

where the second term is position-independent and cancels exactly in any finite-difference Laplacian. Every curvature statistic derived from $\Delta \log\det g$ is thus a function of the hardcoded $\lambda$ and the coordinate-norm distribution *only* — it contains no trained parameter. This was verified empirically on the cp3000 checkpoint: the estimator run on the trained `metric_chol`, on the identity, and via the chol-free closed form $\kappa = -\tfrac{D}{2}\partial_k^2\log(1+0.5\tanh\lVert x\rVert)$ agree to <0.1% (trained: $-141.381$; identity: $-141.367$; analytic: $-141.335$), while a random-Cholesky control deviates only through log-det clamp saturation. An in-repo reproduction (`scripts/kappa_invariance_test.py`, CPU, float64, ball-interior positions) sharpens this to machine precision: trained vs identity relative deviation $1.2\times10^{-12}$, with $\lVert L_{\text{trained}} - I\rVert/\lVert I\rVert = 0.079$ independently confirming metric convergence to flat, and a radius sweep ($\kappa = -130.8$ at $r{=}0.1$ to $-4.4$ at $r{=}0.9$) locating the published $-5.63$/$-5.72$ on the fixed curve near the $r \approx 0.95$ projection boundary.

The consequence: the training regularizer $\mathrm{MSE}(\kappa, -1)$ was optimizing a target **no parameter could reach**. That it sat at $\kappa \approx -5.7$ throughout training without consequence is itself the signature of an unfalsifiable loss — no amount of training could have moved it, and no evaluation through this channel could have falsified the hyperbolicity claim. We propose the general methodological warning: *in conformal-factor architectures, curvature telemetry must be checked for weight-sensitivity before being reported as a training outcome.* A sufficient test is the one used here: evaluate the estimator under (trained, identity, random) metric parameters at fixed positions; agreement between trained and identity indicates the telemetry is an architecture constant.

Remediation implemented in the codebase (2026-07): the estimator now computes the full conformal scalar-curvature formula $R = \lambda^{-1}\left[-(n{-}1)\Delta\log\lambda - \tfrac{(n-1)(n-2)}{4}\lvert\nabla\log\lambda\rvert^2\right]$ at sectional scale, and the conformal amplitude $a$ is trainable under `config.learnable_conformal=True`, making the regularizer bindable. Default configuration preserves the historical behavior for checkpoint reproducibility, with the caveat documented at the telemetry site.

### 2.7 Terminology Note: Two Metrics Named "MFR"

Two unrelated quantities share the acronym MFR in this project's artifacts and must not be conflated:

1. **MFR = Model-First Reasoning compliance** (`mfr_utils.py`, `eval_arc.py`): a *prompting-protocol* metric — whether the model's output respects the two-phase Model-First Reasoning format. Measured value on ARC (n=100): **0.0%** (`arc_evaluation_results.json`).
2. **MFR = Manifold Faithfulness Rate** (thesis documents): a *geometric-operations* metric — the pass rate of mathematical faithfulness tests on the Poincaré kernels. Measured value: **100%**, with the scope caveat of §1 (it verifies fixed kernels, not learned geometry).

The "94.2% MFR" figure appearing in earlier documents (`docs/llms.txt`, `WIKI.md`, thesis PDF) matches *neither* measured value and has no surviving provenance; it is retracted. In this paper we avoid the acronym entirely: we write **MFR-compliance** for (1) and **kernel faithfulness** for (2).

## 3. Claim Tiers (Pre-Registered Scope)

| Claim | Tier | Supporting artifact |
|---|---|---|
| Poincaré kernels (Möbius add, exp/log, distance) are mathematically correct | EMPIRICAL | Ganea-formula verification; kernel faithfulness tests (100%); regression suite `tests/test_audit_fixes.py` |
| Learned metric converged to flat ($LL^\top \approx I$) | EMPIRICAL | cp3000 eigenspectrum analysis |
| Curvature telemetry weight-invariant with hardcoded $\lambda$ | PROVEN + EMPIRICAL | Three-line derivation + cp3000 invariance test (§2.6) |
| Zero geometric transfer through frozen downstream layers | EMPIRICAL | ARC-Challenge 54.86%, identical to base Qwen; ARC eval accuracy 0.01 (n=100) |
| Degenerate softmax routing as mechanistic cause of non-transfer | HYPOTHESIS | Bundle-Lock diagnostics; entropy collapse telemetry |
| Falsification protocol vs compute-matched LoRA | PRE-REGISTERED | §2.3–§2.4; results pending |

