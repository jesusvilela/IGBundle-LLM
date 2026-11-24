# IGBundle Research Thesis

## Core Hypothesis

A geometric adapter layer based on fiber bundle theory — with Poincare ball base manifold,
categorical fiber sections, Hamiltonian dynamics, and Riemannian curvature regularization —
can improve reasoning quality in large language models by providing a structured
information-geometric inductive bias that vanilla LoRA adapters lack.

## Research Axes

### Axis 1: Fiber Bundle Geometry for Neural Networks
The mathematical framework: principal bundles, connections, parallel transport,
holonomy, and curvature applied to transformer hidden states. The adapter operates
at a single transformer layer (L12) and introduces geometric structure into the
residual stream.

### Axis 2: Information Geometry of Language Models
Fisher-Rao metric, natural gradient, KL divergence as Riemannian distance on
the statistical manifold of token distributions. How curvature of this manifold
relates to model uncertainty and reasoning difficulty.

### Axis 3: Neuromorphic Memory (NMEM)
Biologically-inspired memory with FitzHugh-Nagumo excitable decay, FFT spectral
forgetting (detail before gist), Gabor retroactive interference, multi-anchor
retrieval (Method of Loci), savant preservation, ghost entries.

### Axis 4: Geometric Steering at Inference
The GSP (Geometric Steering Probe) — a GENERIC-compliant homeostatic controller
that uses curvature K and entropy S as feedback signals to modulate generation
in real time, without retraining.

### Axis 5: Falsification & Empirical Validation
Controlled experiment: geometric single-layer adapter vs multi-layer vanilla LoRA
at matched parameter count and matched layer coverage. Null hypothesis: vanilla >= geometric.
Three eval tiers: benchmarks, ARC-AGI reasoning, manifold faithfulness.

## Key Technical Components
- Base model: Qwen 2.5-7B (NF4 quantized)
- Adapter: GeometricIGBundleAdapter at layer 12
- Base manifold: Poincare ball H^64 (D=64, curvature c=1)
- Fibers: K=16 categorical sections, P=8 mixture components
- Dynamics: Hamiltonian system with symplectic integration
- Training: Odyssey v3 pipeline with geometric + language losses
- Inference: Neural Glass (Gradio) with GSP controller
- Memory: NMEM (neuromorphic) replacing IMD (3-tier CLS)

## Current Results
- Tier 3 Faithfulness: 6/6 PASSED (K=-5.63, S=0.95, JSD=0.424, PT=0.041)
- Geometry is real and measurable, not decorative
- EOS suppression crisis resolved via inference-time boost
- Falsification experiment: Configs B/C trained, Tier 1/2 evals running
