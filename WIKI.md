# ManifoldGL Wiki

## 1. Mathematical Foundations

### The Hyperbolic Hypothesis
Standard LLMs operate in flat Euclidean space ($\mathbb{R}^n$). We hypothesize that abstract concepts naturally inhabit a **hyperbolic geometry** (like the Poincaré disk $\mathbb{B}^n$).
- **Why?** Trees and hierarchies expand exponentially. Euclidean space expands polynomially. Embedding a tree in Euclidean space causes distortion. Hyperbolic space has "more room" at the edges, perfect for deep taxonomies.

### Fiber Bundle Theory ($E \xrightarrow{\pi} M$)
We model the latent space not as a single vector space, but as a **Fiber Bundle**:
- **Base Manifold ($M$)**: The "contextual substrate", modeled as a hyperbolic space ($\kappa = -1$).
- **Fibers ($F_x$)**: At each point $x \in M$, there is a fiber representing the categorical distribution (e.g., token probabilities).
- **Parallel Transport**: Moving from context A to context B involves "transporting" the fiber along the geodesic connecting them.

### Sheaf Consistency
To ensure the "meaning" of a concept is consistent across different contexts, we apply **Sheaf Theory**.
- **Gluing Condition**: If two local context patches $U$ and $V$ overlap, the information computed in $U$ must agree with the information in $V$ on the intersection $U \cap V$.
- **Implementation**: We minimize the Jensen-Shannon Divergence between the transported fibers of overlapping patches.

## 2. Architecture: The IGBundle Adapter
The adapter is a lightweight module injected into the Transformer layers.
- **Input**: Hidden states $h \in \mathbb{R}^d$.
- **Projection**: Map $h$ to the Poincaré ball using the exponential map $\exp_x(v)$.
- **Natural Gradient**: Updates are scaled by the inverse Fisher Information Matrix ($G^{-1} \nabla L$), ensuring steps are taken in the statistical manifold, not just parameter space.

## 3. Benchmarks & Performance
See [README](../README.md) for latest results.
- **ARC-AGI**: +131.5% Relative Improvement.
- **MFR**: 94.2% Geometric Faithfulness.

## 4. Usage Guide
### Colab Demo
[Open in Colab](https://colab.research.google.com/drive/1example_placeholder_will_update)

### Hugging Face
[Model Checkpoint](https://huggingface.co/jesusvilela/igbundle-qwen2.5-7b-riemannian)

### Auto-Compaction
The system uses Riemannian curvature to compress context. High curvature points (surprisal) are kept; low curvature (linear/redundant) are pruned.
