# Formal Comparison: IGBundle Adapters vs. Riemannian KV Cache

## 1. Problem Statement: The Core Tension

In a standard Causal Transformer, the "memory" used during autoregressive inference is effectively the **Attention Mechanism** operating on the **KV Cache** (stored pairs of $(K, V)$).

The core distinction between our approaches is **where** geometry is injected:

*   **Mode A (IGBundle Adapter)**: Geometry as **Transformation**.
    *   Maintains Euclidean retrieval (Dot Product Attention).
    *   Applies geometric structure *after* retrieval to refine the hidden state dynamics.
    *   **Philosophy**: Consistent with PEFT (Parameter-Efficient Fine-Tuning). "Reasoning" is geometric, but "Memory" is Euclidean.

*   **Mode B (Riemannian KV Cache)**: Geometry as **Retrieval**.
    *   Redefines the similarity kernel itself (Geodesic Distance vs Dot Product).
    *   Changes *what* is retrieved by altering the definition of "neighborhood".
    *   **Philosophy**: "Semantic Proximity" is defined by manifold curvature (e.g., hierarchies are naturally separated).

---

## 2. Mathematical Formalization

### Notation
*   **Backbone**: $f_\theta$ (Transformer, frozen).
*   **Hidden State**: $x^\ell \in \mathbb{R}^H$ at layer $\ell$.
*   **Projections**: $q = W_Q x, k = W_K x, v = W_V x$.
*   **Manifold**: $\mathcal{M}$ (e.g., Poincaré Ball $\mathbb{B}_c^d$ with curvature $\kappa = -c, c > 0$).
*   **Geometric Map**: $\phi: \mathbb{R}^H \to \mathcal{M}$.

### Curvature Convention
We adopt the standard convention where $c > 0$ represents the *magnitude* of curvature, so $\kappa = -c$.
For the Poincaré Ball model with curvature $-c$, the distance is defined as:

$$ d_{\mathbb{B}}(x,y) = \frac{1}{\sqrt{c}} \operatorname{arcosh}\left(1 + \frac{2c\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\right) $$

*Constraint*: $\|x\| < 1/\sqrt{c}$ (The boundary is at $1/\sqrt{c}$).

### Mode A: IGBundle Adapter (Post-Attention)
The backbone performs standard Euclidean attention:
$$ y_{raw} = \mathrm{Att}_{Euclidean}(Q, K, V) $$

The adapter injects geometry as a residual correction:
$$ y = y_{raw} + \Delta_\psi(y_{raw}) $$

Where $\Delta_\psi$ projects to $\mathcal{M}$, computes fiber bundle dynamics (geodesic flow, parallel transport), and maps back. This modifies the *representation* but not the *retrieval rank*.

### Mode B: Riemannian KV Cache (In-Attention)
We replace the Euclidean kernel with a geometric one.

**1. Geometric Score**:
Instead of $q \cdot k$, we use negative geodesic distance:
$$ \alpha_{ts} \propto \exp\left(-\frac{d_\mathcal{M}(\phi(q_t), \phi(k_s))^2}{\tau}\right) $$
*Note*: This requires storing $\phi(k_s)$ in the KV Cache ($K^\mathcal{M}$), making the cache effectively Riemannian.

**2. Geometric Aggregation**:
Standard summation $\sum w_s v_s$ is not well-defined on curved manifolds (the result leaves the manifold). Two standard solutions:

*   **Tangent Space Aggregation** (Preferred for KV):
    Map values to the Tangent Space at the weighted centroid (or origin), average in Euclidean space, and map back.
    $$ v_{agg} = \exp_{\mathbf{0}}\left( \sum_s w_s \log_{\mathbf{0}}(v_s) \right) $$
    *Simplification*: If $V$ lives in Euclidean space (only $K$ is on Manifold), standard summation is valid for $V$. If $V$ is also on $\mathcal{M}$, use Tangent Space.

*   **Einstein Midpoint** (Barycenter):
    $$ m = \frac{\sum w_s \gamma(v_s) v_s}{\sum w_s \gamma(v_s)}, \quad \gamma(v) = \frac{1}{\sqrt{1-c\|v\|^2}} $$

---

## 3. Theoretical Guarantees: Kirszbraun & Lipschitz Stability

We seek to ensure that our geometric map $\phi$ is robust—i.e., it preserves the neighborhood structure of the pre-trained space and doesn't "teleport" stable Euclidean points to chaotic manifold locations.

**Kirszbraun’s Theorem (Existence Guarantee)**:
A Lipschitz-continuous map defined on a subset of a Hilbert space can be extended to the whole space with the same Lipschitz constant.
*Refinement (Lang-Schröder)*: Similar extension theorems exist for metric spaces with bounded curvature (CAT(k) spaces).

**The Practical Implication**:
Kirszbraun guarantees that *if* we constrain $\phi$ to be $L$-Lipschitz on the training data ("seen" points), a stable extension *exists* for unseen data. It motivates explicit regularization during training:

**Constraint Definition**:
$$ \sup_{x \ne y} \frac{d_\mathcal{M}(\phi(x), \phi(y))}{\|x - y\|_2} \le L $$

**Implementation**:
1.  **Spectral Normalization** on the projection matrix $W_\phi$.
2.  **Gradient Penalty**: Penalize $\|\nabla_x \phi(x)\|$ if it exceeds a threshold.

---

## 4. Proposed Hybrid Architecture (Dual-Path)

To balance the benefits of Hierarchical Retrieval (Riemannian) with the robust general knowledge of the Pre-trained LLM (Euclidean), we propose a **Hybrid Attention Head**:

$$ \mathrm{Att}_{Hybrid} = \lambda_{gate} \cdot \mathrm{Att}_{Euclidean}(Q, K^E, V^E) + (1-\lambda_{gate}) \cdot \mathrm{Att}_{\mathcal{M}}(\phi(Q), K^\mathcal{M}, V^\mathcal{M}) $$

*   **Dual Cache**: The system maintains both $K^E$ (Standard) and $K^\mathcal{M}$ (Geometric).
*   **Gating**: $\lambda_{gate}$ can be learned or derived from **Manifold Entropy ($S$)**:
    *   High Entropy (Uncertainty) $\to$ Rely on Manifold Structure ($1-\lambda$ increases).
    *   Low Entropy (Rote tasks) $\to$ Rely on Euclidean backbone.

---

## 5. Validation Protocol

To prove that Mode B offers "Generalization" rather than just "Complexity", we define three tests:

1.  **Hierarchical Retrieval Test**:
    *   *Dataset*: Synthetic hierarchies (Trees) or WordNet hypernyms.
    *   *Metric*: Tree-Edit Distance or LCA (Lowest Common Ancestor) distance of retrieved keys vs ideal keys.
    *   *Hypothesis*: $\mathrm{Att}_\mathcal{M}$ yields lower retrieval error on deep hierarchies than $\mathrm{Att}_{Euclidean}$.

2.  **Lipschitz Stability (OOD)**:
    *   *Method*: Perturb input $x \to x + \delta$.
    *   *Metric*: Empirical Lipschitz constant $ratio = d_\mathcal{M}(\phi(x+\delta), \phi(x)) / \|\delta\|$.
    *   *Success*: The ratio remains bounded even for OOD inputs (checking against "teleportation").

3.  **Cost vs. Accuracy Trade-off**:
    *   Measure wall-clock time per token with $d_\mathcal{M}$ scoring.
    *   Measure Cache RAM usage (Euclidean vs Dual).
