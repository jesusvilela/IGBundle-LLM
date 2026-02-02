# Bridging Geometric Structure to Reasoning Dynamics

## Hypothesis: Geodesic-Guided Attention
Incorporate manifold distances into the model's reasoning process. Instead of standard attention (which is agnostic to geometry), use **geodesic distances** on the learned manifold to guide attention weights:
- Tokens/concepts that are closer on the hyperbolic manifold get higher attention weights (favor local context).
- Farther concepts get exponentially lower weight, unless "pulled in" by query relevance.

## Rationale
On the manifold, distance = dissimilarity. A geodesic attention would naturally enforce a form of **hierarchical reasoning**: the model focuses on closely related concepts first, then gradually extends outward (like expanding a search tree).
This could prevent the model from jumping to unrelated context and **preserve semantic coherence** during reasoning.

## Proposed Implementation Steps
1. **Distance Matrix Computation**: During forward pass, compute pairwise hyperbolic distances between token representations (using the Poincaré metric).
2. **Geodesic Attention**: Modify the attention softmax: 
   $$ \alpha_{ij} = \mathrm{softmax}(-d_{\mathbb{B}}(x_i, x_j) / \tau) $$
   where $d_{\mathbb{B}}$ is the hyperbolic distance and $\tau$ is a temperature.
3. **Parallel Transport in Attention**: Ensure that when attending across disparate manifold regions, we transport information along the manifold (using the connection) rather than in ambient space. This maintains consistency of what a head “knows” about distant tokens.

## Potential Benefits
- **Locality Bias**: Encourages using local context (which is likely more relevant) before global context, aligning with human-like stepwise reasoning.
- **Hierarchical Inference**: Might naturally follow the hierarchy in data (first resolve general context, then specifics).
- **Reduced Semantic Drift**: Attention guided by manifold structure could keep the model’s intermediate states on-track (less off-topic divergence).

## Evaluation Plan
Design a controlled task (e.g., a tree traversal puzzle or a hierarchical classification) to compare:
- **Standard Attention vs. Geodesic Attention** using the same model.
Metrics: task accuracy, attention entropy (does geodesic version concentrate more?), and qualitative inspection of attention maps on the manifold.
