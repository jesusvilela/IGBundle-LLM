# Adaptive Curvature Dynamics & Analysis

## Observation
The current model fixes target curvature $K_{\text{target}} = -1$ everywhere. In practice, different regions of the semantic space might require different curvatures:
- Some concepts might live in “flatter” subspaces, others in more highly curved pockets (e.g., very hierarchical domains like taxonomy could benefit from strong negative curvature, whereas a linear scale concept like numbers might be nearly flat).

## Hypothesis: Dynamic Curvature (Ricci Flow on Manifold)
Allow the manifold curvature to *evolve during training* or even vary across the space:
- Use a small network to predict a local curvature $K(x)$ for each point or region.
- Penalize deviation from an overall negative curvature prior, but not force uniform $-1$ everywhere.

This becomes analogous to **Ricci flow** in Differential Geometry: the metric (and thus curvature) adjusts in different areas to optimize some objective (here, maybe to minimize task loss while keeping a general curvature prior).

## Proposed Experiment
1. Implement a **curvature field**: $K(z) = -1 * \sigma(KNN(z))$ for example, where $\sigma$ could be a learned function based on neighbor properties (or a separate tiny MLP that takes latent representation and outputs a curvature value).
2. Add a term to loss that encourages $K(z)$ to be mostly negative (to maintain hyperbolicity on average) but not strictly -1 everywhere.
3. Train on a small dataset and monitor:
   - Does $K(z)$ diverge from -1 in some regions? (e.g., does the model find it beneficial to flatten some parts of the space?)
   - Does allowing curvature freedom improve task performance?

## Analysis Goals
- **Region-specific Geometry**: Identify if certain semantic areas (e.g., abstract concepts vs. concrete descriptors) correlate with different curvatures.
- **Convergence**: See if the curvature field stabilizes or keeps evolving (like a physical process reaching equilibrium).
- **Comparison with Fixed Curvature**: Does this flexibility yield higher MFR or task accuracy, or conversely does it break the manifold consistency?

## Potential Outcomes
A positive result would indicate that a **one-size-fits-all curvature is suboptimal**, and that models need a *heterogeneous geometry*. This could open up a new line of “adaptive geometry” research in representation learning.
Conversely, if performance doesn’t improve, it reinforces the idea that the primary issue is not geometry fidelity but how it's used for reasoning.
