# CONTEXT: ManifoldGL Phase 2 (Hamiltonian Dynamics)

## The Paradigm Shift
We are moving from **Static Riemannian Geometry** (Phase 1) to **Hamiltonian Dynamical Systems** (Phase 2).
Instead of simple geometric embeddings, our fiber latents now possess **Momentum**.
They evolve according to Hamiltonian equations of motion on the Poincaré Ball manifold.

## Core Concepts
1.  **Hamiltonian Evolution (Forward Pass)**:
    *   The "Thought Process" is modeled as a physical trajectory in phase space $(q, p)$.
    *   $H(q, p) = T(p) + V_{geo}(q) + V_{sem}(q)$.
    *   The system "thinks" by simulating physics for $T$ timesteps.

2.  **Retrospection (Backward Pass)**:
    *   To ensure semantic stability, we implement a **Time-Reversal Symmetry Check**.
    *   We reverse momentum at the end of the thought: $(q_T, p_T) \to (q_T, -p_T)$.
    *   We simulate backwards.
    *   **Retrospection Loss**: The distance between the reconstruction $q_0'$ and the original $q_0$.
    *   *Interpretation*: If the system forgets where it came from, it's hallucinating.

3.  **Implementation Strategy**:
    *   **Fiber Representation (Option A)**: Latent $z \in \mathbb{R}^d$ acts as the coordinate $q$.
    *   **Effect Discipline**: Updates are constrained by adjacency graphs (Section 4.3).

## Quick Start
*   **Math Specs**: See `MANIFOLD_GL_PHASE2_GIST.md`
*   **Action Plan**: See `TODO.md` (Start with TODO-001)

## Phase 3 Certification (Post-Deployment)
**Status: VERIFIED (Neurosymbolic Manifold v3.0)**
The theoretical "Retrospection" mechanism has been successfully implemented as a runtime **Neurosymbolic Jump** in Phase 3.
*   **Behavior**: When the model deviates from the "True Geodesic" (Constraint Violation or Loop), the Retrospection Error (Energy) spikes.
*   **Action**: The system triggers a "Context Reset" and "Orthogonal Expansion" (Hyper-Jump).
*   **Proof**: The "Neural Glass" demo confirms loop breaking and constraint enforcement (e.g., "Sky Blue" test).
The "Ghost" is now active in the shell.
