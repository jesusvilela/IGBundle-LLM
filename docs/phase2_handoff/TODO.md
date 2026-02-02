# TODO.md - Phase 2 Implementation Plan

## Prioritized Tasks

### 🔴 Critical Path

- [ ] **TODO-001: Implement Symplectic Integrator**
    *   **File**: `src/igbundle/dynamics/hamiltonian.py`
    *   **Task**: Implement `LeapfrogIntegrator` class.
    *   **Requirement**: Must handle non-Euclidean geometry (use `exp_map` for position update).
    
- [ ] **TODO-002: Implement Retrospection Loss**
    *   **File**: `src/igbundle/training/losses.py`
    *   **Task**: Create `RetrospectionLoss` module.
    *   **logic**: Forward sim -> Stop gradient -> Backward sim -> distance(start, end).

- [ ] **TODO-003: Update Fiber State**
    *   **File**: `src/igbundle/fibers/state.py`
    *   **Task**: Add `momentum` tensor to `FiberState`. Update initialization to sample momentum (e.g., from Gaussian).

### 🟡 Integration

- [ ] **TODO-004: Adjacency Graph**
    *   **File**: `src/igbundle/graph/adjacency.py`
    *   **Task**: Implement `AdjacencyGraph` with `get_neighbors(fid)`.
    *   **Edges**: Tree (Parent-Child) + Overlap (Shared Tokens).

- [ ] **TODO-005: Effect Discipline Router**
    *   **File**: `src/igbundle/modules/bundle_router.py`
    *   **Task**: Update `forward` to use `propagate()` + `anchor_off_locus()`.

### 🟢 Training

- [ ] **TODO-006: Phase 2 Training Loop**
    *   **File**: `train_phase2.py`
    *   **Task**: Create training loop that minimizes `L_task + lambda * L_retro`.
    *   **Validation**: Ensure `L_retro` decreases over time (System becomes reversible).

## Code Snippets

**Anchor Off-Locus Logic:**
```python
def anchor_off_locus(fibers, allowed, prev_states, alpha=0.95):
    for fid in fibers:
        if fid not in allowed:
            fibers[fid].z = alpha * prev_states[fid].z + (1-alpha) * fibers[fid].z
```
