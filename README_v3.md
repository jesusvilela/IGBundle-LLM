# Neurosymbolic Manifold v3.0

**Project**: Information Geometric Bundle (IGBundle) - Phase 3
**Status**: DEPLOYED

## Overview
This repository contains the implementation of a **Neurosymbolic Thinking System** that combines:
1.  **Geometric Manifold Learning**: Thoughts exist on a Poincare Disk ($K=-1$).
2.  **Semantic Potential Field** ($V(q)$): A learned energy landscape that "tilts" reasoning towards truth.
3.  **Meta-Cognitive Loop** (System 2): Recursive self-correction before generation.
4.  **Neurosymbolic Hyer-Jump**: A mechanism to forcefully break repetition loops by inverting logic.

## Quick Start
1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
    
2.  **Run Neural Glass (Demo)**:
    ```bash
    python app_neural_glass.py
    ```
    *   This launches a Gradio UI at `http://localhost:7865`.
    *   Watch the "Thought Manifold Trace" for System 2 events ("Energy X -> Y") and Hyper-Jumps.

## Architecture
*   `src/igbundle/modules/geometric_adapter.py`: Core logic (Manifold + Physics).
*   `src/igbundle/dynamics/hamiltonian.py`: Physics engine (Symplectic Integration).
*   `src/igbundle/cognition/meta.py`: Meta-Cognitive Loop.
*   `src/igbundle/fibers/executor.py`: Fiber Jumping logic.

## Weights
*   `output/phase3_adapter_potential.pt`: Trained Adapter weights (Phase 3).

## Visualization
Run `python vis_potential.py` to see the Potential Landscape.
