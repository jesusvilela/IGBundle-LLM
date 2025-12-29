# ManifoldGL: Information-Geometric Bundle Adapters for LLMs

![Manifold Topology](igbundle_topology.png)

> "Language is non-Euclidean. Meaning lives in the fibers."

## Abstract

**ManifoldGL** (IGBundle-LLM) is an experimental research framework that challenges the "flat space" assumption of contemporary Large Language Models. By integrating concepts from **Differential Geometry** and **Sheaf Theory**, we propose that semantic ambiguity and context-dependence are best modeled as curvature in a fiber bundle, rather than vector superpositions in a Euclidean space.

This repository contains the implementation of the **IGBundle Adapter**, a bottleneck architecture that projects standard Transformer activations into a low-dimensional "bundle space" where consistency is enforced via Sheaf Cohomology constraints.

## Theoretical Foundation

Traditional Transformers treat word embeddings as points in a flat vector space $\mathbb{R}^d$. However, the semantic space of natural language is inherently hierarchical and curved (hyperbolic).

We hypothesize:
1.  **Base Manifold ($M$)**: The structural "grammar" of language forms a base manifold.
2.  **Fiber Bundle ($E \xrightarrow{\pi} M$)**: The set of all possible meanings for a given context forms the fibers $F$.
3.  **Parallel Transport**: The attention mechanism acts as a connection $\nabla$, transporting meaning along the path of the sentence.

Our architecture computes the **Local Section** of this bundle. The internal metric **Sigma ($\sigma$)** measures the *holonomy* or curvature of the path—essentially quantifying how much "ambiguity" or "information density" exists in the current context.

## Research Artifacts

### 1. The Thesis
For a comprehensive overview of the mathematical motivation, methodology, and results, please refer to the project thesis:
📄 **[Read the Thesis (PDF)](IGBundle_Thesis.pdf)**

### 2. Interactive Topology
The visualization above represents the learned 256-dimensional tangent bundle projected into $\mathbb{R}^3$. You can examine the interactive topology report here:
✨ **[View Interactive Manifold](igbundle_topology.html)**

## Installation & Replication

The framework is optimized for consumer hardware (8GB VRAM) using 4-bit quantization and gradient accumulation.

```powershell
# Windows Setup (Powershell)
& "unsloth_env\Scripts\Activate.ps1"
```

### Reproducing Results
1.  **Train**: `python train.py --config configs/qwen25_7b_igbundle_lora.yaml`
2.  **Validate**: `python validate_effects.py` (Compares Base vs Bundle outputs)
3.  **Visualize**: `python generate_braintop_viz.py`

## License

(c) Jesús Vilela Jato, all rights reserved.
