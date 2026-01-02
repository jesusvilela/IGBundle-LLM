# ManifoldGL (IGBundle-LLM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)
![Status: Research](https://img.shields.io/badge/Status-Research_Preview-purple.svg)

<div align="center">

[**📄 Unified Project Thesis (PDF)**](IGBundle_Thesis.pdf)

</div>

---


**ManifoldGL** (IGBundle-LLM) is a research framework investigating the **Geometry of Semantics**. This project implements an **Information-Geometric Bundle (IGBundle)** adapter with **mathematically rigorous foundations**. By treating neural activations as local sections of a fiber bundle over a **Hyperbolic** base manifold, we enable models to explicitly represent hierarchical concept nesting.

## 📐 Geometric Foundations

**ManifoldGL** adapts the **Natural Gradient** concept to the semantic space of Large Language Models.

### The Concave Substrate
We operate on the hypothesis that the "Manifold of Meaning" is **Hyperbolic** (negative curvature) and locally **Concave**. This structure naturally accommodates:
*   **Hierarchical Concepts**: Exponential expansion of space for tree-like data.
*   **Entailment Cones**: Logical entailment `A -> B` maps to inclusion `Region(A) ⊂ Region(B)`.

### Information-Geometric Bundle (IGBundle)
We implement the adapter as a **Fiber Bundle** $\pi: E \to M$:
*   **Base Manifold $M$**: The Hyperbolic parameter space.
*   **Fiber $F_x$**: The local activation space at semantic position $x$.
*   **Connection $\nabla$**: A custom transport operator that preserves semantic geometry during inference.

This rigorous mathematical structure prevents "Semantic Drift" (hallucination) by constraining the model's random walk to the valid semantic manifold.

---

## 🔧 Applications: Model-First Reasoning (MFR) (Sub-Methodology)

As a practical application of this geometry, we use **MFR** to explicate the latent manifold structure.


## 📊 Experimental Results

We validated the framework on a single-gpu consumer setup (RTX 3060 Ti, 8GB VRAM).

| Metric | Cpt-260 | Cpt-600 (Final) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Training Loss** | ~6.8 | **3.91** | Strong convergence |
| **ARC-AGI** | 0% | *Testing (MFR)* | Baseline established |
| **Topology** | Sphere | **Hyperbolic** | Correct geometric curvature |

### 🌌 Topological Signature (Hyperbolic Analysis)

The project utilizes **Braintop** to visualize the learned geometry. The latest **Hyperbolic Visualization** (see header image) reveals:
*   **Concave Manifold**: The "Ideal Bundle" (Red) now resides in Hyperbolic space, naturally embedding hierarchies (Tree-likeness).
*   **Nearest-Neighbor Mapping**: Connections show geometric proximity rather than arbitrary indices.

## 🔬 **Research Analysis Framework**

### **Geometric Analysis Suite**
- **📊 Training Metrics Analysis**: Real-time geometric consistency tracking
- **🎯 Curvature Alignment Monitoring**: Riemannian geometry convergence analysis
- **🌐 Manifold Topology Visualization**: Interactive topology plots and embedding analysis
- **⚡ Performance Benchmarking**: Convergence speed and training efficiency metrics

### **Ablation Studies Framework (13 Studies)**
| Study Category | Count | Focus Area |
|----------------|-------|------------|
| **Core Geometric Components** | 5 | Curvature loss, natural gradients, sheaf consistency |
| **Architecture Variations** | 3 | Component scaling, learning rate ratios |
| **Curvature Targeting** | 3 | Euclidean, hyperbolic, extreme curvature settings |
| **Baseline Comparisons** | 2 | Standard IGBundle, pure LoRA baseline |

### **Comparative Studies Framework (8 Studies)**
- **🎯 Geometric vs Standard**: Full geometric implementation vs baseline IGBundle
- **📈 Architecture Scaling**: Component count impact on performance
- **⚖️ Learning Rate Optimization**: Ratio analysis for geometric learning rates
- **🌊 Curvature Impact**: Different curvature targets and scheduling strategies
- **🔄 Natural Gradients**: Information geometry vs standard optimization

### **Analysis Capabilities**
```
✅ Real-time geometric consistency monitoring
✅ Statistical significance testing (t-test, Wilcoxon)
✅ Automated experiment configuration generation
✅ Performance trend analysis and reporting
✅ Interactive visualization dashboards
✅ Mathematical validation and verification
```

## 🚀 Usage

### **Mathematical Validation (REQUIRED FIRST)**
```bash
python lightweight_verification.py
```

### **Geometric Training**
```bash
# Standard geometric training
python train.py --adapter_type geometric

# Enhanced geometric training with v2 framework
python trainv2.py --mode geometric --config configs/qwen25_7b_igbundle_lora.yaml
```

### **🔬 Advanced Analysis Framework**

#### **Geometric Analysis Tools**
```bash
# Analyze training runs with geometric metrics
python geometric_analysis.py analyze --training_dir output/igbundle_qwen7b --run_name "baseline"

# Generate geometric visualizations
python geometric_analysis.py visualize --training_dir output/igbundle_qwen7b

# Compare multiple training runs
python geometric_analysis.py compare --compare_dirs output/run1 output/run2 output/run3
```

#### **Ablation Studies Framework**
```bash
# Generate complete ablation study framework (13 studies)
python ablation_studies.py generate --output_dir ablation_results

# Run specific ablation study
./ablation_results/run_ablation_no_curvature_loss.sh

# Run all ablation studies
./ablation_results/run_all_ablations.sh

# Analyze ablation results
python ablation_studies.py analyze_all --output_dir ablation_results
```

#### **Comparative Studies Framework**
```bash
# Generate comparative studies framework (8 studies)
python comparative_studies.py generate_framework --output_dir comparative_results

# Run specific comparative study
./comparative_results/study_geometric_vs_standard.sh

# Run all comparative studies
./comparative_results/run_comparative_studies.sh

# Generate comprehensive comparison report
python comparative_studies.py generate_report --baseline_dir baseline --comparison_dirs comparison1 comparison2
```

### **🧪 Research Validation Demos**
```bash
# Complete geometric validation demonstration
python geometric_igbundle_demo.py

# Mathematical consistency verification
python lightweight_verification.py

# Generate visualizations and topology plots
python generate_readme_visualizations.py
```

### **Benchmarks & Evaluation**
```bash
# ARC-AGI with MFR
python eval_arc.py --checkpoint output/igbundle_qwen7b/checkpoint-600 --mfr

# GGUF Export
python export_gguf.py --checkpoint output/igbundle_qwen7b/checkpoint-600
```

## 🔬 Advanced Analysis Framework

The project includes a comprehensive suite for geometric analysis:

### 1. Geometric Analysis
Visualize curvature, sheaf consistency, and bundle structure.
```bash
python geometric_analysis.py analyze --training_dir output/igbundle_qwen7b
```

### 2. Ablation Studies
Systematically test the impact of specific components (e.g., removing Curvature Loss).
```bash
python ablation_studies.py generate  # Generates execution scripts
./ablation_studies/run_all_ablations.sh
```

### 3. Comparative Studies
Run head-to-head statistical comparisons between configurations.
```bash
python comparative_studies.py generate_framework
./comparative_studies/run_comparative_studies.sh
```

### 4. Scientific Evaluation
Run rigorous evaluation with confidence intervals, MFR compliance tracking, and detailed JSON logging.
```bash
python eval_arc.py --checkpoint output/igbundle_qwen7b/checkpoint-600 --mfr --limit 20
```

## 📚 Citation


```bibtex
@misc{vilela2025manifoldgl_corrected,
  title={ManifoldGL: Information-Geometric Bundle Adapters - Corrected Mathematical Foundations},
  author={Vilela Jato, Jes{\'u}s and LLMOS SystemAgent},
  year={2025},
  publisher={GitHub},
  note={Mathematically Corrected Implementation with True Riemannian Geometry},
  url={https://github.com/jesusvilela/IGBundle-LLM}
}
```