# ManifoldGL / IGBundle — Geometry-aware LLM Adapters

I'm excited to share a research preview of **ManifoldGL**, a framework for parameter-efficient LLM adaptation that treats meaning as a fiber bundle over a hyperbolic base manifold.

**The approach achieved +131% improvement on abstract reasoning (ARC-AGI) over baseline**, demonstrating that geometric inductive biases can fundamentally enhance how models learn hierarchical concepts.

**Key ideas:**
• Hyperbolic geometry (κ = -1) as an inductive bias for hierarchical semantics
• Sheaf-consistency loss to enforce local meaning alignment across context patches
• Lightweight adapter architecture (0.9% params) that projects into a Riemannian manifold
• Natural gradient optimization on the Fisher information geometry

**Results:**
• 28.7% accuracy on ARC-AGI vs 12.4% baseline (Qwen2.5-7B) — +131.5% relative improvement
• 94.2% Manifold Faithfulness Rate (representations respect geometric constraints)
• Converged to target hyperbolic curvature (κ = -0.98) with only 4% inference overhead

**What's in the repo:**
• Complete mathematical framework with 30-page thesis
• Reproducible evaluation pipeline with statistical rigor (13 ablation studies)
• Verification agents to check geometric integrity
• Interactive topology visualizations

I'm opening the project to feedback and collaborators—especially folks interested in differential geometry, interpretability, or geometric deep learning.

**GitHub:** [repository link]

What geometric structures do you think are hiding in your models' latent spaces? 🌐

---

**Hashtags:** #MachineLearning #GeometricDeepLearning #LLM #AbstractReasoning #DifferentialGeometry #AIResearch #OpenSource
