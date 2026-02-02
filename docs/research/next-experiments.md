# Next Experiments and Analyses

## 1. Full ARC Benchmark Run (Phase II)
**Goal**: Obtain the actual task accuracy for ManifoldGL on ARC-AGI with n=100 tasks.
- Use the improved pipeline to run the evaluation.
- Record accuracy, and also measure **MFR during evaluation** (does it remain 100% at inference time?).
- Compare to baseline Qwen-7B on the same task set.
- Analyze failure cases (collect qualitative examples where model fails to see if geometry suggests a reason).

## 2. Ablation Study Execution
Run the critical ablations one by one:
- `no_curvature_loss`: Does performance drop without curvature regularization? Hypothesis: yes, if hyperbolicity was helping organization.
- `euclidean_target` (flat manifold): Does using a Euclidean latent (σ = 0 effectively) change the representations notably? Measure MFR (should drop) and task accuracy (likely drops if our hypothesis holds).
- `no_sheaf_loss`: Turn off the sheaf consistency and see if the model starts to develop contradictory local representations (monitor if some parts of context go out of sync).
- `LoRA_baseline`: Swap out the manifold components with a standard LoRA of the same capacity to see if just more parameters would have done the same (expect ManifoldGL to do better on structural tasks).

Each ablation should be run for enough steps (preferably 1000) to observe a difference. Report not just accuracy, but also geometric metrics (curvature achieved, etc.).

## 3. Visualization and Interpretation
- **Christoffel Symbols Analysis**: If we have an approximation of Christoffel symbols from the model, visualize them or their distribution. Do they correlate with intuitive transitions (e.g., higher magnitude when context shifts from one topic to another)? This might be tough, but could reveal if the model’s learned connection has meaningful structure.
- **Geodesic Paths**: Take two distant concepts in the manifold (from the vocabulary) and interpolate (move along the geodesic). Decode intermediate points (via the decoder) to see if we get a sensible semantic interpolation (like concept A → abstract blend → concept B). This tests if manifold distances correspond to real semantic continuity.
- **Fiber Category Meaning**: For the categorical part of the representation, try to interpret each dimension or cluster of the fiber. Are certain fiber components always active for certain types of tokens (e.g., maybe one corresponds to “noun vs verb”, another to “abstract vs concrete”)? This could be done by inputting various known tokens and looking at the fiber output.

## 4. Integration of Dynamics (Prototype)
As discussed, implement a simple **Hamiltonian dynamics** experiment:
- Define a toy Hamiltonian H on the manifold (for instance, kinetic energy = movement on manifold, potential = some function of position like distance to a target concept).
- Simulate the model’s representations as particles moving under this H for a few steps (this is outside the model’s own operation; think of it as a post-process).
- See if this process can solve a simple reasoning task (e.g., move a concept representation to a goal state by following energy gradients).
While rudimentary, this will build intuition for coupling dynamics with geometry. 

## 5. Multi-Task or Few-Shot Transfer
Test if the learned manifold helps in few-shot learning:
- Fine-tune the ManifoldGL adapter on a **different reasoning task** (say, a smaller puzzle dataset or logical QA) but reuse the same pre-trained geometric adapter (either frozen or lightly fine-tuned).
- Does it learn faster or better than a fresh LoRA would? If the manifold truly captures universal semantics, we expect positive transfer.

Each experiment above should be logged with detailed metrics and, where possible, visualizations. As results come in, update the thesis or documentation to reflect **empirical findings**, not just theoretical expectations.
