## IGBundle-LLM: Information-Geometric Bundle Adapter System
This codemap traces the IGBundle-LLM system, a research implementation of Information-Geometric Bundle Adapters for Large Language Models. The system injects geometric adapters into transformer layers [1a], operates using hyperbolic geometry on Poincare balls [3a], and maintains sheaf-theoretic consistency through specialized loss functions [5e]. Key components include the training pipeline with geometric regularization [1d], hyperbolic distance computations [3b], ARC-AGI evaluation with statistical validation [4e], and an autonomous verification system that continuously monitors geometric integrity [6c].

### 1. Training Pipeline with IGBundle Injection
How the system trains LLMs with geometric bundle adapters injected into transformer layers
### 1a. Inject IGBundle Adapter (`train.py:246`)
Wraps HuggingFace model with geometric bundle adapters
```text
model = wrap_hf_candidate(model, cfg.ig_adapter)
```
### 1b. Initialize State Collection (`train.py:265`)
Sets up hooks to capture bundle states during training
```text
collector = StateCollector()
```
### 1c. Create Sheaf Loss Function (`train.py:268`)
Initializes sheaf consistency loss for geometric constraints
```text
sheaf_loss = SheafLoss(cfg.loss.num_patches, cfg.ig_adapter.latent_dim, cfg.loss.tau)
```
### 1d. Compute Auxiliary Loss (`train.py:131`)
Adds sheaf consistency loss to standard language model loss
```text
aux_loss = torch.stack(layer_losses).mean() * self.lambda_glue
```
### 1e. Combine Losses (`train.py:138`)
Final loss combining LM loss with geometric regularization
```text
total_loss = loss + aux_loss
```
### 2. IGBundle Adapter Forward Pass
How geometric bundle adapters transform hidden states using hyperbolic geometry
### 2a. Input Bottleneck Projection (`adapter.py:86`)
Projects hidden states to bottleneck dimension
```text
h_bot = self.input_proj(x)
```
### 2b. Generate Mixture Parameters (`adapter.py:90`)
Creates Gaussian means for bundle components
```text
m = self.proj_m(h_bot).view(B, T, self.P, self.D_lat)
```
### 2c. Compute Bundle Affinity (`adapter.py:109`)
Calculates component interactions using hyperbolic geometry
```text
A = compute_affinity_matrix(m, log_sigma, d_fiber, self.cfg.alpha, self.cfg.beta, geometry=geometry)
```
### 2d. Message Passing (`adapter.py:116`)
Aggregates messages across bundle components
```text
mixed_msg = mix_messages(A, processed_feats)
```
### 2e. Output Transformation (`adapter.py:152`)
Returns transformed hidden states and new bundle state
```text
return x + self.scale * out, state_new
```
### 3. Hyperbolic Geometry Operations
Core geometric computations using Poincare ball model for hyperbolic space
### 3a. Project to Poincare Ball (`ops.py:107`)
Maps Euclidean means to hyperbolic space using tanh
```text
means_hyp = torch.tanh(means)
```
### 3b. Compute Geodesic Distances (`ops.py:111`)
Calculates hyperbolic distances between bundle components
```text
D_base = compute_poincare_distance(means_hyp, means_hyp)
```
### 3c. Poincare Distance Formula (`ops.py:33`)
Implements the hyperbolic distance calculation
```text
val = 1.0 + 2.0 * sq_euc_dist / ((1.0 - x_sq_norm) * (1.0 - y_sq_norm) + alpha_num_stab)
```
### 3d. Combine Geometric Energies (`ops.py:129`)
Combines base manifold and fiber space energies
```text
energy = alpha * (D_base / T_ij) + beta * D_fiber
```
### 3e. Compute Affinity Matrix (`ops.py:132`)
Normalizes energies to create mixing weights
```text
A = F.softmax(-energy, dim=-1)
```
### 4. Model Evaluation Pipeline
How trained models are evaluated on ARC-AGI benchmark with geometric validation
### 4a. Load Base Model (`eval_arc.py:52`)
Loads quantized base model for evaluation
```text
model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_id, max_seq_length = 6144, dtype = None, load_in_4bit = True)
```
### 4b. Load IGBundle Weights (`eval_arc.py:62`)
Loads trained geometric adapter weights
```text
ig_weights = os.path.join(checkpoint_path, "adapter_weights.pt")
```
### 4c. Inject Adapter Weights (`eval_arc.py:65`)
Injects IGBundle parameters into model
```text
model.load_state_dict(torch.load(ig_weights, map_location="cuda"), strict=False)
```
### 4d. Generate Model Response (`eval_arc.py:113`)
Generates solution for ARC reasoning tasks
```text
out1_tokens = model.generate(**inp1, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
```
### 4e. Statistical Validation (`eval_arc.py:160`)
Computes confidence intervals for scientific rigor
```text
lower, upper = calculate_confidence_interval(correct, total)
```
### 5. Sheaf Consistency Loss
How sheaf-theoretic constraints enforce local consistency across bundle patches
### 5a. Extract Mixture State (`losses.py:39`)
Extracts Gaussian means from bundle state
```text
m = mixture_state.m.float()
```
### 5b. Compute Patch Distances (`losses.py:56`)
Calculates distances to sheaf patch centers
```text
dists = ((m_exp - c_exp).pow(2).sum(dim=-1))
```
### 5c. Soft Patch Assignment (`losses.py:57`)
Computes soft assignments of components to patches
```text
gamma = F.softmax(-dists / self.tau, dim=-1)
```
### 5d. Aggregate Fiber Beliefs (`losses.py:67`)
Computes patch-wise aggregated fiber distributions
```text
p_bar = p_bar_num / (total_mass_r.unsqueeze(-1) + 1e-6)
```
### 5e. Compute Sheaf Loss (`losses.py:96`)
Final sheaf consistency loss using Jensen-Shannon divergence
```text
loss = (weighted_js * mask).sum().float() / total_elements
```
### 6. Autonomous Verification System
How automated crews maintain geometric integrity and thesis alignment
### 6a. Initialize Crew Manager (`auxiliary_crew.py:285`)
Spawns automated verification agents
```text
crew = CrewManager()
```
### 6b. Spawn Geometric Analysts (`auxiliary_crew.py:226`)
Creates agents focused on hyperbolic geometry validation
```text
self.agents.append(GeometricAnalyst(f"Geo-{i}", "Analyst"))
```
### 6c. Trigger Development Cycle (`auxiliary_crew.py:278`)
Initiates training and evaluation every 5 cycles
```text
await self.supervisor.run_development_phase()
```
### 6d. Execute Training (`auxiliary_crew.py:160`)
Runs training pipeline automatically
```text
subprocess.run(["python", "train.py"], check=True)
```
### 6e. Execute Evaluation (`auxiliary_crew.py:168`)
Runs ARC-AGI benchmark evaluation
```text
subprocess.run(["python", "eval_arc.py"], check=True)
```
