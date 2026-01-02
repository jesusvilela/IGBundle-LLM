# IGBundle Geometric Ablation Studies

**Total Studies**: 13
**Generated**: 2026-01-02 23:11:05

## Study Overview


### High Impact Studies (5 studies)

#### no_curvature_loss
**Research Question**: How much does curvature regularization contribute to geometric learning?

**Description**: Disable curvature regularization to test Riemannian geometry impact

**Key Parameters Modified**:
- `geometric_training.lambda_curvature`: 0.0

---

#### no_natural_gradients
**Research Question**: What is the impact of information-geometric optimization vs standard optimization?

**Description**: Disable natural gradients, use standard Adam optimization

**Key Parameters Modified**:
- `geometric_training.use_natural_gradients`: False

---

#### euclidean_target
**Research Question**: Is hyperbolic geometry essential, or does any curvature help?

**Description**: Target Euclidean (zero) curvature instead of hyperbolic

**Key Parameters Modified**:
- `geometric_training.initial_target_curvature`: 0.0
- `geometric_training.final_target_curvature`: 0.0
- `geometric_training.target_curvature_schedule`: constant

---

#### standard_igbundle
**Research Question**: What is the total improvement from geometric corrections?

**Description**: Use original IGBundle adapter for comparison

**Key Parameters Modified**:
- `training_mode`: standard
- `geometric_training.lambda_curvature`: 0.0
- `geometric_training.lambda_bundle`: 0.0
- `geometric_training.lambda_lambda`: 0.0
- `geometric_training.use_natural_gradients`: False

---

#### lora_only_baseline
**Research Question**: What is the total benefit of IGBundle vs pure LoRA?

**Description**: LoRA-only training without any IGBundle components

**Key Parameters Modified**:
- `skip_igbundle`: True

---


### Medium Impact Studies (7 studies)

#### no_sheaf_consistency
**Research Question**: How important are sheaf-theoretic consistency constraints?

**Description**: Disable sheaf consistency constraints

**Key Parameters Modified**:
- `geometric_training.lambda_sheaf`: 0.0

---

#### no_lambda_calculus
**Research Question**: What role does lambda calculus play in geometric semantics?

**Description**: Disable lambda calculus operations in fiber bundles

**Key Parameters Modified**:
- `geometric_training.lambda_lambda`: 0.0

---

#### no_bundle_structure
**Research Question**: How critical is bundle topology preservation for performance?

**Description**: Disable bundle structure preservation

**Key Parameters Modified**:
- `geometric_training.lambda_bundle`: 0.0
- `geometric_training.preserve_bundle_topology`: False

---

#### minimal_components
**Research Question**: What is the minimum architecture needed for geometric benefits?

**Description**: Reduce to minimal number of mixture components

**Key Parameters Modified**:
- `ig_adapter.num_components`: 2
- `ig_adapter.num_categories`: 8

---

#### large_architecture
**Research Question**: Do larger geometric architectures provide proportional benefits?

**Description**: Increase architectural capacity

**Key Parameters Modified**:
- `ig_adapter.num_components`: 8
- `ig_adapter.num_categories`: 32
- `ig_adapter.latent_dim`: 256

---

#### balanced_learning_rates
**Research Question**: What is the optimal base-to-fiber learning rate ratio?

**Description**: Use equal learning rates for base and fiber updates

**Key Parameters Modified**:
- `ig_adapter.eta_b`: 0.05
- `ig_adapter.eta_f`: 0.05

---

#### extreme_hyperbolic
**Research Question**: Is there an optimal curvature range for language modeling?

**Description**: Target very high negative curvature

**Key Parameters Modified**:
- `geometric_training.final_target_curvature`: -5.0
- `geometric_training.lambda_curvature`: 0.05

---


### Low Impact Studies (1 studies)

#### high_fiber_learning
**Research Question**: Does faster fiber learning improve semantic capture?

**Description**: Dramatically increase fiber learning rate

**Key Parameters Modified**:
- `ig_adapter.eta_f`: 0.2
- `ig_adapter.eta_b`: 0.01

---


## Execution Instructions

### Run Individual Study
```bash
# Run specific ablation
./ablation_studies/run_ablation_<study_name>.sh
```

### Run All Studies
```bash
# Run all ablations sequentially (estimated time: 2-3 hours)
./ablation_studies/run_all_ablations.sh
```

### Analyze Results
```bash
# Analyze individual study
python ablation_studies.py analyze --ablation <study_name>

# Comprehensive analysis of all studies
python ablation_studies.py analyze_all
```

## Expected Outcomes

1. **Component Importance Ranking**: Which geometric components contribute most to performance
2. **Architecture Sensitivity**: Optimal architectural choices for geometric learning
3. **Learning Rate Analysis**: Optimal ratios for base vs fiber learning rates
4. **Curvature Impact**: Evidence for hyperbolic vs Euclidean geometry
5. **Baseline Comparisons**: Quantified improvement from geometric corrections

## Analysis Framework

Each ablation study generates:
- Training metrics (loss, convergence, stability)
- Geometric quality metrics (curvature alignment, sheaf consistency)
- Resource utilization metrics (memory, time)
- Statistical significance tests comparing to baseline

Results will be compiled into a comprehensive ablation analysis report.
