#!/bin/bash
# Master Ablation Study Script
# Runs all ablation studies sequentially

set -e

echo "🔬 IGBundle Geometric Ablation Studies"
echo "======================================="


echo "🔬 [1/13] Running: no_curvature_loss"
echo "Expected Impact: high"
./ablation_studies/run_ablation_no_curvature_loss.sh

echo "✅ Completed: no_curvature_loss"
echo "---"

echo "🔬 [2/13] Running: no_natural_gradients"
echo "Expected Impact: high"
./ablation_studies/run_ablation_no_natural_gradients.sh

echo "✅ Completed: no_natural_gradients"
echo "---"

echo "🔬 [3/13] Running: no_sheaf_consistency"
echo "Expected Impact: medium"
./ablation_studies/run_ablation_no_sheaf_consistency.sh

echo "✅ Completed: no_sheaf_consistency"
echo "---"

echo "🔬 [4/13] Running: no_lambda_calculus"
echo "Expected Impact: medium"
./ablation_studies/run_ablation_no_lambda_calculus.sh

echo "✅ Completed: no_lambda_calculus"
echo "---"

echo "🔬 [5/13] Running: no_bundle_structure"
echo "Expected Impact: medium"
./ablation_studies/run_ablation_no_bundle_structure.sh

echo "✅ Completed: no_bundle_structure"
echo "---"

echo "🔬 [6/13] Running: minimal_components"
echo "Expected Impact: medium"
./ablation_studies/run_ablation_minimal_components.sh

echo "✅ Completed: minimal_components"
echo "---"

echo "🔬 [7/13] Running: large_architecture"
echo "Expected Impact: medium"
./ablation_studies/run_ablation_large_architecture.sh

echo "✅ Completed: large_architecture"
echo "---"

echo "🔬 [8/13] Running: balanced_learning_rates"
echo "Expected Impact: medium"
./ablation_studies/run_ablation_balanced_learning_rates.sh

echo "✅ Completed: balanced_learning_rates"
echo "---"

echo "🔬 [9/13] Running: high_fiber_learning"
echo "Expected Impact: low"
./ablation_studies/run_ablation_high_fiber_learning.sh

echo "✅ Completed: high_fiber_learning"
echo "---"

echo "🔬 [10/13] Running: euclidean_target"
echo "Expected Impact: high"
./ablation_studies/run_ablation_euclidean_target.sh

echo "✅ Completed: euclidean_target"
echo "---"

echo "🔬 [11/13] Running: extreme_hyperbolic"
echo "Expected Impact: medium"
./ablation_studies/run_ablation_extreme_hyperbolic.sh

echo "✅ Completed: extreme_hyperbolic"
echo "---"

echo "🔬 [12/13] Running: standard_igbundle"
echo "Expected Impact: high"
./ablation_studies/run_ablation_standard_igbundle.sh

echo "✅ Completed: standard_igbundle"
echo "---"

echo "🔬 [13/13] Running: lora_only_baseline"
echo "Expected Impact: high"
./ablation_studies/run_ablation_lora_only_baseline.sh

echo "✅ Completed: lora_only_baseline"
echo "---"

echo "📊 All ablation studies completed!"
echo "🎯 Running comprehensive analysis..."

python ablation_studies.py analyze_all

echo "✅ Ablation studies complete. Check ./ablation_studies/ for results."
