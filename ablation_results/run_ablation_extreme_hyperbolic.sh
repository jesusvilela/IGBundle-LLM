#!/bin/bash
# Ablation Study: extreme_hyperbolic
# Research Question: Is there an optimal curvature range for language modeling?
# Expected Impact: medium

set -e

echo "🔬 Running Ablation Study: extreme_hyperbolic"
echo "📋 Description: Target very high negative curvature"
echo "❓ Research Question: Is there an optimal curvature range for language modeling?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_extreme_hyperbolic.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_extreme_hyperbolic \
    --debug

echo "✅ Ablation study completed: extreme_hyperbolic"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation extreme_hyperbolic

echo "🎯 Ablation study extreme_hyperbolic completed successfully"
