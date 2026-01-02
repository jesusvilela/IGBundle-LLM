#!/bin/bash
# Ablation Study: euclidean_target
# Research Question: Is hyperbolic geometry essential, or does any curvature help?
# Expected Impact: high

set -e

echo "🔬 Running Ablation Study: euclidean_target"
echo "📋 Description: Target Euclidean (zero) curvature instead of hyperbolic"
echo "❓ Research Question: Is hyperbolic geometry essential, or does any curvature help?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_euclidean_target.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_euclidean_target \
    --debug

echo "✅ Ablation study completed: euclidean_target"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation euclidean_target

echo "🎯 Ablation study euclidean_target completed successfully"
