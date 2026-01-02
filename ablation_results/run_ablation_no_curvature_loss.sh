#!/bin/bash
# Ablation Study: no_curvature_loss
# Research Question: How much does curvature regularization contribute to geometric learning?
# Expected Impact: high

set -e

echo "🔬 Running Ablation Study: no_curvature_loss"
echo "📋 Description: Disable curvature regularization to test Riemannian geometry impact"
echo "❓ Research Question: How much does curvature regularization contribute to geometric learning?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_no_curvature_loss.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_no_curvature_loss \
    --debug

echo "✅ Ablation study completed: no_curvature_loss"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation no_curvature_loss

echo "🎯 Ablation study no_curvature_loss completed successfully"
