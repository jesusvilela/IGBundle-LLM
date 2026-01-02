#!/bin/bash
# Ablation Study: high_fiber_learning
# Research Question: Does faster fiber learning improve semantic capture?
# Expected Impact: low

set -e

echo "🔬 Running Ablation Study: high_fiber_learning"
echo "📋 Description: Dramatically increase fiber learning rate"
echo "❓ Research Question: Does faster fiber learning improve semantic capture?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_high_fiber_learning.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_high_fiber_learning \
    --debug

echo "✅ Ablation study completed: high_fiber_learning"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation high_fiber_learning

echo "🎯 Ablation study high_fiber_learning completed successfully"
