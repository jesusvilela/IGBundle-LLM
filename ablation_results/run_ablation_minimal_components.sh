#!/bin/bash
# Ablation Study: minimal_components
# Research Question: What is the minimum architecture needed for geometric benefits?
# Expected Impact: medium

set -e

echo "🔬 Running Ablation Study: minimal_components"
echo "📋 Description: Reduce to minimal number of mixture components"
echo "❓ Research Question: What is the minimum architecture needed for geometric benefits?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_minimal_components.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_minimal_components \
    --debug

echo "✅ Ablation study completed: minimal_components"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation minimal_components

echo "🎯 Ablation study minimal_components completed successfully"
