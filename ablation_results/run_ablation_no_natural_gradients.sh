#!/bin/bash
# Ablation Study: no_natural_gradients
# Research Question: What is the impact of information-geometric optimization vs standard optimization?
# Expected Impact: high

set -e

echo "🔬 Running Ablation Study: no_natural_gradients"
echo "📋 Description: Disable natural gradients, use standard Adam optimization"
echo "❓ Research Question: What is the impact of information-geometric optimization vs standard optimization?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_no_natural_gradients.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_no_natural_gradients \
    --debug

echo "✅ Ablation study completed: no_natural_gradients"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation no_natural_gradients

echo "🎯 Ablation study no_natural_gradients completed successfully"
