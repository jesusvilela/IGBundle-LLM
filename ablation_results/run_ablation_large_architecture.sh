#!/bin/bash
# Ablation Study: large_architecture
# Research Question: Do larger geometric architectures provide proportional benefits?
# Expected Impact: medium

set -e

echo "🔬 Running Ablation Study: large_architecture"
echo "📋 Description: Increase architectural capacity"
echo "❓ Research Question: Do larger geometric architectures provide proportional benefits?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_large_architecture.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_large_architecture \
    --debug

echo "✅ Ablation study completed: large_architecture"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation large_architecture

echo "🎯 Ablation study large_architecture completed successfully"
