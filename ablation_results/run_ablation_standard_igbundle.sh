#!/bin/bash
# Ablation Study: standard_igbundle
# Research Question: What is the total improvement from geometric corrections?
# Expected Impact: high

set -e

echo "🔬 Running Ablation Study: standard_igbundle"
echo "📋 Description: Use original IGBundle adapter for comparison"
echo "❓ Research Question: What is the total improvement from geometric corrections?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_standard_igbundle.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_standard_igbundle \
    --debug

echo "✅ Ablation study completed: standard_igbundle"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation standard_igbundle

echo "🎯 Ablation study standard_igbundle completed successfully"
