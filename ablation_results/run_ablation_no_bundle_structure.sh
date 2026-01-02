#!/bin/bash
# Ablation Study: no_bundle_structure
# Research Question: How critical is bundle topology preservation for performance?
# Expected Impact: medium

set -e

echo "🔬 Running Ablation Study: no_bundle_structure"
echo "📋 Description: Disable bundle structure preservation"
echo "❓ Research Question: How critical is bundle topology preservation for performance?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_no_bundle_structure.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_no_bundle_structure \
    --debug

echo "✅ Ablation study completed: no_bundle_structure"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation no_bundle_structure

echo "🎯 Ablation study no_bundle_structure completed successfully"
