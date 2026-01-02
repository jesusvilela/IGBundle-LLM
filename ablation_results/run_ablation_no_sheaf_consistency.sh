#!/bin/bash
# Ablation Study: no_sheaf_consistency
# Research Question: How important are sheaf-theoretic consistency constraints?
# Expected Impact: medium

set -e

echo "🔬 Running Ablation Study: no_sheaf_consistency"
echo "📋 Description: Disable sheaf consistency constraints"
echo "❓ Research Question: How important are sheaf-theoretic consistency constraints?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_no_sheaf_consistency.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_no_sheaf_consistency \
    --debug

echo "✅ Ablation study completed: no_sheaf_consistency"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation no_sheaf_consistency

echo "🎯 Ablation study no_sheaf_consistency completed successfully"
