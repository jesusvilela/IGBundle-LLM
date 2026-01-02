#!/bin/bash
# Ablation Study: no_lambda_calculus
# Research Question: What role does lambda calculus play in geometric semantics?
# Expected Impact: medium

set -e

echo "🔬 Running Ablation Study: no_lambda_calculus"
echo "📋 Description: Disable lambda calculus operations in fiber bundles"
echo "❓ Research Question: What role does lambda calculus play in geometric semantics?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_no_lambda_calculus.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_no_lambda_calculus \
    --debug

echo "✅ Ablation study completed: no_lambda_calculus"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation no_lambda_calculus

echo "🎯 Ablation study no_lambda_calculus completed successfully"
