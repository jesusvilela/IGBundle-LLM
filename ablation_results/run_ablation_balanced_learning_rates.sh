#!/bin/bash
# Ablation Study: balanced_learning_rates
# Research Question: What is the optimal base-to-fiber learning rate ratio?
# Expected Impact: medium

set -e

echo "🔬 Running Ablation Study: balanced_learning_rates"
echo "📋 Description: Use equal learning rates for base and fiber updates"
echo "❓ Research Question: What is the optimal base-to-fiber learning rate ratio?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_balanced_learning_rates.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_balanced_learning_rates \
    --debug

echo "✅ Ablation study completed: balanced_learning_rates"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation balanced_learning_rates

echo "🎯 Ablation study balanced_learning_rates completed successfully"
