#!/bin/bash
# Ablation Study: lora_only_baseline
# Research Question: What is the total benefit of IGBundle vs pure LoRA?
# Expected Impact: high

set -e

echo "🔬 Running Ablation Study: lora_only_baseline"
echo "📋 Description: LoRA-only training without any IGBundle components"
echo "❓ Research Question: What is the total benefit of IGBundle vs pure LoRA?"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \
    --config ablation_results/config_ablation_lora_only_baseline.yaml \
    --mode auto \
    --dataset_size 1000 \
    --output_dir ./output/ablation_lora_only_baseline \
    --debug

echo "✅ Ablation study completed: lora_only_baseline"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation lora_only_baseline

echo "🎯 Ablation study lora_only_baseline completed successfully"
