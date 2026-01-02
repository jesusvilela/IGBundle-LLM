#!/bin/bash
# Master Comparative Studies Script

echo "🔬 IGBundle Comparative Studies Framework"
echo "========================================="
echo ""


echo "🔬 [1/8] Study Available: geometric_vs_standard"
echo "Description: Compare full geometric implementation vs standard IGBundle"
echo "Script: ./comparative_studies/study_geometric_vs_standard.sh"
echo "---"

echo "🔬 [2/8] Study Available: geometric_vs_lora"
echo "Description: Compare geometric IGBundle vs pure LoRA baseline"
echo "Script: ./comparative_studies/study_geometric_vs_lora.sh"
echo "---"

echo "🔬 [3/8] Study Available: curvature_impact_study"
echo "Description: Systematic study of curvature regularization impact"
echo "Script: ./comparative_studies/study_curvature_impact_study.sh"
echo "---"

echo "🔬 [4/8] Study Available: natural_gradients_study"
echo "Description: Impact of information-geometric optimization"
echo "Script: ./comparative_studies/study_natural_gradients_study.sh"
echo "---"

echo "🔬 [5/8] Study Available: architecture_scaling_study"
echo "Description: Effect of architectural scale on geometric learning"
echo "Script: ./comparative_studies/study_architecture_scaling_study.sh"
echo "---"

echo "🔬 [6/8] Study Available: learning_rate_ratio_study"
echo "Description: Optimal ratios for base vs fiber learning rates"
echo "Script: ./comparative_studies/study_learning_rate_ratio_study.sh"
echo "---"

echo "🔬 [7/8] Study Available: curvature_target_study"
echo "Description: Comparison of different target curvatures"
echo "Script: ./comparative_studies/study_curvature_target_study.sh"
echo "---"

echo "🔬 [8/8] Study Available: curvature_scheduling_study"
echo "Description: Impact of curvature scheduling strategies"
echo "Script: ./comparative_studies/study_curvature_scheduling_study.sh"
echo "---"

echo ""
echo "📋 To run individual study:"
echo "   ./comparative_studies/study_<name>.sh"
echo ""
echo "📊 To analyze results:"
echo "   python comparative_studies.py run_study --study <name> --baseline_dir <dir> --comparison_dirs <dirs>"
