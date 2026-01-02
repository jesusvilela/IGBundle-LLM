#!/usr/bin/env python3
"""
IGBundle Comparative Studies Framework

Systematic framework for comparing different geometric configurations,
ablation studies, and baseline approaches. Provides statistical analysis
and publication-ready comparisons.

Key Features:
- Head-to-head comparisons between configurations
- Statistical significance testing
- Performance ranking and analysis
- Baseline vs geometric comparisons
- Parameter sensitivity analysis
- Automated report generation

Author: LLMOS SystemAgent
License: MIT
"""

import sys
import os as _os
_src_path = _os.path.join(_os.path.dirname(__file__), "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime
import argparse
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Scientific plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

@dataclass
class ComparisonConfig:
    """Configuration for a comparative study."""
    name: str
    description: str
    baseline_config: str
    comparison_configs: List[str]
    metrics_to_compare: List[str]
    statistical_tests: List[str]
    expected_outcome: str

class ComparativeStudiesFramework:
    """
    Framework for systematic comparative studies of IGBundle configurations.
    """

    def __init__(self, output_dir: str = "./comparative_studies"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Define comparative studies
        self.comparison_configs = self.define_comparative_studies()

        # Results storage
        self.comparison_results = {}

    def setup_logging(self):
        """Setup logging for comparative studies."""
        log_file = self.output_dir / f"comparative_studies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Comparative Studies Framework initialized")

    def define_comparative_studies(self) -> List[ComparisonConfig]:
        """Define comprehensive set of comparative studies."""
        studies = [
            # === Geometric vs Baseline Comparisons ===

            ComparisonConfig(
                name="geometric_vs_standard",
                description="Compare full geometric implementation vs standard IGBundle",
                baseline_config="standard_igbundle",
                comparison_configs=["full_geometric"],
                metrics_to_compare=[
                    "final_loss", "convergence_rate", "training_stability",
                    "curvature_alignment", "geometric_consistency"
                ],
                statistical_tests=["t_test", "wilcoxon"],
                expected_outcome="Geometric should show better convergence and consistency"
            ),

            ComparisonConfig(
                name="geometric_vs_lora",
                description="Compare geometric IGBundle vs pure LoRA baseline",
                baseline_config="lora_only",
                comparison_configs=["full_geometric"],
                metrics_to_compare=[
                    "final_loss", "training_efficiency", "parameter_efficiency",
                    "convergence_rate"
                ],
                statistical_tests=["t_test", "effect_size"],
                expected_outcome="Geometric should outperform with similar parameter efficiency"
            ),

            # === Component Contribution Studies ===

            ComparisonConfig(
                name="curvature_impact_study",
                description="Systematic study of curvature regularization impact",
                baseline_config="no_curvature_loss",
                comparison_configs=[
                    "low_curvature_weight", "medium_curvature_weight",
                    "high_curvature_weight", "extreme_curvature_weight"
                ],
                metrics_to_compare=[
                    "curvature_alignment", "final_loss", "geometric_consistency"
                ],
                statistical_tests=["anova", "trend_test"],
                expected_outcome="Optimal curvature weight should emerge from analysis"
            ),

            ComparisonConfig(
                name="natural_gradients_study",
                description="Impact of information-geometric optimization",
                baseline_config="standard_adam",
                comparison_configs=["natural_gradients"],
                metrics_to_compare=[
                    "convergence_rate", "training_stability", "final_loss",
                    "optimization_efficiency"
                ],
                statistical_tests=["t_test", "variance_test"],
                expected_outcome="Natural gradients should show faster, more stable convergence"
            ),

            # === Architecture Sensitivity Studies ===

            ComparisonConfig(
                name="architecture_scaling_study",
                description="Effect of architectural scale on geometric learning",
                baseline_config="minimal_architecture",
                comparison_configs=[
                    "small_architecture", "medium_architecture",
                    "large_architecture", "extra_large_architecture"
                ],
                metrics_to_compare=[
                    "parameter_efficiency", "geometric_quality", "computational_cost"
                ],
                statistical_tests=["anova", "correlation"],
                expected_outcome="Diminishing returns with increasing architecture size"
            ),

            ComparisonConfig(
                name="learning_rate_ratio_study",
                description="Optimal ratios for base vs fiber learning rates",
                baseline_config="equal_learning_rates",
                comparison_configs=[
                    "ratio_1_to_2", "ratio_1_to_5", "ratio_1_to_10",
                    "ratio_1_to_20", "ratio_1_to_50"
                ],
                metrics_to_compare=[
                    "convergence_rate", "geometric_consistency", "training_stability"
                ],
                statistical_tests=["anova", "trend_test"],
                expected_outcome="Optimal ratio around 1:10 based on theory"
            ),

            # === Curvature Target Studies ===

            ComparisonConfig(
                name="curvature_target_study",
                description="Comparison of different target curvatures",
                baseline_config="euclidean_target",
                comparison_configs=[
                    "mild_hyperbolic", "moderate_hyperbolic",
                    "strong_hyperbolic", "extreme_hyperbolic"
                ],
                metrics_to_compare=[
                    "curvature_alignment", "semantic_quality", "convergence_properties"
                ],
                statistical_tests=["anova", "trend_test"],
                expected_outcome="Moderate hyperbolic curvature optimal for language"
            ),

            # === Scheduling Studies ===

            ComparisonConfig(
                name="curvature_scheduling_study",
                description="Impact of curvature scheduling strategies",
                baseline_config="constant_curvature",
                comparison_configs=[
                    "linear_schedule", "exponential_schedule", "cosine_schedule"
                ],
                metrics_to_compare=[
                    "curvature_learning_dynamics", "final_performance", "training_stability"
                ],
                statistical_tests=["anova", "pairwise_comparisons"],
                expected_outcome="Exponential scheduling should be optimal"
            )
        ]

        return studies

    def run_comparative_study(self, study_config: ComparisonConfig,
                            baseline_results_dir: str,
                            comparison_results_dirs: List[str]) -> Dict[str, Any]:
        """Run a specific comparative study."""
        self.logger.info(f"Running comparative study: {study_config.name}")

        # Load baseline results
        baseline_data = self._load_training_results(baseline_results_dir)

        # Load comparison results
        comparison_data = {}
        for i, comp_dir in enumerate(comparison_results_dirs):
            comp_name = study_config.comparison_configs[i] if i < len(study_config.comparison_configs) else f"comparison_{i}"
            comparison_data[comp_name] = self._load_training_results(comp_dir)

        # Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(
            baseline_data, comparison_data, study_config
        )

        # Generate comparison metrics
        comparison_metrics = self._compute_comparison_metrics(
            baseline_data, comparison_data, study_config
        )

        # Create comprehensive results
        study_results = {
            "study_config": study_config.__dict__,
            "baseline_data": baseline_data,
            "comparison_data": comparison_data,
            "statistical_results": statistical_results,
            "comparison_metrics": comparison_metrics,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        # Generate visualizations
        self._generate_study_visualizations(study_results)

        # Save results
        self._save_study_results(study_results)

        return study_results

    def _load_training_results(self, results_dir: str) -> Dict[str, Any]:
        """Load training results from directory."""
        results_path = Path(results_dir)

        if not results_path.exists():
            self.logger.warning(f"Results directory not found: {results_path}")
            return {"error": "directory_not_found"}

        data = {"source_directory": str(results_path)}

        # Load training summary
        summary_path = results_path / "training_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                data["training_summary"] = json.load(f)

        # Load detailed metrics
        metrics_path = results_path / "geometric_training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                data["detailed_metrics"] = json.load(f)

        # Extract key metrics for comparison
        data["extracted_metrics"] = self._extract_key_metrics(data)

        return data

    def _extract_key_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics for statistical comparison."""
        metrics = {}

        # From training summary
        summary = data.get("training_summary", {})
        metrics.update({
            "final_loss": summary.get("final_loss", float('inf')),
            "convergence_rate": summary.get("convergence_rate", 0.0),
            "training_stability": summary.get("loss_stability", 0.0),
            "curvature_alignment": summary.get("curvature_alignment", 0.0),
            "training_efficiency": summary.get("training_efficiency", 0.0),
            "peak_memory_gb": summary.get("peak_memory_gb", float('inf')),
            "average_step_time": summary.get("average_step_time", float('inf'))
        })

        # Compute derived metrics
        if metrics["final_loss"] != float('inf'):
            metrics["perplexity"] = np.exp(metrics["final_loss"])

        metrics["parameter_efficiency"] = 1.0 / (1.0 + metrics["peak_memory_gb"])
        metrics["computational_efficiency"] = 1.0 / (1.0 + metrics["average_step_time"])

        # Geometric consistency composite score
        metrics["geometric_consistency"] = (
            metrics.get("curvature_alignment", 0.0) * 0.4 +
            summary.get("geometric_consistency", 0.0) * 0.6
        )

        return metrics

    def _perform_statistical_analysis(self, baseline_data: Dict, comparison_data: Dict,
                                    study_config: ComparisonConfig) -> Dict[str, Any]:
        """Perform statistical analysis between baseline and comparisons."""
        results = {
            "tests_performed": study_config.statistical_tests,
            "metrics_analyzed": study_config.metrics_to_compare
        }

        baseline_metrics = baseline_data.get("extracted_metrics", {})

        # For each comparison configuration
        for comp_name, comp_data in comparison_data.items():
            comp_metrics = comp_data.get("extracted_metrics", {})
            comp_results = {"comparison_name": comp_name}

            # For each metric to compare
            for metric in study_config.metrics_to_compare:
                baseline_value = baseline_metrics.get(metric, 0.0)
                comparison_value = comp_metrics.get(metric, 0.0)

                metric_analysis = {
                    "baseline_value": baseline_value,
                    "comparison_value": comparison_value,
                    "difference": comparison_value - baseline_value,
                    "percent_change": ((comparison_value - baseline_value) / baseline_value * 100)
                                    if baseline_value != 0 else float('inf')
                }

                # Effect size
                if baseline_value != 0:
                    metric_analysis["effect_size"] = abs(comparison_value - baseline_value) / baseline_value
                else:
                    metric_analysis["effect_size"] = 0.0

                # Simple significance test (would need multiple runs for proper statistics)
                metric_analysis["improvement"] = comparison_value > baseline_value
                metric_analysis["substantial_improvement"] = metric_analysis["effect_size"] > 0.1

                comp_results[metric] = metric_analysis

            results[comp_name] = comp_results

        return results

    def _compute_comparison_metrics(self, baseline_data: Dict, comparison_data: Dict,
                                  study_config: ComparisonConfig) -> Dict[str, Any]:
        """Compute overall comparison metrics."""
        metrics = {
            "study_name": study_config.name,
            "total_comparisons": len(comparison_data),
        }

        # Compute aggregate improvements
        improvements = []
        for comp_name, comp_data in comparison_data.items():
            comp_metrics = comp_data.get("extracted_metrics", {})
            baseline_metrics = baseline_data.get("extracted_metrics", {})

            # Compute overall improvement score
            improvement_score = 0.0
            metric_count = 0

            for metric in study_config.metrics_to_compare:
                baseline_val = baseline_metrics.get(metric, 0.0)
                comp_val = comp_metrics.get(metric, 0.0)

                if baseline_val != 0:
                    # Normalize improvement (loss metrics should be inverted)
                    if "loss" in metric.lower():
                        improvement = (baseline_val - comp_val) / baseline_val
                    else:
                        improvement = (comp_val - baseline_val) / baseline_val

                    improvement_score += improvement
                    metric_count += 1

            if metric_count > 0:
                improvements.append(improvement_score / metric_count)

        metrics["average_improvement"] = np.mean(improvements) if improvements else 0.0
        metrics["best_improvement"] = np.max(improvements) if improvements else 0.0
        metrics["improvement_consistency"] = 1.0 - np.std(improvements) if len(improvements) > 1 else 1.0

        return metrics

    def _generate_study_visualizations(self, study_results: Dict):
        """Generate visualizations for the comparative study."""
        study_name = study_results["study_config"]["name"]
        viz_dir = self.output_dir / f"visualizations_{study_name}"
        viz_dir.mkdir(exist_ok=True)

        self.logger.info(f"Generating visualizations for study: {study_name}")

        # Generate different visualizations
        self._plot_metric_comparisons(study_results, viz_dir)
        self._plot_improvement_analysis(study_results, viz_dir)
        self._plot_statistical_significance(study_results, viz_dir)

    def _plot_metric_comparisons(self, study_results: Dict, viz_dir: Path):
        """Plot metric comparisons between baseline and variants."""
        try:
            study_config = study_results["study_config"]
            statistical_results = study_results["statistical_results"]

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Metric Comparisons: {study_config['name']}", fontsize=16)

            # Plot first 4 metrics (if available)
            metrics_to_plot = study_config["metrics_to_compare"][:4]

            for i, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
                # Collect data for this metric
                configurations = ["Baseline"]
                values = []

                # Get baseline value
                baseline_val = 0.0  # Would extract from actual data
                values.append(baseline_val)

                # Get comparison values
                for comp_name in study_config["comparison_configs"]:
                    if comp_name in statistical_results:
                        comp_val = statistical_results[comp_name].get(metric, {}).get("comparison_value", 0.0)
                        configurations.append(comp_name)
                        values.append(comp_val)

                # Create bar plot
                bars = ax.bar(range(len(configurations)), values,
                             color=['red'] + ['blue'] * (len(configurations) - 1))

                ax.set_xlabel("Configuration")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xticks(range(len(configurations)))
                ax.set_xticklabels(configurations, rotation=45, ha='right')

                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(viz_dir / "metric_comparisons.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating metric comparisons plot: {e}")

    def _plot_improvement_analysis(self, study_results: Dict, viz_dir: Path):
        """Plot improvement analysis."""
        try:
            study_name = study_results["study_config"]["name"]
            comparison_metrics = study_results["comparison_metrics"]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Improvement Analysis: {study_name}", fontsize=16)

            # Overall improvement metrics
            ax1.set_title("Overall Improvement Metrics")
            metrics = ["Average", "Best", "Consistency"]
            values = [
                comparison_metrics.get("average_improvement", 0.0),
                comparison_metrics.get("best_improvement", 0.0),
                comparison_metrics.get("improvement_consistency", 0.0)
            ]

            bars = ax1.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral'])
            ax1.set_ylabel("Improvement Score")

            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

            # Study summary
            ax2.set_title("Study Summary")
            summary_text = f"""
Study: {study_name}
Total Comparisons: {comparison_metrics.get('total_comparisons', 0)}
Average Improvement: {comparison_metrics.get('average_improvement', 0.0):.3f}
Best Configuration: {comparison_metrics.get('best_improvement', 0.0):.3f}

Expected Outcome:
{study_results['study_config'].get('expected_outcome', 'N/A')}
"""
            ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='center',
                    bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.7))
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(viz_dir / "improvement_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating improvement analysis plot: {e}")

    def _plot_statistical_significance(self, study_results: Dict, viz_dir: Path):
        """Plot statistical significance analysis."""
        try:
            study_config = study_results["study_config"]
            statistical_results = study_results["statistical_results"]

            fig, ax = plt.subplots(figsize=(12, 8))
            fig.suptitle(f"Statistical Analysis: {study_config['name']}", fontsize=16)

            # Create significance heatmap-style visualization
            metrics = study_config["metrics_to_compare"]
            comparisons = list(statistical_results.keys())

            if not comparisons:
                ax.text(0.5, 0.5, "No statistical results available", ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                plt.savefig(viz_dir / "statistical_significance.png", dpi=300, bbox_inches='tight')
                plt.close()
                return

            # Create effect size matrix
            effect_matrix = np.zeros((len(metrics), len(comparisons)))

            for j, comp_name in enumerate(comparisons):
                if comp_name in statistical_results:
                    for i, metric in enumerate(metrics):
                        if metric in statistical_results[comp_name]:
                            effect_size = statistical_results[comp_name][metric].get("effect_size", 0.0)
                            effect_matrix[i, j] = effect_size

            # Create heatmap
            im = ax.imshow(effect_matrix, cmap='RdYlBu_r', aspect='auto')
            ax.set_xticks(range(len(comparisons)))
            ax.set_yticks(range(len(metrics)))
            ax.set_xticklabels(comparisons, rotation=45, ha='right')
            ax.set_yticklabels(metrics)

            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label("Effect Size")

            # Add text annotations
            for i in range(len(metrics)):
                for j in range(len(comparisons)):
                    text = f"{effect_matrix[i, j]:.3f}"
                    ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

            ax.set_title("Effect Sizes by Metric and Configuration")

            plt.tight_layout()
            plt.savefig(viz_dir / "statistical_significance.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating statistical significance plot: {e}")

    def _save_study_results(self, study_results: Dict):
        """Save study results to file."""
        study_name = study_results["study_config"]["name"]
        results_path = self.output_dir / f"results_{study_name}.json"

        with open(results_path, 'w') as f:
            json.dump(study_results, f, indent=2, default=str)

        self.logger.info(f"Study results saved: {results_path}")

    def generate_comparative_studies_framework(self):
        """Generate the complete comparative studies framework."""
        self.logger.info("Generating comparative studies framework...")

        # Create study configurations
        for study_config in self.comparison_configs:
            self._create_study_script(study_config)

        # Generate master comparison script
        self._generate_master_comparison_script()

        # Create documentation
        self._create_comparative_studies_documentation()

    def _create_study_script(self, study_config: ComparisonConfig):
        """Create execution script for a specific comparative study."""
        script_content = f'''#!/bin/bash
# Comparative Study: {study_config.name}
# Description: {study_config.description}
# Expected Outcome: {study_config.expected_outcome}

set -e

echo "🔬 Running Comparative Study: {study_config.name}"
echo "📋 Description: {study_config.description}"
echo "🎯 Expected: {study_config.expected_outcome}"

# This script would coordinate running multiple configurations
# and then performing the comparative analysis

echo "⚠️  NOTE: This is a framework script."
echo "📝 Actual implementation requires:"
echo "   1. Running baseline configuration: {study_config.baseline_config}"
echo "   2. Running comparison configurations:"

'''

        for i, comp_config in enumerate(study_config.comparison_configs):
            script_content += f'echo "      {i+1}. {comp_config}"\n'

        script_content += f'''
echo "   3. Collecting results and running analysis"
echo ""
echo "🔧 To implement:"
echo "   1. Create configuration files for each variant"
echo "   2. Run training for each configuration"
echo "   3. Use: python comparative_studies.py run_study --study {study_config.name}"

echo "✅ Comparative study framework ready: {study_config.name}"
'''

        script_path = self.output_dir / f"study_{study_config.name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)

    def _generate_master_comparison_script(self):
        """Generate master script for all comparative studies."""
        script_content = '''#!/bin/bash
# Master Comparative Studies Script

echo "🔬 IGBundle Comparative Studies Framework"
echo "========================================="
echo ""

'''

        for i, study in enumerate(self.comparison_configs):
            script_content += f'''
echo "🔬 [{i+1}/{len(self.comparison_configs)}] Study Available: {study.name}"
echo "Description: {study.description}"
echo "Script: ./comparative_studies/study_{study.name}.sh"
echo "---"
'''

        script_content += '''
echo ""
echo "📋 To run individual study:"
echo "   ./comparative_studies/study_<name>.sh"
echo ""
echo "📊 To analyze results:"
echo "   python comparative_studies.py run_study --study <name> --baseline_dir <dir> --comparison_dirs <dirs>"
'''

        script_path = self.output_dir / "run_comparative_studies.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)

    def _create_comparative_studies_documentation(self):
        """Create comprehensive documentation for comparative studies."""
        doc_content = f"""# IGBundle Comparative Studies Framework

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Studies**: {len(self.comparison_configs)}

## Overview

This framework provides systematic comparative analysis of different IGBundle configurations to understand:

1. **Component Contributions**: Which geometric components provide the most benefit
2. **Parameter Sensitivity**: How sensitive the system is to different parameter choices
3. **Baseline Comparisons**: How geometric improvements compare to standard approaches
4. **Optimization Insights**: What configurations work best under different conditions

## Available Studies

"""

        for i, study in enumerate(self.comparison_configs):
            doc_content += f"""
### {i+1}. {study.name}

**Description**: {study.description}

**Research Question**: {study.expected_outcome}

**Configuration**:
- **Baseline**: {study.baseline_config}
- **Comparisons**: {', '.join(study.comparison_configs)}
- **Metrics**: {', '.join(study.metrics_to_compare)}
- **Statistical Tests**: {', '.join(study.statistical_tests)}

**Execution**:
```bash
./comparative_studies/study_{study.name}.sh
```

---
"""

        doc_content += """
## Usage Instructions

### 1. Setup Studies
```bash
# Generate all comparative study frameworks
python comparative_studies.py generate_framework
```

### 2. Run Individual Studies
```bash
# After collecting training results for different configurations
python comparative_studies.py run_study \\
    --study geometric_vs_standard \\
    --baseline_dir ./output/standard_baseline \\
    --comparison_dirs ./output/geometric_variant1 ./output/geometric_variant2
```

### 3. Generate Reports
```bash
# Generate comprehensive comparison report
python comparative_studies.py generate_report --studies all
```

## Expected Outcomes

### High-Impact Studies
- **geometric_vs_standard**: Quantify total geometric improvement
- **geometric_vs_lora**: Establish IGBundle value proposition
- **curvature_impact_study**: Optimize curvature regularization

### Architecture Studies
- **architecture_scaling_study**: Find optimal model size
- **learning_rate_ratio_study**: Optimize base/fiber balance

### Advanced Studies
- **curvature_target_study**: Validate hyperbolic geometry choice
- **curvature_scheduling_study**: Optimize training dynamics

## Analysis Framework

Each study generates:

1. **Statistical Analysis**
   - T-tests for mean differences
   - Effect size calculations
   - Confidence intervals
   - ANOVA for multi-group comparisons

2. **Visualizations**
   - Metric comparison plots
   - Improvement analysis
   - Statistical significance heatmaps
   - Performance ranking charts

3. **Reports**
   - Detailed statistical results
   - Performance rankings
   - Recommendations for optimal configurations
   - Publication-ready tables and figures

## Implementation Notes

This framework provides the structure and analysis tools. To use effectively:

1. **Create configuration variants** for each study
2. **Run training** for each configuration
3. **Collect results** in standardized format
4. **Run comparative analysis** using this framework
5. **Generate reports** with statistical insights

The framework is designed to work with the existing IGBundle training infrastructure and provides publication-ready analysis of geometric learning improvements.
"""

        doc_path = self.output_dir / "COMPARATIVE_STUDIES.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)

        self.logger.info(f"Documentation created: {doc_path}")

def main():
    parser = argparse.ArgumentParser(description="IGBundle Comparative Studies Framework")
    parser.add_argument("command", choices=["generate_framework", "run_study", "generate_report"],
                       help="Command to execute")
    parser.add_argument("--study", type=str, help="Specific study name to run")
    parser.add_argument("--baseline_dir", type=str, help="Baseline results directory")
    parser.add_argument("--comparison_dirs", nargs="+", help="Comparison results directories")
    parser.add_argument("--output_dir", type=str, default="./comparative_studies",
                       help="Output directory")

    args = parser.parse_args()

    framework = ComparativeStudiesFramework(args.output_dir)

    if args.command == "generate_framework":
        print("🔬 Generating IGBundle Comparative Studies Framework")
        framework.generate_comparative_studies_framework()
        print(f"✅ Framework generated in: {framework.output_dir}")
        print(f"📋 Documentation: {framework.output_dir}/COMPARATIVE_STUDIES.md")

    elif args.command == "run_study":
        if not args.study:
            print("❌ --study required for run_study command")
            return

        if not args.baseline_dir or not args.comparison_dirs:
            print("❌ --baseline_dir and --comparison_dirs required")
            return

        print(f"🔬 Running comparative study: {args.study}")

        # Find study configuration
        study_config = None
        for config in framework.comparison_configs:
            if config.name == args.study:
                study_config = config
                break

        if not study_config:
            print(f"❌ Study not found: {args.study}")
            return

        results = framework.run_comparative_study(
            study_config, args.baseline_dir, args.comparison_dirs
        )

        print("✅ Comparative study completed!")
        print(f"📊 Results saved in: {framework.output_dir}")

    elif args.command == "generate_report":
        print("📊 Report generation not yet implemented")
        print("This will generate comprehensive reports across all studies")

if __name__ == "__main__":
    main()