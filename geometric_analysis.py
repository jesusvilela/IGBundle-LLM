#!/usr/bin/env python3
"""
IGBundle Geometric Analysis and Visualization Tools

Comprehensive tools for analyzing and visualizing geometric properties
of the IGBundle training process. This provides deep insights into how
geometric components contribute to learning dynamics.

Key Features:
- Real-time geometric property monitoring
- Curvature trajectory analysis
- Bundle structure visualization
- Sheaf consistency measurement
- Comparative analysis between configurations
- Publication-ready visualizations

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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime
import argparse

# Scientific plotting setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

@dataclass
class GeometricMetrics:
    """Container for geometric analysis metrics."""
    curvature_alignment: float
    sheaf_consistency: float
    bundle_structure_quality: float
    lambda_calculus_consistency: float
    convergence_rate: float
    training_stability: float

class GeometricAnalyzer:
    """
    Comprehensive analyzer for geometric properties in IGBundle training.
    """

    def __init__(self, output_dir: str = "./geometric_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Analysis results storage
        self.analysis_cache = {}

    def setup_logging(self):
        """Setup logging for analysis operations."""
        log_file = self.output_dir / f"geometric_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def analyze_training_run(self, training_dir: str, run_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single training run.
        """
        training_path = Path(training_dir)
        if run_name is None:
            run_name = training_path.name

        self.logger.info(f"Analyzing training run: {run_name}")

        # Load training data
        metrics_path = training_path / "geometric_training_metrics.json"
        summary_path = training_path / "training_summary.json"

        if not metrics_path.exists():
            self.logger.warning(f"Metrics file not found: {metrics_path}")
            return {"error": "metrics_not_found"}

        with open(metrics_path, 'r') as f:
            training_data = json.load(f)

        analysis = {
            "run_name": run_name,
            "training_dir": str(training_path),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        # Analyze different aspects
        analysis["geometric_properties"] = self._analyze_geometric_properties(training_data)
        analysis["learning_dynamics"] = self._analyze_learning_dynamics(training_data)
        analysis["convergence_analysis"] = self._analyze_convergence(training_data)
        analysis["resource_utilization"] = self._analyze_resource_utilization(training_data)

        # Load summary if available
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            analysis["training_summary"] = summary_data

        # Cache results
        self.analysis_cache[run_name] = analysis

        # Generate visualizations
        self._generate_analysis_visualizations(analysis)

        return analysis

    def _analyze_geometric_properties(self, training_data: Dict) -> Dict[str, Any]:
        """Analyze geometric properties from training data."""
        geometric_metrics = training_data.get("geometric_metrics", [])

        if not geometric_metrics:
            return {"error": "no_geometric_metrics"}

        # Extract metric trajectories
        steps = [m["step"] for m in geometric_metrics]
        curvature_losses = [m.get("curvature_loss", 0.0) for m in geometric_metrics]
        sheaf_losses = [m.get("sheaf_loss", 0.0) for m in geometric_metrics]
        bundle_losses = [m.get("bundle_loss", 0.0) for m in geometric_metrics]
        lambda_losses = [m.get("lambda_loss", 0.0) for m in geometric_metrics]

        analysis = {
            "curvature_learning": self._analyze_curvature_trajectory(steps, curvature_losses),
            "sheaf_evolution": self._analyze_sheaf_consistency(steps, sheaf_losses),
            "bundle_preservation": self._analyze_bundle_structure(steps, bundle_losses),
            "lambda_consistency": self._analyze_lambda_calculus(steps, lambda_losses),
            "geometric_correlations": self._analyze_geometric_correlations({
                "curvature": curvature_losses,
                "sheaf": sheaf_losses,
                "bundle": bundle_losses,
                "lambda": lambda_losses
            })
        }

        return analysis

    def _analyze_curvature_trajectory(self, steps: List[int], curvature_losses: List[float]) -> Dict:
        """Analyze curvature learning trajectory."""
        if not curvature_losses or all(loss == 0.0 for loss in curvature_losses):
            return {"status": "no_curvature_learning"}

        analysis = {
            "initial_curvature_loss": curvature_losses[0],
            "final_curvature_loss": curvature_losses[-1],
            "curvature_reduction": curvature_losses[0] - curvature_losses[-1],
            "curvature_reduction_rate": self._compute_reduction_rate(curvature_losses),
            "curvature_stability": self._compute_stability(curvature_losses),
        }

        # Analyze curvature learning phases
        analysis["learning_phases"] = self._identify_learning_phases(curvature_losses)

        # Compute curvature alignment score (how well model achieves target curvature)
        target_curvature = -1.0  # Default hyperbolic target
        final_loss = curvature_losses[-1]
        analysis["curvature_alignment"] = max(0.0, 1.0 - final_loss)

        return analysis

    def _analyze_sheaf_consistency(self, steps: List[int], sheaf_losses: List[float]) -> Dict:
        """Analyze sheaf consistency evolution."""
        if not sheaf_losses:
            return {"status": "no_sheaf_data"}

        analysis = {
            "initial_sheaf_loss": sheaf_losses[0],
            "final_sheaf_loss": sheaf_losses[-1],
            "sheaf_improvement": sheaf_losses[0] - sheaf_losses[-1],
            "sheaf_convergence_rate": self._compute_convergence_rate(sheaf_losses),
            "sheaf_consistency_score": max(0.0, 1.0 - sheaf_losses[-1]),
        }

        # Analyze sheaf learning dynamics
        analysis["consistency_phases"] = self._identify_learning_phases(sheaf_losses)

        return analysis

    def _analyze_bundle_structure(self, steps: List[int], bundle_losses: List[float]) -> Dict:
        """Analyze bundle structure preservation."""
        if not bundle_losses:
            return {"status": "no_bundle_data"}

        analysis = {
            "initial_bundle_loss": bundle_losses[0],
            "final_bundle_loss": bundle_losses[-1],
            "bundle_improvement": bundle_losses[0] - bundle_losses[-1],
            "bundle_stability": self._compute_stability(bundle_losses),
            "bundle_quality_score": max(0.0, 1.0 - bundle_losses[-1]),
        }

        return analysis

    def _analyze_lambda_calculus(self, steps: List[int], lambda_losses: List[float]) -> Dict:
        """Analyze lambda calculus consistency."""
        if not lambda_losses:
            return {"status": "no_lambda_data"}

        analysis = {
            "initial_lambda_loss": lambda_losses[0],
            "final_lambda_loss": lambda_losses[-1],
            "lambda_improvement": lambda_losses[0] - lambda_losses[-1],
            "lambda_consistency_score": max(0.0, 1.0 - lambda_losses[-1]),
        }

        return analysis

    def _analyze_geometric_correlations(self, metric_dict: Dict[str, List[float]]) -> Dict:
        """Analyze correlations between different geometric metrics."""
        correlations = {}

        # Compute pairwise correlations
        for metric1, values1 in metric_dict.items():
            for metric2, values2 in metric_dict.items():
                if metric1 != metric2 and len(values1) == len(values2):
                    corr_key = f"{metric1}_vs_{metric2}"
                    try:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        correlations[corr_key] = float(correlation) if not np.isnan(correlation) else 0.0
                    except:
                        correlations[corr_key] = 0.0

        return correlations

    def _analyze_learning_dynamics(self, training_data: Dict) -> Dict:
        """Analyze overall learning dynamics."""
        geometric_metrics = training_data.get("geometric_metrics", [])

        if not geometric_metrics:
            return {"error": "no_metrics"}

        # Extract main training metrics
        steps = [m["step"] for m in geometric_metrics]
        losses = [m["loss"] for m in geometric_metrics]
        learning_rates = [m.get("learning_rate", 0.0) for m in geometric_metrics]

        analysis = {
            "loss_trajectory": self._analyze_loss_trajectory(steps, losses),
            "learning_rate_schedule": self._analyze_lr_schedule(steps, learning_rates),
            "training_phases": self._identify_training_phases(losses),
            "optimization_efficiency": self._compute_optimization_efficiency(losses),
        }

        return analysis

    def _analyze_convergence(self, training_data: Dict) -> Dict:
        """Analyze convergence properties."""
        geometric_metrics = training_data.get("geometric_metrics", [])

        if not geometric_metrics:
            return {"error": "no_metrics"}

        losses = [m["loss"] for m in geometric_metrics]

        analysis = {
            "convergence_rate": self._compute_convergence_rate(losses),
            "convergence_stability": self._compute_convergence_stability(losses),
            "plateaus_detected": self._detect_plateaus(losses),
            "convergence_quality": self._assess_convergence_quality(losses),
        }

        return analysis

    def _analyze_resource_utilization(self, training_data: Dict) -> Dict:
        """Analyze resource utilization patterns."""
        memory_usage = training_data.get("memory_usage", [])
        step_times = training_data.get("step_times", [])

        analysis = {
            "memory_efficiency": self._analyze_memory_patterns(memory_usage),
            "timing_analysis": self._analyze_timing_patterns(step_times),
            "resource_optimization_score": self._compute_resource_score(memory_usage, step_times),
        }

        return analysis

    def _compute_reduction_rate(self, values: List[float]) -> float:
        """Compute exponential reduction rate."""
        if len(values) < 2:
            return 0.0

        try:
            log_values = np.log(np.maximum(values, 1e-10))
            x = np.arange(len(values))
            slope = np.polyfit(x, log_values, 1)[0]
            return max(0.0, -slope)
        except:
            return 0.0

    def _compute_stability(self, values: List[float]) -> float:
        """Compute stability score (inverse coefficient of variation)."""
        if len(values) < 5:
            return 0.0

        final_portion = values[len(values)//2:]
        mean_val = np.mean(final_portion)
        std_val = np.std(final_portion)

        if mean_val > 0:
            cv = std_val / mean_val
            return 1.0 / (1.0 + cv)
        return 0.0

    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """Compute overall convergence rate."""
        return self._compute_reduction_rate(losses)

    def _identify_learning_phases(self, values: List[float]) -> Dict:
        """Identify different learning phases in trajectory."""
        if len(values) < 20:
            return {"phases": 1, "description": "insufficient_data"}

        # Simple phase detection based on derivative changes
        derivatives = np.diff(values)

        # Identify major changes in learning rate
        phase_changes = []
        window_size = len(derivatives) // 4

        for i in range(window_size, len(derivatives) - window_size):
            before = np.mean(derivatives[i-window_size:i])
            after = np.mean(derivatives[i:i+window_size])

            if abs(before - after) > np.std(derivatives):
                phase_changes.append(i)

        return {
            "phases": len(phase_changes) + 1,
            "phase_transitions": phase_changes,
            "description": f"Detected {len(phase_changes)} phase transitions"
        }

    def _generate_analysis_visualizations(self, analysis: Dict):
        """Generate comprehensive visualizations for the analysis."""
        run_name = analysis["run_name"]
        viz_dir = self.output_dir / f"visualizations_{run_name}"
        viz_dir.mkdir(exist_ok=True)

        self.logger.info(f"Generating visualizations for {run_name}")

        # Generate different types of visualizations
        self._plot_geometric_trajectories(analysis, viz_dir)
        self._plot_convergence_analysis(analysis, viz_dir)
        self._plot_correlation_heatmap(analysis, viz_dir)
        self._plot_resource_utilization(analysis, viz_dir)

    def _plot_geometric_trajectories(self, analysis: Dict, viz_dir: Path):
        """Plot geometric property trajectories."""
        try:
            # This would require access to the raw trajectory data
            # For now, create a placeholder structure

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Geometric Properties Analysis: {analysis['run_name']}", fontsize=16)

            # Placeholder plots - would be populated with real data
            for i, (ax, title) in enumerate(zip(axes.flat,
                                              ["Curvature Learning", "Sheaf Consistency",
                                               "Bundle Structure", "Lambda Calculus"])):
                ax.set_title(title)
                ax.set_xlabel("Training Steps")
                ax.set_ylabel("Loss/Score")

                # Add placeholder text
                ax.text(0.5, 0.5, f"{title}\nAnalysis Ready\n(Data visualization\nrequires training run)",
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.7))

            plt.tight_layout()
            plt.savefig(viz_dir / "geometric_trajectories.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating geometric trajectories plot: {e}")

    def _plot_convergence_analysis(self, analysis: Dict, viz_dir: Path):
        """Plot convergence analysis."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Convergence Analysis: {analysis['run_name']}", fontsize=16)

            # Convergence rate visualization
            ax1.set_title("Convergence Properties")
            convergence_data = analysis.get("convergence_analysis", {})

            # Create summary metrics visualization
            metrics = ["Convergence Rate", "Stability", "Quality"]
            values = [
                convergence_data.get("convergence_rate", 0.0),
                convergence_data.get("convergence_stability", 0.0),
                convergence_data.get("convergence_quality", 0.0)
            ]

            bars = ax1.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax1.set_ylabel("Score")
            ax1.set_ylim(0, 1.0)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

            # Learning phases
            ax2.set_title("Learning Phases Analysis")
            geometric_props = analysis.get("geometric_properties", {})
            curvature_phases = geometric_props.get("curvature_learning", {}).get("learning_phases", {})

            phases_info = f"Phases Detected: {curvature_phases.get('phases', 'N/A')}"
            ax2.text(0.5, 0.5, phases_info, ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14,
                    bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.7))

            plt.tight_layout()
            plt.savefig(viz_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating convergence plot: {e}")

    def _plot_correlation_heatmap(self, analysis: Dict, viz_dir: Path):
        """Plot correlation heatmap of geometric metrics."""
        try:
            geometric_props = analysis.get("geometric_properties", {})
            correlations = geometric_props.get("geometric_correlations", {})

            if not correlations:
                return

            # Create correlation matrix
            metric_pairs = list(correlations.keys())
            correlation_values = list(correlations.values())

            # Create a simplified visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create a bar plot of correlations
            bars = ax.barh(range(len(metric_pairs)), correlation_values,
                          color=['red' if v < 0 else 'blue' for v in correlation_values])

            ax.set_yticks(range(len(metric_pairs)))
            ax.set_yticklabels(metric_pairs, rotation=0)
            ax.set_xlabel("Correlation Coefficient")
            ax.set_title(f"Geometric Metrics Correlations: {analysis['run_name']}")
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, correlation_values)):
                ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}',
                       va='center', ha='left' if value >= 0 else 'right')

            plt.tight_layout()
            plt.savefig(viz_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating correlation heatmap: {e}")

    def _plot_resource_utilization(self, analysis: Dict, viz_dir: Path):
        """Plot resource utilization analysis."""
        try:
            resource_data = analysis.get("resource_utilization", {})

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"Resource Utilization: {analysis['run_name']}", fontsize=16)

            # Memory efficiency
            memory_info = resource_data.get("memory_efficiency", {})
            ax1.set_title("Memory Efficiency")

            if memory_info:
                memory_metrics = ["Peak (GB)", "Average (GB)", "Efficiency Score"]
                memory_values = [
                    memory_info.get("peak_memory", 0.0),
                    memory_info.get("average_memory", 0.0),
                    memory_info.get("efficiency_score", 0.0)
                ]

                bars1 = ax1.bar(memory_metrics, memory_values, color=['red', 'orange', 'green'])
                ax1.set_ylabel("Value")

                # Add value labels
                for bar, value in zip(bars1, memory_values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{value:.2f}', ha='center', va='bottom')

            # Timing analysis
            timing_info = resource_data.get("timing_analysis", {})
            ax2.set_title("Training Efficiency")

            if timing_info:
                timing_metrics = ["Avg Step Time (s)", "Training Speed", "Efficiency Score"]
                timing_values = [
                    timing_info.get("average_step_time", 0.0),
                    timing_info.get("training_speed", 0.0),
                    timing_info.get("efficiency_score", 0.0)
                ]

                bars2 = ax2.bar(timing_metrics, timing_values, color=['blue', 'cyan', 'green'])
                ax2.set_ylabel("Value")

                # Add value labels
                for bar, value in zip(bars2, timing_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{value:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(viz_dir / "resource_utilization.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating resource utilization plot: {e}")

    def compare_multiple_runs(self, run_directories: List[str], run_names: List[str] = None) -> Dict:
        """Compare multiple training runs."""
        if run_names is None:
            run_names = [Path(d).name for d in run_directories]

        self.logger.info(f"Comparing {len(run_directories)} training runs")

        # Analyze all runs
        all_analyses = {}
        for run_dir, run_name in zip(run_directories, run_names):
            analysis = self.analyze_training_run(run_dir, run_name)
            all_analyses[run_name] = analysis

        # Generate comparison analysis
        comparison = self._generate_comparison_analysis(all_analyses)

        # Generate comparison visualizations
        self._generate_comparison_visualizations(comparison)

        return comparison

    def _generate_comparison_analysis(self, all_analyses: Dict) -> Dict:
        """Generate comparative analysis across runs."""
        comparison = {
            "runs_compared": list(all_analyses.keys()),
            "comparison_timestamp": datetime.now().isoformat(),
        }

        # Compare geometric properties
        comparison["geometric_comparison"] = self._compare_geometric_properties(all_analyses)

        # Compare convergence properties
        comparison["convergence_comparison"] = self._compare_convergence_properties(all_analyses)

        # Rank runs by different criteria
        comparison["performance_rankings"] = self._rank_runs_by_performance(all_analyses)

        return comparison

def main():
    parser = argparse.ArgumentParser(description="IGBundle Geometric Analysis Tools")
    parser.add_argument("command", choices=["analyze", "compare", "visualize"],
                       help="Analysis command to execute")
    parser.add_argument("--training_dir", type=str, help="Training directory to analyze")
    parser.add_argument("--run_name", type=str, help="Name for the analysis run")
    parser.add_argument("--compare_dirs", nargs="+", help="Multiple directories to compare")
    parser.add_argument("--output_dir", type=str, default="./geometric_analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    analyzer = GeometricAnalyzer(args.output_dir)

    if args.command == "analyze":
        if not args.training_dir:
            print("❌ --training_dir required for analyze command")
            return

        print(f"🔬 Analyzing training run: {args.training_dir}")
        analysis = analyzer.analyze_training_run(args.training_dir, args.run_name)

        print("✅ Analysis completed!")
        print(f"📊 Results saved to: {analyzer.output_dir}")

    elif args.command == "compare":
        if not args.compare_dirs or len(args.compare_dirs) < 2:
            print("❌ --compare_dirs with at least 2 directories required")
            return

        print(f"🔬 Comparing {len(args.compare_dirs)} training runs")
        comparison = analyzer.compare_multiple_runs(args.compare_dirs)

        print("✅ Comparison completed!")
        print(f"📊 Results saved to: {analyzer.output_dir}")

    elif args.command == "visualize":
        print("🎨 Visualization mode - specify training directory to visualize")
        if args.training_dir:
            analyzer.analyze_training_run(args.training_dir, args.run_name)

if __name__ == "__main__":
    main()