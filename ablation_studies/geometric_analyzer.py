#!/usr/bin/env python3
"""
Geometric Property Analyzer for IGBundle Ablation Studies

Analyzes geometric properties from training outputs to understand
which components contribute to geometric learning effectiveness.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import torch

class GeometricPropertyAnalyzer:
    """Analyzes geometric properties from training outputs."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def analyze_ablation_results(self, ablation_name: str) -> Dict:
        """Analyze results from a specific ablation study."""
        results_dir = Path(f"./output/ablation_{ablation_name}")

        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return {}

        # Load training metrics
        metrics_path = results_dir / "geometric_training_metrics.json"
        summary_path = results_dir / "training_summary.json"

        analysis = {"ablation_name": ablation_name}

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            analysis["detailed_metrics"] = self._analyze_training_trajectory(metrics_data)

        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            analysis["summary_metrics"] = summary_data

        return analysis

    def _analyze_training_trajectory(self, metrics_data: Dict) -> Dict:
        """Analyze the training trajectory for geometric properties."""
        history = metrics_data.get("geometric_metrics", [])

        if not history:
            return {}

        # Extract key metrics over time
        steps = [m["step"] for m in history]
        losses = [m["loss"] for m in history]
        curvature_losses = [m["curvature_loss"] for m in history]
        sheaf_losses = [m["sheaf_loss"] for m in history]

        analysis = {
            "convergence_rate": self._compute_convergence_rate(losses),
            "curvature_learning": self._analyze_curvature_learning(curvature_losses),
            "geometric_consistency": self._analyze_geometric_consistency(history),
            "training_stability": self._analyze_stability(losses),
        }

        return analysis

    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """Compute convergence rate from loss trajectory."""
        if len(losses) < 10:
            return 0.0

        # Exponential fit
        x = np.arange(len(losses))
        log_losses = np.log(np.maximum(losses, 1e-10))
        try:
            slope = np.polyfit(x, log_losses, 1)[0]
            return max(0.0, -slope)
        except:
            return 0.0

    def _analyze_curvature_learning(self, curvature_losses: List[float]) -> Dict:
        """Analyze curvature learning dynamics."""
        if not curvature_losses:
            return {"status": "no_curvature_data"}

        return {
            "initial_curvature_loss": curvature_losses[0],
            "final_curvature_loss": curvature_losses[-1],
            "curvature_improvement": curvature_losses[0] - curvature_losses[-1],
            "curvature_stability": 1.0 / (1.0 + np.std(curvature_losses[-10:])),
        }

    def _analyze_geometric_consistency(self, history: List[Dict]) -> Dict:
        """Analyze geometric consistency across training."""
        if not history:
            return {}

        final_metrics = history[-1]
        return {
            "final_sheaf_loss": final_metrics.get("sheaf_loss", 0.0),
            "final_bundle_loss": final_metrics.get("bundle_loss", 0.0),
            "final_lambda_loss": final_metrics.get("lambda_loss", 0.0),
            "geometric_total": (
                final_metrics.get("sheaf_loss", 0.0) +
                final_metrics.get("bundle_loss", 0.0) +
                final_metrics.get("lambda_loss", 0.0)
            )
        }

    def _analyze_stability(self, losses: List[float]) -> Dict:
        """Analyze training stability."""
        if len(losses) < 10:
            return {"status": "insufficient_data"}

        final_third = losses[len(losses)//3*2:]
        return {
            "final_loss_std": np.std(final_third),
            "final_loss_mean": np.mean(final_third),
            "coefficient_of_variation": np.std(final_third) / np.mean(final_third) if np.mean(final_third) > 0 else float('inf'),
            "stability_score": 1.0 / (1.0 + np.std(final_third) / np.mean(final_third)) if np.mean(final_third) > 0 else 0.0
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["analyze", "analyze_all"])
    parser.add_argument("--ablation", type=str, help="Specific ablation to analyze")

    args = parser.parse_args()

    analyzer = GeometricPropertyAnalyzer("./ablation_studies")

    if args.command == "analyze":
        if not args.ablation:
            print("Error: --ablation required for analyze command")
            exit(1)

        results = analyzer.analyze_ablation_results(args.ablation)
        print(f"Analysis completed for {args.ablation}")
        print(json.dumps(results, indent=2))

    elif args.command == "analyze_all":
        print("Comprehensive analysis not yet implemented")
        print("This will analyze all ablation studies and generate comparison reports")
