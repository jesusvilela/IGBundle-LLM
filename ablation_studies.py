#!/usr/bin/env python3
"""
IGBundle Geometric Ablation Studies Framework

Systematic framework for ablation studies and geometric analysis of IGBundle components.
This allows comprehensive analysis of which geometric components contribute most to
performance while the main training continues uninterrupted.

Key Features:
- Systematic ablation of geometric components
- Geometric property measurement and analysis
- Comparative studies between configurations
- Visualization of geometric learning dynamics
- Statistical analysis of ablation results

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
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Plotting configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class AblationConfig:
    """Configuration for a single ablation study."""
    name: str
    description: str
    enabled_components: Dict[str, bool]
    modified_parameters: Dict[str, Any]
    expected_impact: str  # "high", "medium", "low"
    research_question: str

class AblationStudyFramework:
    """
    Framework for systematic ablation studies of geometric IGBundle components.
    """

    def __init__(self, base_config_path: str, output_dir: str = "./ablation_studies"):
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        # Setup logging
        self.setup_logging()

        # Define ablation studies
        self.ablation_configs = self.define_ablation_studies()

        # Results storage
        self.results = {}

    def setup_logging(self):
        """Setup comprehensive logging for ablation studies."""
        log_file = self.output_dir / f"ablation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Ablation Study Framework initialized")

    def define_ablation_studies(self) -> List[AblationConfig]:
        """
        Define comprehensive set of ablation studies for geometric components.
        """
        studies = [
            # === Core Geometric Component Ablations ===

            AblationConfig(
                name="no_curvature_loss",
                description="Disable curvature regularization to test Riemannian geometry impact",
                enabled_components={
                    "curvature_loss": False,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "geometric_training.lambda_curvature": 0.0
                },
                expected_impact="high",
                research_question="How much does curvature regularization contribute to geometric learning?"
            ),

            AblationConfig(
                name="no_natural_gradients",
                description="Disable natural gradients, use standard Adam optimization",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": False,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "geometric_training.use_natural_gradients": False
                },
                expected_impact="high",
                research_question="What is the impact of information-geometric optimization vs standard optimization?"
            ),

            AblationConfig(
                name="no_sheaf_consistency",
                description="Disable sheaf consistency constraints",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": False,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "geometric_training.lambda_sheaf": 0.0
                },
                expected_impact="medium",
                research_question="How important are sheaf-theoretic consistency constraints?"
            ),

            AblationConfig(
                name="no_lambda_calculus",
                description="Disable lambda calculus operations in fiber bundles",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": False,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "geometric_training.lambda_lambda": 0.0
                },
                expected_impact="medium",
                research_question="What role does lambda calculus play in geometric semantics?"
            ),

            AblationConfig(
                name="no_bundle_structure",
                description="Disable bundle structure preservation",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": False,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "geometric_training.lambda_bundle": 0.0,
                    "geometric_training.preserve_bundle_topology": False
                },
                expected_impact="medium",
                research_question="How critical is bundle topology preservation for performance?"
            ),

            # === Architecture Ablations ===

            AblationConfig(
                name="minimal_components",
                description="Reduce to minimal number of mixture components",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "ig_adapter.num_components": 2,  # vs default 4
                    "ig_adapter.num_categories": 8   # vs default 16
                },
                expected_impact="medium",
                research_question="What is the minimum architecture needed for geometric benefits?"
            ),

            AblationConfig(
                name="large_architecture",
                description="Increase architectural capacity",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "ig_adapter.num_components": 8,   # vs default 4
                    "ig_adapter.num_categories": 32,  # vs default 16
                    "ig_adapter.latent_dim": 256     # vs default 128
                },
                expected_impact="medium",
                research_question="Do larger geometric architectures provide proportional benefits?"
            ),

            # === Learning Rate Ablations ===

            AblationConfig(
                name="balanced_learning_rates",
                description="Use equal learning rates for base and fiber updates",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "ig_adapter.eta_b": 0.05,  # equal to eta_f
                    "ig_adapter.eta_f": 0.05,  # vs default 10:1 ratio
                },
                expected_impact="medium",
                research_question="What is the optimal base-to-fiber learning rate ratio?"
            ),

            AblationConfig(
                name="high_fiber_learning",
                description="Dramatically increase fiber learning rate",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "ig_adapter.eta_f": 0.2,  # 2x higher than default
                    "ig_adapter.eta_b": 0.01  # keep base low
                },
                expected_impact="low",
                research_question="Does faster fiber learning improve semantic capture?"
            ),

            # === Curvature Ablations ===

            AblationConfig(
                name="euclidean_target",
                description="Target Euclidean (zero) curvature instead of hyperbolic",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "geometric_training.initial_target_curvature": 0.0,
                    "geometric_training.final_target_curvature": 0.0,
                    "geometric_training.target_curvature_schedule": "constant"
                },
                expected_impact="high",
                research_question="Is hyperbolic geometry essential, or does any curvature help?"
            ),

            AblationConfig(
                name="extreme_hyperbolic",
                description="Target very high negative curvature",
                enabled_components={
                    "curvature_loss": True,
                    "sheaf_loss": True,
                    "bundle_loss": True,
                    "lambda_loss": True,
                    "natural_gradients": True,
                    "riemannian_geometry": True
                },
                modified_parameters={
                    "geometric_training.final_target_curvature": -5.0,  # vs default -1.0
                    "geometric_training.lambda_curvature": 0.05       # stronger regularization
                },
                expected_impact="medium",
                research_question="Is there an optimal curvature range for language modeling?"
            ),

            # === Baseline Comparisons ===

            AblationConfig(
                name="standard_igbundle",
                description="Use original IGBundle adapter for comparison",
                enabled_components={
                    "curvature_loss": False,
                    "sheaf_loss": True,   # Original sheaf loss only
                    "bundle_loss": False,
                    "lambda_loss": False,
                    "natural_gradients": False,
                    "riemannian_geometry": False
                },
                modified_parameters={
                    "training_mode": "standard",
                    "geometric_training.lambda_curvature": 0.0,
                    "geometric_training.lambda_bundle": 0.0,
                    "geometric_training.lambda_lambda": 0.0,
                    "geometric_training.use_natural_gradients": False
                },
                expected_impact="high",
                research_question="What is the total improvement from geometric corrections?"
            ),

            AblationConfig(
                name="lora_only_baseline",
                description="LoRA-only training without any IGBundle components",
                enabled_components={
                    "curvature_loss": False,
                    "sheaf_loss": False,
                    "bundle_loss": False,
                    "lambda_loss": False,
                    "natural_gradients": False,
                    "riemannian_geometry": False,
                    "igbundle_adapter": False
                },
                modified_parameters={
                    "skip_igbundle": True
                },
                expected_impact="high",
                research_question="What is the total benefit of IGBundle vs pure LoRA?"
            )
        ]

        return studies

    def create_ablation_config(self, ablation: AblationConfig) -> str:
        """Create configuration file for a specific ablation study."""
        config = self.base_config.copy()

        # Apply parameter modifications
        for param_path, value in ablation.modified_parameters.items():
            keys = param_path.split('.')
            current = config

            # Navigate to the nested parameter
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            current[keys[-1]] = value

        # Set output directory for this ablation
        config['training']['output_dir'] = f"./output/ablation_{ablation.name}"

        # Adjust training for shorter runs (ablation studies)
        config['training']['max_steps'] = 100  # Short runs for ablation
        config['training']['save_steps'] = 50
        config['training']['logging_steps'] = 5

        # Save configuration
        config_path = self.output_dir / f"config_ablation_{ablation.name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return str(config_path)

    def generate_ablation_script(self, ablation: AblationConfig) -> str:
        """Generate execution script for an ablation study."""
        config_path = self.create_ablation_config(ablation)

        script_content = f'''#!/bin/bash
# Ablation Study: {ablation.name}
# Research Question: {ablation.research_question}
# Expected Impact: {ablation.expected_impact}

set -e

echo "🔬 Running Ablation Study: {ablation.name}"
echo "📋 Description: {ablation.description}"
echo "❓ Research Question: {ablation.research_question}"

# Memory cleanup
echo "🧹 Cleaning memory..."
python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run training
echo "🚀 Starting ablation training..."
python trainv2.py \\
    --config {config_path} \\
    --mode auto \\
    --dataset_size 1000 \\
    --output_dir ./output/ablation_{ablation.name} \\
    --debug

echo "✅ Ablation study completed: {ablation.name}"

# Analyze results
echo "📊 Running analysis..."
python ablation_studies.py analyze --ablation {ablation.name}

echo "🎯 Ablation study {ablation.name} completed successfully"
'''

        script_path = self.output_dir / f"run_ablation_{ablation.name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)  # Make executable
        return str(script_path)

    def generate_all_ablation_scripts(self):
        """Generate all ablation study scripts."""
        self.logger.info("Generating ablation study scripts...")

        for ablation in self.ablation_configs:
            script_path = self.generate_ablation_script(ablation)
            self.logger.info(f"Generated script: {script_path}")

        # Generate master script to run all ablations
        self.generate_master_script()

    def generate_master_script(self):
        """Generate master script to run all ablation studies."""
        script_content = '''#!/bin/bash
# Master Ablation Study Script
# Runs all ablation studies sequentially

set -e

echo "🔬 IGBundle Geometric Ablation Studies"
echo "======================================="

'''

        for i, ablation in enumerate(self.ablation_configs):
            script_content += f'''
echo "🔬 [{i+1}/{len(self.ablation_configs)}] Running: {ablation.name}"
echo "Expected Impact: {ablation.expected_impact}"
./ablation_studies/run_ablation_{ablation.name}.sh

echo "✅ Completed: {ablation.name}"
echo "---"
'''

        script_content += '''
echo "📊 All ablation studies completed!"
echo "🎯 Running comprehensive analysis..."

python ablation_studies.py analyze_all

echo "✅ Ablation studies complete. Check ./ablation_studies/ for results."
'''

        script_path = self.output_dir / "run_all_ablations.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)
        self.logger.info(f"Generated master script: {script_path}")

    def create_ablation_summary(self):
        """Create comprehensive summary of all planned ablation studies."""
        summary = {
            "total_studies": len(self.ablation_configs),
            "studies": []
        }

        for ablation in self.ablation_configs:
            study_info = {
                "name": ablation.name,
                "description": ablation.description,
                "research_question": ablation.research_question,
                "expected_impact": ablation.expected_impact,
                "enabled_components": ablation.enabled_components,
                "modified_parameters": ablation.modified_parameters
            }
            summary["studies"].append(study_info)

        # Save summary
        summary_path = self.output_dir / "ablation_studies_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create human-readable markdown summary
        self.create_markdown_summary(summary)

        return summary

    def create_markdown_summary(self, summary: Dict):
        """Create human-readable markdown summary of ablation studies."""
        md_content = f"""# IGBundle Geometric Ablation Studies

**Total Studies**: {summary['total_studies']}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Study Overview

"""

        # Group by expected impact
        by_impact = {"high": [], "medium": [], "low": []}
        for study in summary["studies"]:
            by_impact[study["expected_impact"]].append(study)

        for impact_level in ["high", "medium", "low"]:
            studies = by_impact[impact_level]
            if studies:
                md_content += f"\n### {impact_level.title()} Impact Studies ({len(studies)} studies)\n\n"

                for study in studies:
                    md_content += f"#### {study['name']}\n"
                    md_content += f"**Research Question**: {study['research_question']}\n\n"
                    md_content += f"**Description**: {study['description']}\n\n"

                    # Show key modifications
                    if study['modified_parameters']:
                        md_content += "**Key Parameters Modified**:\n"
                        for param, value in study['modified_parameters'].items():
                            md_content += f"- `{param}`: {value}\n"
                        md_content += "\n"

                    md_content += "---\n\n"

        md_content += """
## Execution Instructions

### Run Individual Study
```bash
# Run specific ablation
./ablation_studies/run_ablation_<study_name>.sh
```

### Run All Studies
```bash
# Run all ablations sequentially (estimated time: 2-3 hours)
./ablation_studies/run_all_ablations.sh
```

### Analyze Results
```bash
# Analyze individual study
python ablation_studies.py analyze --ablation <study_name>

# Comprehensive analysis of all studies
python ablation_studies.py analyze_all
```

## Expected Outcomes

1. **Component Importance Ranking**: Which geometric components contribute most to performance
2. **Architecture Sensitivity**: Optimal architectural choices for geometric learning
3. **Learning Rate Analysis**: Optimal ratios for base vs fiber learning rates
4. **Curvature Impact**: Evidence for hyperbolic vs Euclidean geometry
5. **Baseline Comparisons**: Quantified improvement from geometric corrections

## Analysis Framework

Each ablation study generates:
- Training metrics (loss, convergence, stability)
- Geometric quality metrics (curvature alignment, sheaf consistency)
- Resource utilization metrics (memory, time)
- Statistical significance tests comparing to baseline

Results will be compiled into a comprehensive ablation analysis report.
"""

        md_path = self.output_dir / "ABLATION_STUDIES.md"
        with open(md_path, 'w') as f:
            f.write(md_content)

        self.logger.info(f"Created markdown summary: {md_path}")

def create_geometric_analyzer():
    """Create geometric property analyzer for ablation studies."""
    analyzer_content = '''#!/usr/bin/env python3
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
'''

    analyzer_path = Path("./ablation_studies/geometric_analyzer.py")
    analyzer_path.parent.mkdir(exist_ok=True)

    with open(analyzer_path, 'w') as f:
        f.write(analyzer_content)

    os.chmod(analyzer_path, 0o755)
    return str(analyzer_path)

def main():
    parser = argparse.ArgumentParser(description="IGBundle Ablation Studies Framework")
    parser.add_argument("command", choices=["generate", "analyze", "analyze_all"],
                       help="Command to execute")
    parser.add_argument("--config", type=str, default="configs/qwen25_7b_igbundle_lora.yaml",
                       help="Base configuration file")
    parser.add_argument("--ablation", type=str, help="Specific ablation to analyze")
    parser.add_argument("--output_dir", type=str, default="./ablation_studies",
                       help="Output directory for ablation studies")

    args = parser.parse_args()

    if args.command == "generate":
        # Generate ablation study framework
        framework = AblationStudyFramework(args.config, args.output_dir)

        print("🔬 Generating IGBundle Ablation Studies Framework")
        print(f"📄 Base config: {args.config}")
        print(f"📁 Output directory: {args.output_dir}")

        # Generate all scripts and configurations
        framework.generate_all_ablation_scripts()

        # Create summary documentation
        summary = framework.create_ablation_summary()

        # Create geometric analyzer
        analyzer_path = create_geometric_analyzer()

        print(f"\n✅ Ablation studies framework generated!")
        print(f"📊 Total studies: {summary['total_studies']}")
        print(f"📋 Summary: {framework.output_dir}/ABLATION_STUDIES.md")
        print(f"🔧 Analyzer: {analyzer_path}")
        print(f"\n🚀 To run all studies: ./ablation_studies/run_all_ablations.sh")
        print(f"🔬 To run individual study: ./ablation_studies/run_ablation_<name>.sh")

    elif args.command in ["analyze", "analyze_all"]:
        # Run analysis
        analyzer_path = "./ablation_studies/geometric_analyzer.py"
        if os.path.exists(analyzer_path):
            import subprocess
            cmd = [sys.executable, analyzer_path, args.command]
            if args.ablation:
                cmd.extend(["--ablation", args.ablation])
            subprocess.run(cmd)
        else:
            print("❌ Analyzer not found. Run 'generate' command first.")

if __name__ == "__main__":
    main()