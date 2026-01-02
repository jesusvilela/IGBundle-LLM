"""
Information-Geometric Meta-Learning for IGBundle

This module implements learned information geometry that adapts Fisher information
metrics based on data patterns and task requirements, replacing fixed Fisher
metrics with adaptive information-geometric optimization.

Research Hypothesis: Learned information geometry will outperform fixed Fisher
metrics by adapting to specific data characteristics and optimization requirements.

Author: LLMOS AI Scientist Agent
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math

from torch.optim.optimizer import Optimizer


class MetaFisherNetwork(nn.Module):
    """
    Neural network that learns Fisher information matrix structure
    from data patterns and optimization history.
    """

    def __init__(self, param_dim: int, config: Any):
        super().__init__()
        self.param_dim = param_dim
        self.history_length = getattr(config, 'fisher_history_length', 10)
        self.latent_dim = getattr(config, 'fisher_latent_dim', 64)

        # Gradient history encoder
        self.gradient_encoder = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 32)
        )

        # Fisher structure predictor
        self.fisher_predictor = nn.Sequential(
            nn.Linear(32 * self.history_length + param_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, param_dim),  # Diagonal Fisher approximation
            nn.Softplus()  # Ensure positive values
        )

        # Meta-learning network for Fisher adaptation
        self.meta_adapter = nn.Sequential(
            nn.Linear(param_dim + 3, 64),  # params + [loss, grad_norm, fisher_trace]
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, param_dim)
        )

        # Gradient history buffer
        self.register_buffer('gradient_history', torch.zeros(self.history_length, 32))
        self.register_buffer('history_pointer', torch.zeros(1, dtype=torch.long))

    def forward(self, parameters: torch.Tensor, gradients: torch.Tensor,
                loss: float, optimization_stats: Dict[str, float]) -> torch.Tensor:
        """
        Compute adaptive Fisher information diagonal.

        Args:
            parameters: Current model parameters (flattened)
            gradients: Current gradients (flattened)
            loss: Current loss value
            optimization_stats: Dict with optimization statistics

        Returns:
            fisher_diagonal: Adaptive Fisher information diagonal
        """
        batch_size = parameters.shape[0] if parameters.dim() > 1 else 1

        # Encode current gradients
        grad_encoded = self.gradient_encoder(gradients.flatten())

        # Update gradient history
        current_idx = self.history_pointer.item()
        self.gradient_history[current_idx] = grad_encoded.detach()
        self.history_pointer[0] = (current_idx + 1) % self.history_length

        # Prepare features for Fisher prediction
        history_flat = self.gradient_history.flatten()  # (history_length * 32,)
        params_flat = parameters.flatten() if parameters.dim() > 1 else parameters

        fisher_input = torch.cat([history_flat, params_flat])

        # Predict base Fisher diagonal
        base_fisher = self.fisher_predictor(fisher_input)

        # Meta-adaptation based on optimization statistics
        grad_norm = optimization_stats.get('grad_norm', 1.0)
        fisher_trace = base_fisher.sum().item()

        meta_input = torch.cat([
            params_flat,
            torch.tensor([loss, grad_norm, fisher_trace], device=parameters.device)
        ])

        fisher_adaptation = self.meta_adapter(meta_input)

        # Combine base and adapted Fisher
        adaptive_fisher = base_fisher + 0.1 * fisher_adaptation

        # Ensure numerical stability
        adaptive_fisher = torch.clamp(adaptive_fisher, min=1e-8, max=1e4)

        return adaptive_fisher


class AdaptiveInformationGeometry(nn.Module):
    """
    Adaptive information geometry that learns optimal information metrics
    for different phases of training and different data patterns.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Task-specific metric learning
        self.task_encoder = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

        # Metric structure network
        self.metric_net = nn.Sequential(
            nn.Linear(32 + 3, 128),  # task_features + [progress, performance, complexity]
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.latent_dim),
            nn.Softplus()
        )

        # Hierarchical information structure
        self.info_hierarchy = nn.ModuleList([
            nn.Linear(config.latent_dim, config.latent_dim)
            for _ in range(3)  # 3 levels of hierarchy
        ])

    def forward(self, task_features: torch.Tensor, training_progress: float,
                performance_metrics: Dict[str, float]) -> torch.Tensor:
        """
        Compute adaptive information metric.

        Args:
            task_features: (B, T, D) - task-specific feature representations
            training_progress: Float [0, 1] - training completion fraction
            performance_metrics: Dict with performance statistics

        Returns:
            information_metric: (B, T, D) - adaptive information metric diagonal
        """
        B, T, D = task_features.shape

        # Encode task characteristics
        task_encoded = self.task_encoder(task_features)  # (B, T, 32)

        # Prepare context features
        performance = performance_metrics.get('recent_performance', 0.5)
        complexity = performance_metrics.get('task_complexity', 0.5)

        context = torch.tensor([training_progress, performance, complexity],
                             device=task_features.device)
        context_expanded = context.view(1, 1, 3).expand(B, T, -1)

        # Combine task and context
        combined_input = torch.cat([task_encoded, context_expanded], dim=-1)  # (B, T, 35)

        # Predict information metric
        info_metric = self.metric_net(combined_input)  # (B, T, D)

        # Apply hierarchical refinement
        for hierarchy_level in self.info_hierarchy:
            info_metric = info_metric + 0.1 * torch.tanh(hierarchy_level(info_metric))

        return info_metric


class HierarchicalNaturalGradients(Optimizer):
    """
    Multi-level natural gradient optimizer that uses different information
    metrics for different parameter groups and training phases.
    """

    def __init__(self, params, config, meta_fisher: MetaFisherNetwork,
                 adaptive_geometry: AdaptiveInformationGeometry):

        defaults = dict(
            lr=getattr(config, 'natural_grad_lr', 1e-3),
            momentum=getattr(config, 'natural_grad_momentum', 0.9),
            fisher_alpha=getattr(config, 'fisher_alpha', 0.95),
            hierarchy_levels=getattr(config, 'hierarchy_levels', 3)
        )

        super().__init__(params, defaults)

        self.meta_fisher = meta_fisher
        self.adaptive_geometry = adaptive_geometry
        self.step_count = 0

        # Initialize state for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['fisher_diagonal'] = torch.ones_like(p.data)
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['gradient_history'] = []

    def step(self, closure=None, task_features=None, performance_metrics=None):
        """
        Performs a single optimization step with hierarchical natural gradients.

        Args:
            closure: Optional closure function
            task_features: Current task feature representations
            performance_metrics: Dict with current performance metrics
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.step_count += 1
        training_progress = min(self.step_count / 10000, 1.0)  # Assume max 10k steps

        for group_idx, group in enumerate(self.param_groups):
            hierarchy_level = group_idx % self.defaults['hierarchy_levels']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data

                # Update gradient history
                if len(state['gradient_history']) >= 5:
                    state['gradient_history'].pop(0)
                state['gradient_history'].append(grad.clone())

                # Compute optimization statistics
                grad_norm = grad.norm().item()
                optimization_stats = {
                    'grad_norm': grad_norm,
                    'hierarchy_level': hierarchy_level,
                    'step_count': self.step_count
                }

                # Get adaptive Fisher information
                if self.meta_fisher is not None:
                    fisher_diag = self.meta_fisher(
                        p.data.flatten(),
                        grad.flatten(),
                        loss.item() if loss is not None else 0.0,
                        optimization_stats
                    )

                    # Reshape Fisher to match parameter shape
                    if fisher_diag.numel() == p.data.numel():
                        fisher_diag = fisher_diag.view_as(p.data)
                    else:
                        fisher_diag = torch.ones_like(p.data) * fisher_diag.mean()
                else:
                    fisher_diag = torch.ones_like(p.data)

                # Update Fisher with momentum
                alpha = group['fisher_alpha']
                state['fisher_diagonal'] = (
                    alpha * state['fisher_diagonal'] +
                    (1 - alpha) * fisher_diag
                )

                # Apply adaptive information geometry
                if (task_features is not None and
                    self.adaptive_geometry is not None and
                    p.data.dim() >= 2):  # Only for suitable parameter shapes

                    # Reshape for geometry computation
                    param_2d = p.data.view(-1, p.data.shape[-1])

                    if param_2d.shape[-1] == task_features.shape[-1]:
                        info_metric = self.adaptive_geometry(
                            task_features[:1, :1, :],  # Use first batch item
                            training_progress,
                            performance_metrics or {}
                        ).squeeze()

                        # Apply information metric modulation
                        if info_metric.numel() == param_2d.shape[-1]:
                            state['fisher_diagonal'] = (
                                state['fisher_diagonal'] *
                                info_metric.view(1, -1).expand_as(param_2d).view_as(p.data)
                            )

                # Compute natural gradient
                fisher_inv = 1.0 / (state['fisher_diagonal'] + 1e-8)
                natural_grad = grad * fisher_inv

                # Apply momentum
                momentum = group['momentum']
                if momentum != 0:
                    state['momentum_buffer'] = (
                        momentum * state['momentum_buffer'] + natural_grad
                    )
                    natural_grad = state['momentum_buffer']

                # Apply hierarchical scaling
                hierarchy_scale = 1.0 / (hierarchy_level + 1)
                scaled_grad = hierarchy_scale * natural_grad

                # Update parameters
                p.data.add_(scaled_grad, alpha=-group['lr'])

        return loss


class MetaGeometricTrainer(nn.Module):
    """
    Complete meta-geometric training system that coordinates all components.
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        # Get total parameter count for meta-Fisher network
        total_params = sum(p.numel() for p in model.parameters())
        effective_params = min(total_params, 10000)  # Limit for computational efficiency

        # Initialize components
        self.meta_fisher = MetaFisherNetwork(effective_params, config)
        self.adaptive_geometry = AdaptiveInformationGeometry(config)

        # Create hierarchical parameter groups
        self.param_groups = self._create_hierarchical_groups(model)

        # Initialize optimizer
        self.optimizer = HierarchicalNaturalGradients(
            self.param_groups, config, self.meta_fisher, self.adaptive_geometry
        )

        # Training statistics
        self.training_stats = {
            'fisher_evolution': [],
            'information_metrics': [],
            'convergence_rates': []
        }

    def _create_hierarchical_groups(self, model) -> List[Dict]:
        """Create hierarchical parameter groups for optimization."""
        groups = []

        # Group 1: Embedding and output layers (highest level)
        high_level_params = []

        # Group 2: Attention and transformation layers (middle level)
        mid_level_params = []

        # Group 3: Geometric and adapter layers (low level)
        low_level_params = []

        for name, param in model.named_parameters():
            if any(keyword in name for keyword in ['embed', 'output', 'head']):
                high_level_params.append(param)
            elif any(keyword in name for keyword in ['attention', 'transformer', 'layer']):
                mid_level_params.append(param)
            else:
                low_level_params.append(param)

        if high_level_params:
            groups.append({'params': high_level_params, 'lr': self.config.lr * 0.1})
        if mid_level_params:
            groups.append({'params': mid_level_params, 'lr': self.config.lr * 0.5})
        if low_level_params:
            groups.append({'params': low_level_params, 'lr': self.config.lr})

        return groups

    def training_step(self, batch, task_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Perform one training step with meta-geometric optimization.

        Args:
            batch: Training batch
            task_features: Optional task-specific features

        Returns:
            step_results: Dict with training step results and statistics
        """
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(batch['input_ids'])
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))

        # Backward pass
        loss.backward()

        # Prepare performance metrics
        performance_metrics = {
            'recent_performance': 1.0 / (loss.item() + 1e-8),
            'task_complexity': self._estimate_task_complexity(batch),
            'convergence_indicator': self._compute_convergence_indicator()
        }

        # Meta-geometric optimization step
        self.optimizer.step(
            task_features=task_features,
            performance_metrics=performance_metrics
        )

        # Collect statistics
        step_stats = self._collect_step_statistics(loss, performance_metrics)

        return {
            'loss': loss.item(),
            'performance_metrics': performance_metrics,
            'geometric_stats': step_stats
        }

    def _estimate_task_complexity(self, batch) -> float:
        """Estimate current task complexity."""
        # Simple heuristic based on sequence length and vocabulary diversity
        seq_length = batch['input_ids'].shape[1]
        vocab_diversity = len(torch.unique(batch['input_ids']))

        complexity = (seq_length / 512) * (vocab_diversity / 1000)
        return min(complexity, 1.0)

    def _compute_convergence_indicator(self) -> float:
        """Compute convergence indicator based on recent loss history."""
        if len(self.training_stats['convergence_rates']) < 5:
            return 0.5

        recent_rates = self.training_stats['convergence_rates'][-5:]
        convergence = 1.0 - np.std(recent_rates) / (np.mean(recent_rates) + 1e-8)

        return max(0.0, min(1.0, convergence))

    def _collect_step_statistics(self, loss: torch.Tensor,
                                performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Collect detailed statistics for analysis."""
        stats = {}

        # Fisher information evolution
        if hasattr(self.optimizer, 'param_groups'):
            fisher_stats = []
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p in self.optimizer.state:
                        fisher_diag = self.optimizer.state[p]['fisher_diagonal']
                        fisher_stats.extend([
                            fisher_diag.mean().item(),
                            fisher_diag.std().item(),
                            fisher_diag.max().item()
                        ])

            if fisher_stats:
                stats['fisher_mean'] = np.mean(fisher_stats[::3])
                stats['fisher_std'] = np.mean(fisher_stats[1::3])
                stats['fisher_max'] = np.mean(fisher_stats[2::3])

        # Update training statistics
        self.training_stats['convergence_rates'].append(loss.item())
        if len(self.training_stats['convergence_rates']) > 100:
            self.training_stats['convergence_rates'].pop(0)

        return stats


def create_meta_geometric_trainer(model, config) -> MetaGeometricTrainer:
    """Factory function to create meta-geometric trainer."""
    return MetaGeometricTrainer(model, config)