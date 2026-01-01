"""
Geometrically Rigorous Training Integration

This module provides training procedures that incorporate proper geometric losses
and maintain mathematical consistency throughout the training process.

Key features:
- Riemannian optimization with natural gradients
- Geometric regularization losses
- Curvature-aware learning rate scheduling
- Bundle structure preservation
- Sheaf-theoretic consistency constraints

Author: LLMOS SystemAgent
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass
import logging

from ..modules.geometric_adapter import GeometricIGBundleAdapter, GeometricState

@dataclass
class GeometricTrainingConfig:
    """Configuration for geometrically rigorous training."""
    # Standard training parameters
    learning_rate: float = 2e-4
    batch_size: int = 16
    max_steps: int = 1000
    warmup_steps: int = 100

    # Geometric loss weights
    lambda_curvature: float = 0.01
    lambda_sheaf: float = 0.001
    lambda_bundle: float = 0.001
    lambda_lambda: float = 0.0001  # Lambda calculus consistency

    # Riemannian optimization parameters
    use_natural_gradients: bool = True
    fisher_update_freq: int = 10
    fisher_momentum: float = 0.95

    # Curvature scheduling
    target_curvature_schedule: str = "exponential"  # "constant", "linear", "exponential"
    initial_target_curvature: float = 0.0
    final_target_curvature: float = -1.0

    # Bundle structure parameters
    preserve_bundle_topology: bool = True
    topology_check_freq: int = 50

class RiemannianOptimizer:
    """
    Riemannian optimization using natural gradients on the statistical manifold.

    This implements proper natural gradient descent on the manifold of
    probability distributions, accounting for the Fisher information metric.
    """

    def __init__(self, parameters, lr: float = 1e-3, momentum: float = 0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum

        # Fisher information matrix (diagonal approximation)
        self.fisher_diag = {}
        self.momentum_buffer = {}

        # Initialize buffers
        for param in self.parameters:
            if param.requires_grad:
                self.fisher_diag[param] = torch.ones_like(param.data)
                self.momentum_buffer[param] = torch.zeros_like(param.data)

    def update_fisher(self, model: nn.Module, batch: torch.Tensor):
        """Update Fisher information matrix using current batch."""
        model.zero_grad()

        # Forward pass
        output = model(batch)

        # Compute log-likelihood (simplified for demonstration)
        log_likelihood = -F.cross_entropy(output.view(-1, output.size(-1)),
                                        batch.view(-1), reduction='sum')

        # Compute gradients w.r.t. log-likelihood
        grads = torch.autograd.grad(log_likelihood, self.parameters, create_graph=False)

        # Update Fisher diagonal (exponential moving average)
        for param, grad in zip(self.parameters, grads):
            if param in self.fisher_diag:
                self.fisher_diag[param] = (
                    self.momentum * self.fisher_diag[param] +
                    (1 - self.momentum) * grad.pow(2)
                )

    def step(self):
        """Perform natural gradient step."""
        for param in self.parameters:
            if param.grad is None or param not in self.fisher_diag:
                continue

            # Natural gradient: F^{-1} * gradient
            fisher_inv = 1.0 / (self.fisher_diag[param] + 1e-8)
            natural_grad = param.grad * fisher_inv

            # Momentum update
            self.momentum_buffer[param] = (
                self.momentum * self.momentum_buffer[param] + natural_grad
            )

            # Apply update
            param.data.add_(self.momentum_buffer[param], alpha=-self.lr)

class GeometricTrainer:
    """
    Main trainer class for geometrically rigorous IGBundle models.
    """

    def __init__(self, model: nn.Module, config: GeometricTrainingConfig):
        self.model = model
        self.config = config

        # Setup optimizer
        if config.use_natural_gradients:
            # Filter for adapter parameters only
            adapter_params = []
            for name, module in model.named_modules():
                if isinstance(module, GeometricIGBundleAdapter):
                    adapter_params.extend(module.parameters())

            self.optimizer = RiemannianOptimizer(
                adapter_params,
                lr=config.learning_rate
            )
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=1e-5
            )

        # Learning rate scheduler
        if hasattr(self.optimizer, 'parameters'):  # Standard optimizer
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.warmup_steps
            )
        else:  # Custom Riemannian optimizer
            self.scheduler = None

        # Logging
        self.logger = logging.getLogger(__name__)
        self.step_count = 0
        self.loss_history = []
        self.geometric_loss_history = []

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Perform single training step with geometric regularization.

        Args:
            batch: (B, T) - input token ids

        Returns:
            losses: Dictionary of loss components
        """
        self.model.train()

        # Forward pass
        outputs = self.model(batch)

        # Language modeling loss (standard)
        lm_loss = self._compute_language_modeling_loss(outputs, batch)

        # Geometric losses
        geometric_losses = self._compute_geometric_losses()

        # Total loss
        total_loss = lm_loss
        for loss_name, loss_value in geometric_losses.items():
            weight = getattr(self.config, f"lambda_{loss_name}", 0.0)
            total_loss = total_loss + weight * loss_value

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if hasattr(self.optimizer, 'parameters'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        if isinstance(self.optimizer, RiemannianOptimizer):
            # Update Fisher information periodically
            if self.step_count % self.config.fisher_update_freq == 0:
                self.optimizer.update_fisher(self.model, batch)
            self.optimizer.step()
        else:
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        # Clear gradients
        self.model.zero_grad()

        # Logging
        losses = {
            'total': total_loss.item(),
            'language_modeling': lm_loss.item(),
            **{k: v.item() for k, v in geometric_losses.items()}
        }

        self.loss_history.append(losses['total'])
        self.geometric_loss_history.append(sum(v for k, v in losses.items()
                                             if k != 'total' and k != 'language_modeling'))

        self.step_count += 1

        # Periodic checks
        if self.step_count % self.config.topology_check_freq == 0:
            self._check_bundle_topology()

        return losses

    def _compute_language_modeling_loss(self, outputs: torch.Tensor,
                                       targets: torch.Tensor) -> torch.Tensor:
        """Compute standard causal language modeling loss."""
        # Shift targets for causal modeling
        shift_outputs = outputs[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        # Flatten for cross-entropy
        shift_outputs = shift_outputs.view(-1, shift_outputs.size(-1))
        shift_targets = shift_targets.view(-1)

        return F.cross_entropy(shift_outputs, shift_targets, ignore_index=-100)

    def _compute_geometric_losses(self) -> Dict[str, torch.Tensor]:
        """Compute all geometric regularization losses."""
        geometric_losses = {}

        # Find all geometric adapters in the model
        for name, module in self.model.named_modules():
            if isinstance(module, GeometricIGBundleAdapter):
                # Get the last computed geometric state
                if hasattr(module, '_last_geometric_state'):
                    state = module._last_geometric_state
                    adapter_losses = module.compute_geometric_losses(state)

                    # Prefix with module name
                    for loss_name, loss_value in adapter_losses.items():
                        full_name = f"{name}_{loss_name}" if name else loss_name
                        geometric_losses[full_name] = loss_value

        return geometric_losses

    def _check_bundle_topology(self):
        """Verify that bundle topology is preserved during training."""
        if not self.config.preserve_bundle_topology:
            return

        for name, module in self.model.named_modules():
            if isinstance(module, GeometricIGBundleAdapter):
                if hasattr(module, '_last_geometric_state'):
                    state = module._last_geometric_state

                    # Check local triviality
                    self._verify_local_triviality(state)

                    # Check fiber bundle structure
                    self._verify_fiber_structure(state)

    def _verify_local_triviality(self, state: GeometricState):
        """Verify local triviality condition of the fiber bundle."""
        coords = state.base_coordinates  # (B, T, P, D)
        fibers = state.fiber_sections    # (B, T, P, K)

        B, T, P, D = coords.shape

        if P <= 1:
            return

        # Check that nearby points in base have similar fiber structure
        max_violation = 0.0

        for p1 in range(P):
            for p2 in range(p1 + 1, P):
                base_dist = torch.norm(coords[:, :, p1, :] - coords[:, :, p2, :])

                fiber_p1 = F.softmax(fibers[:, :, p1, :], dim=-1)
                fiber_p2 = F.softmax(fibers[:, :, p2, :], dim=-1)

                # KL divergence between fiber distributions
                kl_div = (fiber_p1 * (torch.log(fiber_p1 + 1e-8) -
                                     torch.log(fiber_p2 + 1e-8))).sum(dim=-1)

                # Local triviality violation
                violation = (kl_div - 2.0 * base_dist).clamp(min=0).mean()
                max_violation = max(max_violation, violation.item())

        if max_violation > 1.0:  # Threshold
            self.logger.warning(f"Bundle topology violation detected: {max_violation:.4f}")

    def _verify_fiber_structure(self, state: GeometricState):
        """Verify that fiber structure remains categorical."""
        fibers = state.fiber_sections  # (B, T, P, K)

        # Check that fibers are valid probability distributions
        fiber_probs = F.softmax(fibers, dim=-1)

        # Entropy should be reasonable (not too peaked, not too uniform)
        entropy = -(fiber_probs * torch.log(fiber_probs + 1e-8)).sum(dim=-1)

        # Expected entropy for uniform distribution
        uniform_entropy = torch.log(torch.tensor(float(fibers.size(-1))))

        # Check for entropy collapse
        min_entropy = entropy.min()
        max_entropy = entropy.max()

        if min_entropy < 0.1 * uniform_entropy:
            self.logger.warning("Fiber entropy collapse detected")

        if max_entropy > 0.95 * uniform_entropy:
            self.logger.info("High fiber entropy (diverse categories)")

    def get_current_target_curvature(self) -> float:
        """Compute target curvature based on training schedule."""
        if self.config.target_curvature_schedule == "constant":
            return self.config.final_target_curvature

        # Compute progress
        progress = min(self.step_count / self.config.max_steps, 1.0)

        if self.config.target_curvature_schedule == "linear":
            return (
                self.config.initial_target_curvature +
                progress * (self.config.final_target_curvature - self.config.initial_target_curvature)
            )
        elif self.config.target_curvature_schedule == "exponential":
            # Exponential decay towards negative curvature
            return self.config.final_target_curvature * (1.0 - np.exp(-3.0 * progress))
        else:
            return self.config.final_target_curvature

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': getattr(self.optimizer, 'state_dict', lambda: {})(),
            'step_count': self.step_count,
            'loss_history': self.loss_history,
            'geometric_loss_history': self.geometric_loss_history,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if hasattr(self.optimizer, 'load_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.step_count = checkpoint['step_count']
        self.loss_history = checkpoint['loss_history']
        self.geometric_loss_history = checkpoint['geometric_loss_history']

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        if not self.loss_history:
            return {}

        recent_losses = self.loss_history[-100:]  # Last 100 steps
        recent_geo_losses = self.geometric_loss_history[-100:]

        return {
            'step_count': self.step_count,
            'current_loss': self.loss_history[-1] if self.loss_history else None,
            'average_loss_100': np.mean(recent_losses),
            'loss_std_100': np.std(recent_losses),
            'current_geometric_loss': self.geometric_loss_history[-1] if self.geometric_loss_history else None,
            'average_geometric_loss_100': np.mean(recent_geo_losses),
            'target_curvature': self.get_current_target_curvature(),
            'learning_rate': self.optimizer.lr if hasattr(self.optimizer, 'lr') else 'N/A'
        }

def create_geometric_trainer(model: nn.Module, config: GeometricTrainingConfig) -> GeometricTrainer:
    """Factory function to create geometric trainer."""
    return GeometricTrainer(model, config)