"""
Geometric training procedures for IGBundle.

This package provides training methods that incorporate proper
geometric losses and Riemannian optimization.
"""

from .geometric_training import (
    GeometricTrainer,
    GeometricTrainingConfig,
    RiemannianOptimizer,
    create_geometric_trainer
)

__all__ = [
    'GeometricTrainer',
    'GeometricTrainingConfig',
    'RiemannianOptimizer',
    'create_geometric_trainer'
]