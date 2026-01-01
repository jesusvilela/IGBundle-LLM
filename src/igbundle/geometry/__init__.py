"""
Geometric operations for IGBundle.

This package provides mathematically rigorous implementations of:
- Riemannian manifold geometry
- Fiber bundle operations
- Lambda calculus in bundle context
- Information geometry
"""

from .riemannian import (
    RiemannianGeometry,
    FiberBundleLambdaCalculus,
    RiemannianMetric,
    bundle_curvature_loss
)

__all__ = [
    'RiemannianGeometry',
    'FiberBundleLambdaCalculus',
    'RiemannianMetric',
    'bundle_curvature_loss'
]