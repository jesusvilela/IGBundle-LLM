"""
Geometric operations for IGBundle.

The default Poincare affinity lives in modules.ops. The opt-in
geometry.poincare transport reference has bounded numerical checks.
Exports below are legacy prototypes, not certified geometry or calculus.
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