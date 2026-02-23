"""
IGBundle: Information-Geometric Bundle Adapters for LLMs
"""
from .core.config import IGBundleConfig
from .modules.geometric_adapter import GeometricIGBundleAdapter, create_geometric_adapter

__all__ = [
    "IGBundleConfig",
    "GeometricIGBundleAdapter",
    "create_geometric_adapter",
]
