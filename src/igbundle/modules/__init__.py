"""Core IGBundle module exports."""

from .attention import PoincareAttention
from .compiler import ManifoldDecompiler
from .consensus import SheafConsensus
from .delta_fiber import DeltaFiberUpdate, DeltaNetAttention
from .geometric_adapter import GeometricIGBundleAdapter, GeometricState, create_geometric_adapter
from .hybrid_gating import EntropyGating, HybridAttentionHead
from .memory import GeometricPhaseMemory
from .projectors import GeodesicVisionProjector, LinearProjector, MLPProjector, ProjectorBase
from .regularization import LipschitzPenalty, spectral_normalize_module
from .riemannian_attention import RiemannianAttention
from .spectral import SpectrallyNormalizedLinear
from .vision import VisionProjector

__all__ = [
	"DeltaFiberUpdate",
	"DeltaNetAttention",
	"EntropyGating",
	"GeodesicVisionProjector",
	"GeometricIGBundleAdapter",
	"GeometricPhaseMemory",
	"GeometricState",
	"HybridAttentionHead",
	"LinearProjector",
	"LipschitzPenalty",
	"MLPProjector",
	"ManifoldDecompiler",
	"PoincareAttention",
	"ProjectorBase",
	"RiemannianAttention",
	"SheafConsensus",
	"SpectrallyNormalizedLinear",
	"VisionProjector",
	"create_geometric_adapter",
	"spectral_normalize_module",
]
