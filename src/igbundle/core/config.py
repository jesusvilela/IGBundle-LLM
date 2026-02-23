from transformers import PretrainedConfig
from typing import List, Optional

class IGBundleConfig(PretrainedConfig):
    model_type = "igbundle"
    
    def __init__(
        self,
        base_model_name_or_path: str = "unsloth/Qwen2.5-7B-Instruct",
        hidden_size: int = 4096,
        adapter_dim: int = 128,
        manifold_type: str = "riemannian",
        latent_dim: int = 32,
        num_components: int = 4,
        num_categories: int = 8,
        curvature: float = 1.0,
        version: str = "2.0.0",
        supported_modalities: List[str] = ["text"],
        rlvr_enabled: bool = False,
        use_dynamics: bool = False,
        use_geodesic_attn: bool = False,
        vision_dim: int = 1152,
        adapter_scale: float = 1.0,
        dropout: float = 0.1,
        eta_b: float = 0.01,
        eta_f: float = 0.01,
        num_attention_heads: int = 4, # Phase 7 Requirement
        **kwargs,
    ):
        """
        Configuration for IGBundle V2 Adapter.
        
        Args:
            manifold_type: 'euclidean', 'riemannian', 'finsler'
            curvature: Geometric curvature parameter (lambda)
            supported_modalities: List of modalities e.g. ["text", "vision", "audio"]
            rlvr_enabled: Whether the model is tuned with Verifiable Rewards
        """
        self.base_model_name_or_path = base_model_name_or_path
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.manifold_type = manifold_type
        self.latent_dim = latent_dim # New assignment
        self.num_components = num_components # New assignment
        self.num_categories = num_categories # New assignment
        self.curvature = curvature
        self.version = version
        self.supported_modalities = supported_modalities
        self.rlvr_enabled = rlvr_enabled
        self.use_dynamics = use_dynamics # New assignment
        self.use_geodesic_attn = use_geodesic_attn # New assignment
        self.vision_dim = vision_dim # New assignment
        self.adapter_scale = adapter_scale # New assignment
        self.dropout = dropout
        self.eta_b = eta_b
        self.eta_f = eta_f
        self.num_attention_heads = num_attention_heads
        
        super().__init__(**kwargs)

    @property
    def is_multimodal(self) -> bool:
        return len(self.supported_modalities) > 1
