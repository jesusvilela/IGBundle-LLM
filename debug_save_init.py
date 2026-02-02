import os
import sys
import torch

sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter

def save_init():
    print("Initializing Dummy Adapter...")
    config = IGBundleConfig(
        hidden_size=3584, # Qwen 7B
        num_components=8,
        latent_dim=64,
        num_categories=16,
        use_dynamics=True,
        use_geodesic_attn=True,
        supported_modalities=["vision", "text"]
    )
    adapter = create_geometric_adapter(config)
    
    save_dir = "igbundle_unified_training/debug_init"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "adapter_weights.pt")
    
    torch.save(adapter.state_dict(), save_path)
    print(f"Saved dummy weights to {save_path}")

if __name__ == "__main__":
    save_init()
