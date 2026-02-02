
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from igbundle.modules.geometric_adapter import create_geometric_adapter, GeometricIGBundleAdapter
from igbundle.core.config import IGBundleConfig

def test_dynamics():
    print("Initializing Adapter...", flush=True)
    config = IGBundleConfig(
        hidden_size=3584, # Qwen 7B size
        num_components=4,
        latent_dim=16,
        num_categories=8,
        use_dynamics=True,
        supported_modalities=["text"]
    )
    
    adapter = create_geometric_adapter(config).to("cuda")
    adapter.eval() # Inference mode
    print("Adapter initialized.", flush=True)
    
    B, T, D = 1, 10, 3584
    x = torch.randn(B, T, D, device="cuda")
    
    print("Running Forward Pass with use_dynamics=True...", flush=True)
    try:
        # Enable grad context even though we are in inference, 
        # to match app scenario (app uses no_grad usually)
        with torch.no_grad():
            out, state = adapter(x)
            
        print("Forward Pass Successful!")
        print("Output shape:", out.shape)
        print("State keys:", state.__dict__.keys())
        
    except Exception as e:
        print(f"CRASH DETECTED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dynamics()
