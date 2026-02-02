
import torch
from types import SimpleNamespace
from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter

def test_kan_inference():
    print("Testing KAN Geometry Integration...")
    
    # 1. Configuration
    config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=8,
        vocab_size=1000,
        manifold_type="kan", # Activate KAN
        use_dynamics=False, # Test basic KAN first
        use_geodesic_attn=False, # Keep it simple
        eta_b=0.01,
        eta_f=0.01,
        D=16,
        K=4,
        P=8
    )
    
    # 2. Initialize Adapter
    adapter = GeometricIGBundleAdapter(config)
    print("Adapter Initialized.")
    print(f"Manifold Type: {adapter.manifold_type}")
    
    # Check if KAN
    from igbundle.geometry.kan_manifold import KanManifold
    if isinstance(adapter.manifold, KanManifold):
        print("Confirmed: Using KanManifold.")
    else:
        print(f"Error: Manifold is {type(adapter.manifold)}")
        return

    # 3. Dummy Input
    # (Batch, Time, Hidden)
    hidden_states = torch.randn(2, 10, config.hidden_size)
    
    # 4. Forward Pass
    print("Running Forward Pass...")
    try:
        output = adapter(hidden_states)
        print("Forward Pass Successful.")
        print(f"Output Shape: {output.shape}")
        
        if torch.isnan(output).any():
            print("ERROR: NaNs detected in output!")
        else:
            print("Validation: Output is clean (No NaNs).")
            
    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kan_inference()
