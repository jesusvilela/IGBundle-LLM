import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter
from igbundle.dynamics.hamiltonian import HamiltonianSystem
from igbundle.modules.attention import PoincareAttention
from igbundle.modules.vision import VisionProjector
from igbundle.modules.memory import GeometricPhaseMemory

def test_phase_memory():
    print("\nTesting GeometricPhaseMemory...")
    mem = GeometricPhaseMemory(dim=32)
    # T=5 path
    coords = torch.randn(2, 5, 4, 32) # B, T, P, D
    
    context, phases = mem(coords)
    print(f"Context Shape: {context.shape}")
    print(f"Phase Shape: {phases.shape}")
    
    assert context.shape == (2, 5, 32)
    assert not torch.isnan(context).any()
    assert context.shape == (2, 5, 32)
    assert not torch.isnan(context).any()
    print("Phase Memory Test Passed.")

def test_hamiltonian():
    print("Testing HamiltonianSystem...")
    sys = HamiltonianSystem(hidden_dim=64)
    q = torch.randn(2, 64) * 0.1 # Small initial q
    p = torch.zeros_like(q)
    
    # Check energy
    H = sys.hamiltonian(q, p)
    print(f"Initial Energy: {H}")
    
    # Step
    q_new, p_new = sys.symplectic_step(q, p, dt=0.01)
    H_new = sys.hamiltonian(q_new, p_new)
    print(f"New Energy: {H_new}")
    
    # Check conservation (approx)
    diff = (H_new - H).abs().mean().item()
    print(f"Energy Drift: {diff}")
    assert not torch.isnan(q_new).any()
    print("Hamiltonian Test Passed.")

def test_poincare_attention():
    print("\nTesting PoincareAttention...")
    attn = PoincareAttention(dim=64)
    coords = torch.randn(2, 32, 4, 64) * 0.1 # B, T, P, D
    # Values = coords for now
    
    out = attn(coords, coords)
    print(f"Attention Output Shape: {out.shape}")
    assert out.shape == coords.shape
    assert not torch.isnan(out).any()
    print("PoincareAttention Test Passed.")

def test_vision_projector():
    print("\nTesting VisionProjector...")
    proj = VisionProjector(vision_dim=1152, bottleneck_dim=64)
    pixel_values = torch.randn(2, 196, 1152) # B, N, V
    
    out = proj(pixel_values)
    print(f"Projected Vision Shape: {out.shape}")
    assert out.shape == (2, 196, 64)
    print("VisionProjector Test Passed.")

def test_full_adapter():
    print("\nTesting Full V2 Adapter...")
    config = IGBundleConfig(
        hidden_size=128,
        latent_dim=32,
        num_components=4,
        num_categories=5,
        vision_dim=1152,
        use_dynamics=True,
        use_geodesic_attn=True,
        supported_modalities=["vision"]
    )
    
    adapter = GeometricIGBundleAdapter(config)
    x = torch.randn(2, 10, 128) # B, T, H
    
    # Forward pass (Text only)
    out, state = adapter(x)
    print(f"Adapter Output Shape: {out.shape}")
    print(f"Geo State Coords: {state.base_coordinates.shape}")
    print(f"Compiler Logits: {state.op_logits.shape}")
    
    # Forward pass (Multimodal Stub)
    # pixel_values attached
    pass 
    
    print("Full Adapter Test Passed.")

if __name__ == "__main__":
    test_hamiltonian()
    test_poincare_attention()
    test_vision_projector()
    test_phase_memory()
    test_full_adapter()
