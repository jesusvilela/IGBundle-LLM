import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Path Setup
sys.path.append(os.path.abspath("src"))

from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter

def visualize_potential(adapter_path):
    print(f"Loading Adapter from {adapter_path}...")
    
    config = IGBundleConfig(
        hidden_size=3584,
        latent_dim=128,
        num_components=8,
        use_dynamics=True,
        num_leapfrog_steps=4
    )
    
    adapter = create_geometric_adapter(config)
    
    # Load weights
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location="cpu")
        # Load with strict=False to allow potential_net initialization if missing
        missing, unexpected = adapter.load_state_dict(state, strict=False)
        print(f"Loaded weights. Missing keys: {len(missing)} (Expected if migrating from Phase 2)")
    else:
        print("Adapter path not found. Using random init.")
        
    adapter.eval()
    
    # 2D Slice of Poincare Disk
    # We visualize the potential V(q) for the first component P=0
    # varying dim 0 (x) and dim 1 (y), keeping others 0.
    
    N = 100
    x = np.linspace(-0.95, 0.95, N)
    y = np.linspace(-0.95, 0.95, N)
    X, Y = np.meshgrid(x, y)
    
    V_map = np.zeros((N, N))
    
    print("Scanning Potential Landscape...")
    with torch.no_grad():
        for i in range(N):
            for j in range(N):
                # Construct q point
                # Shape: [1, 1, P, D] -> We just need input to potential_net
                # potential_net takes [..., D]
                
                q_point = torch.zeros(adapter.D)
                q_point[0] = X[i, j]
                q_point[1] = Y[i, j]
                
                # Check constraints (Poincare Ball boundary)
                if torch.norm(q_point) >= 1.0:
                    V_map[i, j] = np.nan
                    continue
                
                # Forward Potential
                # NeuralPotential takes [..., D]
                # We can batch this? Yes, but loop is fine for N=100
                v = adapter.potential_net(q_point.unsqueeze(0)) # [1, 1]
                V_map[i, j] = v.item()
                
    # Plot
    plt.figure(figsize=(10, 8))
    # Use 'viridis' or 'plasma'
    # High Potential (Barrier) = Yellow? Low (Attractor) = Blue/Purple?
    plt.imshow(V_map, extent=[-0.95, 0.95, -0.95, 0.95], origin='lower', cmap='magma')
    plt.colorbar(label='Potential Energy V(q)')
    
    # Draw Circle Boundary
    circle = plt.Circle((0, 0), 1.0, color='r', fill=False, linestyle='--', label='Poincare Boundary')
    plt.gca().add_artist(circle)
    
    plt.title(f"Neural Potential Landscape (Slice dim 0-1)\nSource: {os.path.basename(adapter_path)}")
    plt.xlabel("Latent Dim 0")
    plt.ylabel("Latent Dim 1")
    plt.legend()
    
    out_file = "output/potential_landscape.png"
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    # Prefer Phase 3 result, fallback to Phase 2
    path = "output/phase3_adapter_potential.pt"
    if not os.path.exists(path):
        path = "output/phase2_adapter_arc.pt"
        
    visualize_potential(path)
