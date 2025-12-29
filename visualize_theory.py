import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import argparse

# Dummy definition to load weights without full model
class IGBundleLayer(nn.Module):
    def __init__(self, hidden_size, bottleneck_dim):
        super().__init__()
        self.input_proj = nn.Linear(hidden_size, bottleneck_dim)
        # We only need to visualize what we can verify. 
        # Actually proper visualization requires running the model.
        # But we can visualize the WEIGHT distributions of the adapter 
        # to show structure vs random initialization.

def visualize_curvature_distribution(checkpoint_path):
    print("Loading adapter weights...")
    weights_path = os.path.join(checkpoint_path, "adapter_weights.pt")
    if not os.path.exists(weights_path):
        print("Error: adapter_weights.pt not found.")
        return

    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Extract 'sigma' or projection weights
    # Our adapter architecture: input_proj -> processing -> out_proj
    # We can visualize the singular value spectrum of the input projection
    # to see if the mapping to the bundle has collapsed or is distributed.
    
    input_projs = []
    for k, v in state_dict.items():
        if "input_proj.weight" in k:
            input_projs.append(v.float().numpy())
            
    if not input_projs:
        print("No input projections found.")
        return
        
    print(f"Found {len(input_projs)} adapter layers.")
    
    # 1. Singular Value Spectrum
    plt.figure(figsize=(10, 6))
    for i, W in enumerate(input_projs):
        # W is (Bottleneck, Hidden)
        # SVD
        try:
            U, S, V = np.linalg.svd(W, full_matrices=False)
            plt.plot(S, label=f'Layer {i}', alpha=0.5)
        except:
            pass
            
    plt.title("Singular Value Spectrum of Fiber Bundle Projections")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("spectrum.png")
    print("Saved spectrum.png")
    
    # 2. Interactive 3D Manifold visualization (Simulated output space)
    # We will project random unit vectors through the first adapter layer
    # and verify if they form a manifold structure (e.g. not just uniform noise)
    
    W0 = input_projs[0] # (256, 3584)
    # Generate random input sphere
    vecs = np.random.randn(1000, W0.shape[1])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    
    # Project to bundle space
    projected = vecs @ W0.T # (1000, 256)
    
    # Reduce to 3D for visualization (PCA)
    U, S, Vh = np.linalg.svd(projected, full_matrices=False)
    coords_3d = projected @ Vh[:3, :].T
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords_3d[:,0], coords_3d[:,1], coords_3d[:,2], c=coords_3d[:,2], cmap='viridis', alpha=0.6)
    ax.set_title("Tangent Bundle Section (Projected via Layer 0)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    
    print("Displaying interactive plot... (Close window to continue)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="trained_adapter")
    args = parser.parse_args()
    
    visualize_curvature_distribution(args.checkpoint)
