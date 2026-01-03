import torch
import numpy as np
import os
import argparse
from pathlib import Path

# Add braintop/src to path for local development import
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
braintop_src = os.path.join(base_dir, "braintop", "src")
if braintop_src not in sys.path:
    sys.path.append(braintop_src)

from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer

def generate_viz(checkpoint_path, output_file, lite_mode=False):
    print(f"Loading weights from {checkpoint_path}...")
    weights_path = os.path.join(checkpoint_path, "adapter_weights.pt")
    
    if not os.path.exists(weights_path):
        weights_path = "trained_adapter/adapter_weights.pt"
        if not os.path.exists(weights_path):
            print("Error: Could not find adapter_weights.pt")
            return

    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Extract projection weights
    proj_weight = None
    for k, v in state_dict.items():
        if "input_proj.weight" in k:
            proj_weight = v.float().numpy() 
            print(f"Found projection layer: {k} {proj_weight.shape}")
            break
            
    if proj_weight is None:
        print("No input_proj.weight found in adapter.")
        return

    num_nodes = proj_weight.shape[0]
    embeddings = proj_weight
    
    # LITE MODE: Downsample to max 50 nodes for <1MB file size
    if lite_mode:
        print("⚡ LITE MODE ACTIVE: Downsampling to 50 nodes for lightweight web preview.")
        if num_nodes > 50:
            indices = np.random.choice(num_nodes, 50, replace=False)
            embeddings = embeddings[indices]
            num_nodes = 50
    
    print(f"Building topology with {num_nodes} nodes...")
    builder = TopologyBuilder("igbundle_manifold", "IGBundle Fiber Space")
    
    concepts = [f"Fiber Dim {i}" for i in range(num_nodes)]
    builder.add_conceptual_layer(
        "latent_basis", 
        num_nodes=num_nodes, 
        concepts=concepts, 
        embeddings=embeddings
    )
    
    # Lite mode uses simplified manifold representation
    manifold_res = 1.0 if lite_mode else 2.0
    manifold_type = "hyperbolic" if "riemannian" in checkpoint_path.lower() else "euclidean"
    # Allow override
    if "euclidean" in output_file.lower(): manifold_type = "euclidean"
    if "hyperbolic" in output_file.lower() or "riemannian" in output_file.lower(): manifold_type = "hyperbolic"
    
    print(f"Manifold Geometry: {manifold_type.upper()}")
    
    builder.add_riemannian_layer(
        "ideal_bundle", 
        num_nodes=num_nodes, 
        manifold_type=manifold_type, 
        radius=manifold_res
    )
    
    conn_count = 1 if lite_mode else 3
    builder.connect_layers("latent_basis", "ideal_bundle", "nearest", num_connections=conn_count)
    
    topology = builder.build()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualization to {output_file}...")
    visualizer = TopologyVisualizer(topology)
    visualizer.save(output_file)
    
    # Static Export (Only for full mode or specific request)
    if not lite_mode:
        png_file = output_file.replace(".html", ".png")
        print(f"Generating static image to {png_file}...")
        try:
            fig = visualizer.create_figure() 
            fig.write_image(png_file, engine="kaleido")
        except Exception as e:
            print(f"Warning: Could not save PNG: {e}")
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="trained_adapter")
    parser.add_argument("--output", default="output/igbundle_topology.html")
    parser.add_argument("--lite", action="store_true", help="Generate lightweight version for web preview")
    args = parser.parse_args()
    
    generate_viz(args.checkpoint, args.output, args.lite)
