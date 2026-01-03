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

def generate_viz(checkpoint_path, output_file, lite_mode=False, node_metadata=None):
    """
    Main entry point for generating topology visualizations.
    node_metadata: dict mapping node_id or index to metadata dict.
    """
    print(f"Loading weights from {checkpoint_path}...")
    weights_path = os.path.join(checkpoint_path, "adapter_weights.pt")
    
    if not os.path.exists(weights_path):
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin") # Fallback for some formats
        if not os.path.exists(weights_path):
            print("Warning: Could not find adapter_weights.pt, using default layout.")
            state_dict = {}
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
    
    # Extract projection weights for latent basis
    proj_weight = None
    for k, v in state_dict.items():
        if "input_proj.weight" in k:
            proj_weight = v.float().numpy() 
            break
            
    if proj_weight is None:
        # If no weights, we use a random projection for the "scaffold"
        print("Using scaffold projection (uninitialized).")
        num_nodes = 512 # Default
        embeddings = np.random.randn(num_nodes, 128)
    else:
        num_nodes = proj_weight.shape[0]
        embeddings = proj_weight
    
    # LITE MODE: Downsample
    if lite_mode:
        print("⚡ LITE MODE ACTIVE: Downsampling to 50 nodes for lightweight web preview.")
        if num_nodes > 50:
            indices = np.random.choice(num_nodes, 50, replace=False)
            embeddings = embeddings[indices]
            num_nodes = 50
    
    print(f"Building topology with {num_nodes} nodes...")
    builder = TopologyBuilder("igbundle_manifold", "IGBundle Real-time Fiber Space")
    
    concepts = [f"Fiber {i}" for i in range(num_nodes)]
    builder.add_conceptual_layer(
        "latent_basis", 
        num_nodes=num_nodes, 
        concepts=concepts, 
        embeddings=embeddings
    )
    
    manifold_type = "hyperbolic" if "riemannian" in checkpoint_path.lower() else "euclidean"
    if "euclidean" in output_file.lower(): manifold_type = "euclidean"
    if "hyperbolic" in output_file.lower() or "riemannian" in output_file.lower(): manifold_type = "hyperbolic"
    
    print(f"Manifold Geometry: {manifold_type.upper()}")
    
    builder.add_riemannian_layer(
        "ideal_bundle", 
        num_nodes=num_nodes, 
        manifold_type=manifold_type, 
        radius=1.0 if lite_mode else 1.5
    )
    
    # Optional: Apply activations to ideal_bundle nodes
    if node_metadata:
        # Get the layer and inject metadata into nodes
        topo = builder.topology
        layer = topo.get_layer("ideal_bundle")
        if layer:
            for i, node in enumerate(layer.nodes):
                if i < len(node_metadata):
                    node.metadata.update(node_metadata[i])
                    # Also update labels with activation if relevant
                    if "activation" in node_metadata[i]:
                        node.label = f"{node.label} ({node_metadata[i]['activation']:.2f})"

    builder.connect_layers("latent_basis", "ideal_bundle", "nearest", num_connections=1 if lite_mode else 2)
    
    topology = builder.build()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualization to {output_file}...")
    visualizer = TopologyVisualizer(topology)
    
    # Style the visualization if metadata is present
    color_by = "activation" if node_metadata else "layer"
    size_by = "activation" if node_metadata else "degree"
    
    fig = visualizer.create_figure(color_by=color_by, size_by=size_by)
    visualizer.save(output_file)
    print(f"Visualization saved to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="trained_adapter")
    parser.add_argument("--output", default="output/igbundle_topology.html")
    parser.add_argument("--lite", action="store_true", help="Generate lightweight version for web preview")
    args = parser.parse_args()
    
    generate_viz(args.checkpoint, args.output, args.lite)
