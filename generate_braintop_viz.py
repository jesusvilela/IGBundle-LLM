import torch
import numpy as np
import os
import argparse
from braintop.utils.builders import TopologyBuilder
from braintop.core.visualizer import TopologyVisualizer

def generate_viz(checkpoint_path, output_file):
    print(f"Loading weights from {checkpoint_path}...")
    weights_path = os.path.join(checkpoint_path, "adapter_weights.pt")
    
    if not os.path.exists(weights_path):
        weights_path = "trained_adapter/adapter_weights.pt"
        if not os.path.exists(weights_path):
            print("Error: Could not find adapter_weights.pt")
            return

    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Extract projection weights
    # We look for the first input projection to visualize the "entry" into the bundle
    proj_weight = None
    for k, v in state_dict.items():
        if "input_proj.weight" in k:
            proj_weight = v.float().numpy() # [256, 3584] usually
            print(f"Found projection layer: {k} {proj_weight.shape}")
            break
            
    if proj_weight is None:
        print("No input_proj.weight found in adapter.")
        return

    # Use the 256 bottleneck dimensions as nodes
    num_nodes = proj_weight.shape[0]
    embeddings = proj_weight
    
    print("Building topology...")
    builder = TopologyBuilder("igbundle_manifold", "IGBundle Fiber Space")
    
    # Layer 1: Conceptual (The bottleneck neurons projected via PCA)
    concepts = [f"Fiber Dim {i}" for i in range(num_nodes)]
    builder.add_conceptual_layer(
        "latent_basis", 
        num_nodes=num_nodes, 
        concepts=concepts, 
        embeddings=embeddings
    )
    
    # Layer 2: Riemannian Sphere (Simulating the normalized bundle space)
    # We map the same nodes to a sphere to show the "ideal" geometry we aim for
    builder.add_riemannian_layer(
        "ideal_bundle", 
        num_nodes=num_nodes, 
        manifold_type="sphere", 
        radius=2.0
    )
    
    # Connect them to show how learned basis maps to ideal manifold
    builder.connect_layers("latent_basis", "ideal_bundle", "one_to_one")
    
    topology = builder.build()
    
    print(f"Generating visualization to {output_file}...")
    visualizer = TopologyVisualizer(topology)
    visualizer.save(output_file)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="trained_adapter")
    parser.add_argument("--output", default="igbundle_topology.html")
    args = parser.parse_args()
    
    generate_viz(args.checkpoint, args.output)
