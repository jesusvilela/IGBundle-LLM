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


class BraintopRenderer:
    def __init__(self, checkpoint_path, lite_mode=True):
        self.lite_mode = lite_mode
        print(f"Loading topology from {checkpoint_path}...")
        
        # 1. Load Weights (Simplified)
        weights_path = os.path.join(checkpoint_path, "adapter_weights.pt")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
             if not os.path.exists(weights_path):
                 print("Warning: No weights found. Using scaffold.")
                 state_dict = {}
             else:
                  state_dict = torch.load(weights_path, map_location="cpu")
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        # Extract Projection for Structure
        proj_weight = None
        for k, v in state_dict.items():
            if "input_proj.weight" in k:
                proj_weight = v.float().numpy()
                break
        
        if proj_weight is None:
            num_nodes = 512
            embeddings = np.random.randn(num_nodes, 128)
        else:
            num_nodes = proj_weight.shape[0]
            embeddings = proj_weight

        # Downsample
        if lite_mode and num_nodes > 150:
            indices = np.random.choice(num_nodes, 150, replace=False)
            embeddings = embeddings[indices]
            num_nodes = 150

        # 2. Build Persistent Topology
        print(f"Building persistent topology ({num_nodes} nodes)...")
        self.builder = TopologyBuilder("igbundle_realtime", "IGBundle Real-time")
        
        # --- SEMANTIC MAPPING ---
        # Try to map nodes to concepts if ST is available
        concepts = [f"Fiber {i}" for i in range(num_nodes)]
        
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Define Thesis Anchors
            ANCHORS = [
                "Logic", "Math", "Physics", "Code", "Python", "Algorithm", "Data",
                "Creative", "Art", "Poetry", "Dream", "Emotion", "Love", "Chaos",
                "Knowledge", "History", "Truth", "Fact", "Memory", "Time",
                "Empathy", "Human", "Life", "Soul", "Consciousness"
            ]
            
            print("Mapping fibers to semantic anchors...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            anchor_embs = model.encode(ANCHORS)
            
            # Project embeddings (if raw weights) to match dim if needed, 
            # but usually we compare in the same space. 
            # Our "embeddings" are from the model's fiber space (input_proj).
            # The anchor_embs are from MiniLM. 
            # These spaces are NOT aligned unless we trained them to be (which we didn't explicitly, 
            # but the model has learned English).
            # However, direct cosine might be noisy. 
            # BETTER STRATEGY: assign random subset of fibers to be "The Logic Region" 
            # based on the *Check Skills Placement* results we just found!
            # We found Logic~Fiber 231, Creative~Fiber 138.
            # Let's map based on Index Proximity to our empirical centroids.
            
            # Empirical Centroids from x100 Eval
            CENTROIDS = {
                "Logic": 231,
                "Creative": 138,
                "Knowledge": 196,
                "Coding": 245,
                "Empathy": 178
            }
            
            for i in range(num_nodes):
                # Find closest centroid by Index Distance (1D Manifold assumption for simplicity in labeling)
                # Or just assign labels to the neighborhood.
                best_label = None
                min_dist = float('inf')
                
                for label, center in CENTROIDS.items():
                    dist = abs(i - center)
                    # Wrap around not needed for linear indexing, but let's assume local
                    if dist < 20: # Radius of influence
                        if dist < min_dist:
                            min_dist = dist
                            best_label = label
                
                if best_label:
                    concepts[i] = f"Fiber {i} [{best_label}]"
                    
        except ImportError:
            pass

        self.builder.add_conceptual_layer("latent_basis", num_nodes, concepts, embeddings)
        
        manifold_type = "hyperbolic" if "riemannian" in checkpoint_path.lower() else "euclidean"
        self.builder.add_riemannian_layer("ideal_bundle", num_nodes, manifold_type=manifold_type, radius=1.0)
        
        self.builder.connect_layers("latent_basis", "ideal_bundle", "nearest", num_connections=1 if lite_mode else 2)
        
        # Initial Build
        self.topology = self.builder.build()
        self.visualizer = TopologyVisualizer(self.topology)
        self.num_nodes = num_nodes

    def render_frame(self, prompt, embedding=None):
        # 3. Inject Metadata based on Prompt/Embedding
        # If embedding provided, map to nodes
        layer = self.topology.get_layer("ideal_bundle")
        
        if embedding is not None and layer:
            # Simple simulation of mapping: Chunk the embedding to nodes
            # Embedding (384) -> Nodes (150)
            # We resize embedding to match num_nodes
            
            # Normalize embedding
            if embedding.max() > 0: embedding = embedding / embedding.max()
            
            # Map
            activations = np.resize(embedding, self.num_nodes)
            activations = np.abs(activations) # Magnitude
            
            for i, node in enumerate(layer.nodes):
                val = float(activations[i])
                node.metadata['activation'] = val
                node.label = f"Fiber {i}" # Reset label
                if val > 0.5:
                     node.label = f"Fiber {i} ({val:.2f})"
        
        # 4. Generate Figure
        title = f"Topological Act: {prompt}" if prompt else "Resting State"
        fig = self.visualizer.create_figure(color_by="activation", size_by="activation")
        fig.update_layout(title=title)
        return fig

def generate_viz(checkpoint_path, output_file, lite_mode=False, node_metadata=None):
    """
    Main entry point for generating topology visualizations.
    Now uses the Renderer class internally.
    """
    renderer = BraintopRenderer(checkpoint_path, lite_mode)
    
    # Inject metadata if provided (Single Frame logic)
    if node_metadata and isinstance(node_metadata, list) and not isinstance(node_metadata[0], list):
        # Update topology directly
        layer = renderer.topology.get_layer("ideal_bundle")
        for i, node in enumerate(layer.nodes):
            if i < len(node_metadata):
                node.metadata.update(node_metadata[i])

    # Animation Logic (List of Lists)
    if node_metadata and isinstance(node_metadata, list) and len(node_metadata) > 0 and isinstance(node_metadata[0], list):
         print(f"Generating Animation with {len(node_metadata)} frames...")
         # Re-instantiate visualizer just for animation support which is distinct
         # Actually, we can just use the renderer's visualizer
         return renderer.visualizer.create_animation(node_metadata, color_by="activation", size_by="activation")

    # Static Frame
    fig = renderer.visualizer.create_figure(color_by="activation" if node_metadata else "layer", size_by="activation" if node_metadata else "degree")
    
    if output_file:
         Path(output_file).parent.mkdir(parents=True, exist_ok=True)
         renderer.visualizer.save(output_file)
         print(f"Visualization saved to {output_file}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="trained_adapter")
    parser.add_argument("--output", default="output/igbundle_topology.html")
    parser.add_argument("--lite", action="store_true", help="Generate lightweight version for web preview")
    args = parser.parse_args()
    
    generate_viz(args.checkpoint, args.output, args.lite)
