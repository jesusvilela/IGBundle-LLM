
import requests
import json
import numpy as np
import os
import sys

# Import existing wiz
from generate_braintop_viz import generate_viz

# Configuration
API_BASE = "http://192.168.56.1:1234/v1"
MODEL_ID = "ig-bundlellmv2-hamiltonians"

def get_remote_embedding(text):
    print(f"getting embedding for: '{text[:20]}...'")
    url = f"{API_BASE}/embeddings"
    payload = {
        "input": text,
        "model": MODEL_ID
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            # (1, 3584) usually
            embedding = data['data'][0]['embedding']
            return np.array(embedding)
        else:
            print(f"Error getting embedding: {response.text}")
            return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def generate_remote_imaging():
    print("--- Starting Braintop Remote Imaging ---")
    
    # 1. Capture "Thought" State
    prompt = "Explain Parallel Transport."
    embedding = get_remote_embedding(prompt)
    
    if embedding is None:
        print("Failed to get remote embedding. Is the server running with embedding support?")
        # Fallback for demo if server lacks embedding endpoint (common in some chat-only modes)
        print("Using synthetic pattern for demo purposes.")
        embedding = np.random.randn(512) 
    
    # 2. Map Embedding to Nodes (Simulated Activation)
    # The adapter has N nodes. We map the embedding dimensionality to these nodes.
    # If Dim=3584 and Nodes=512, we can just reshape or pool.
    # Let's assume we map the magnitude of chunks.
    
    num_nodes = 512
    # Simple strategy: chunk embedding into num_nodes pieces and take mean
    chunk_size = len(embedding) // num_nodes
    if chunk_size < 1: 
        # Expand
        activations = np.resize(embedding, num_nodes)
    else:
        activations = []
        for i in range(num_nodes):
            start = i * chunk_size
            end = start + chunk_size
            chunk = embedding[start:end]
            if len(chunk) > 0:
                val = np.abs(chunk).mean() # Magnitude as activation
            else:
                val = 0
            activations.append(val)
        activations = np.array(activations)

    # Normalize for Visualization (0-1)
    if activations.max() > 0:
        activations = activations / activations.max()

    # 3. Create Metadata
    node_metadata = []
    for i in range(num_nodes):
        node_metadata.append({
            "activation": float(activations[i]),
            "concept": f"Neuron Group {i}"
        })
        
    # 4. Generate Topology
    # We point to a dummy path for weights, generate_viz handles missing weights by random scaffold
    # But since we have activations, the "scaffold" (structure) matters less than the "color" (activity).
    output_html = "output/braintop_remote_imaging.html"
    
    # Use "output/igbundle_v2_cp300_merged" as "checkpoint" to possibly get some weights
    # Or just use current dir to be safe
    ckpt_path = "output/igbundle_v2_cp300_merged"
    
    generate_viz(
        checkpoint_path=ckpt_path,
        output_file=output_html,
        lite_mode=True, # Web friendly
        node_metadata=node_metadata
    )
    
    print(f"Braintop Imaging Complete: {output_html}")

if __name__ == "__main__":
    generate_remote_imaging()
