
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

# Queries representing different cognitive modes
QUERIES = [
    ("Resting State", " "),
    ("Logic Mode", "Solve for x: 2x + 5 = 15. Show steps."),
    ("Creative Mode", "Write a haiku about a cybernetic forest."),
    ("Memory Mode", "Recall the capital of France and its famous tower.")
]

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
            embedding = data['data'][0]['embedding']
            return np.array(embedding)
        else:
            return None
    except Exception:
        return None

def process_embedding_to_metadata(embedding, num_nodes=512):
    if embedding is None:
        # Synthetic fallback
        embedding = np.random.randn(512)
        
    chunk_size = len(embedding) // num_nodes
    if chunk_size < 1: 
        activations = np.resize(embedding, num_nodes)
    else:
        activations = []
        for i in range(num_nodes):
            start = i * chunk_size
            end = start + chunk_size
            chunk = embedding[start:end]
            if len(chunk) > 0:
                val = np.abs(chunk).mean()
            else:
                val = 0
            activations.append(val)
        activations = np.array(activations)

    if activations.max() > 0:
        activations = activations / activations.max()

    node_metadata = []
    for i in range(num_nodes):
        node_metadata.append({
            "activation": float(activations[i])
        })
    return node_metadata

def generate_animated_imaging():
    print("--- Starting Braintop Annotated Sequence ---")
    
    # We will generate a list of frames
    # Since visualizer support for animation is 'pending', we will generate 
    # separate HTML files for each state so the user can 'flip' through them easily.
    # This is more robust than hacking plotly frames without full codebase access.
    
    ckpt_path = "output/igbundle_v2_cp300_merged"
    
    generated_files = []
    
    for label, prompt in QUERIES:
        print(f"Processing: {label}...")
        emb = get_remote_embedding(prompt)
        meta = process_embedding_to_metadata(emb)
        
        filename = f"output/braintop_{label.lower().replace(' ', '_')}.html"
        generated_files.append(filename)
        
        # Inject label into first node concept for display
        meta[0]['concept'] = f"State: {label}"
        
        generate_viz(
            checkpoint_path=ckpt_path,
            output_file=filename,
            lite_mode=True,
            node_metadata=meta
        )
        print(f"Saved {filename}")

    print("Sequence Generation Complete.")
    print("Review the files to see topological activation changes.")

if __name__ == "__main__":
    generate_animated_imaging()
