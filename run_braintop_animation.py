
import requests
import json
import numpy as np
import os
import sys
import time

# Import existing wiz
from generate_braintop_viz import generate_viz

# Configuration
API_BASE = "http://192.168.56.1:1234/v1"
MODEL_ID = "ig-bundlellmv2-hamiltonians"

# Queries representing transition from Rest -> Logic -> Creative -> Memory
QUERIES = [
    ("State 1: Resting", " "),
    ("State 2: Logic (Loading)", "Solve for x: 2x + 5 = 15."),
    ("State 3: Logic (Solving)", "Step 1: Subtract 5 from both sides. 2x = 10."),
    ("State 4: Logic (Answer)", "Step 2: Divide by 2. x = 5."),
    ("State 5: Creative (Dreaming)", "imagine a neon forest"),
    ("State 6: Creative (Writing)", "neon trees glow bright / circuits hum in silent night / digital roots grow"),
    ("State 7: Memory (Access)", "Recall the capital of France."),
    ("State 8: Memory (Retrieval)", "Paris is the capital of France.")
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
    print("--- Starting Braintop Integrated Animation ---")
    
    ckpt_path = "output/igbundle_v2_cp300_merged"
    output_html = "output/braintop_animated.html"
    
    frames_metadata = []
    
    for label, prompt in QUERIES:
        print(f"Capturing Frame: {label}...")
        emb = get_remote_embedding(prompt)
        meta = process_embedding_to_metadata(emb)
        
        # Inject label into first node concept for display
        meta[0]['concept'] = label
        
        frames_metadata.append(meta)
        time.sleep(0.5)

    print(f"Generating Animation with {len(frames_metadata)} frames...")
    
    generate_viz(
        checkpoint_path=ckpt_path,
        output_file=output_html,
        lite_mode=True,
        node_metadata=frames_metadata # Pass List of Lists
    )
    
    print(f"Animation Saved: {output_html}")

if __name__ == "__main__":
    generate_animated_imaging()
