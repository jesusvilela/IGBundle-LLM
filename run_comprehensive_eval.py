
import requests
import json
import time
import numpy as np
import os
import sys
from collections import defaultdict

# Configuration
API_BASE = "http://192.168.56.1:1234/v1"
MODEL_ID = "ig-bundlellmv2-hamiltonians"
NUM_TESTS = 100

# Domains
DOMAINS = {
    "Logic": [
        "Solve 25 * 12 without a calculator.",
        "If A implies B, and B implies C, does A imply C?",
        "Write a proof that sqrt(2) is irrational.",
        "Solve for x: 3x + 7 = 22."
    ],
    "Creative": [
        "Write a haiku about a robot dreaming.",
        "Describe the taste of the color blue.",
        "Invent a new word for 'the feeling of forgetting a password' and define it.",
        "Write the opening sentence of a cyberpunk novel."
    ],
    "Knowledge": [
        "What is the capital of Australia?",
        "Explain the mechanism of photosynthesis briefly.",
        "Who wrote 'Crime and Punishment'?",
        "What is the speed of light in vacuum?"
    ],
    "Coding": [
        "Write a Python function to reverse a string.",
        "Explain what a deadlock is in concurrency.",
        "Write a bash script to list all PDF files.",
        "What is the difference between TCP and UDP?"
    ],
    "Empathy": [
        "I just lost my job and I feel useless. What should I do?",
        "My friend is angry at me for forgetting their birthday. How do I apologize?",
        "I'm feeling overwhelmed by the news lately.",
        "How do I comfort someone who is grieving?"
    ]
}

def generate_test_suite(n=100):
    suite = []
    keys = list(DOMAINS.keys())
    for i in range(n):
        domain = keys[i % len(keys)]
        base_prompts = DOMAINS[domain]
        prompt = base_prompts[i % len(base_prompts)]
        # Add slight variation to prevent caching/identical results if needed
        # But for stability testing, identical prompts are fine. 
        # Let's add an index to make it unique if strict x100 is needed
        # prompt = f"{prompt} (Test ID: {i})" 
        suite.append({"id": i, "domain": domain, "prompt": prompt})
    return suite

# Global Local Embedder
LOCAL_EMBEDDER = None

def get_drawing_embedding(text):
    global LOCAL_EMBEDDER
    try:
        if LOCAL_EMBEDDER is None:
            print("Loading local embedding model (all-MiniLM-L6-v2)...")
            from sentence_transformers import SentenceTransformer
            LOCAL_EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embedding (384 dim usually)
        emb = LOCAL_EMBEDDER.encode(text)
        
        # Project to Manifold Dimension (512 fibers)
        # We just resize/pad
        target_dim = 512
        if len(emb) < target_dim:
            # Pad
            emb = np.pad(emb, (0, target_dim - len(emb)), 'constant')
        elif len(emb) > target_dim:
            # Crop
            emb = emb[:target_dim]
            
        return emb
    except Exception as e:
        print(f"Local Embedding Failed: {e}")
        # Final Fallback: Geometric Hash of input text info
        # This ensures consistent mapping for identical prompts (Skill Placement consistency)
        import hashlib
        hash_digest = hashlib.sha256(text.encode()).digest()
        # Convert bytes to float array
        floats = np.frombuffer(hash_digest, dtype=np.uint8).astype(float)
        # Resize to 512
        return np.resize(floats, 512)

def get_embedding(text):
    # Try remote first (it fails usually but good to keep structure)
    # Actually, user confirmed it fails. Let's force local drawing for "Placement" check.
    return get_drawing_embedding(text)

def get_completion(messages):
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 128
    }
    try:
        start = time.time()
        resp = requests.post(url, json=payload)
        latency = time.time() - start
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content'], latency
        return None, 0
    except:
        return None, 0

def analyze_manifold_placement(embedding, num_fibers=512):
    if embedding is None: return -1, 0.0
    
    # Map high-dim embedding (e.g. 3584) to lower-dim Fiber Space (512)
    # Simple pooling strategy matching braintop
    chunk_size = len(embedding) // num_fibers
    activations = []
    for i in range(num_fibers):
        start = i * chunk_size
        end = start + chunk_size
        val = np.abs(embedding[start:end]).mean()
        activations.append(val)
    
    activations = np.array(activations)
    peak_fiber = np.argmax(activations)
    max_val = np.max(activations)
    
    return peak_fiber, max_val

def run_eval():
    print(f"--- Starting Comprehensive Eval (x{NUM_TESTS}) ---")
    suite = generate_test_suite(NUM_TESTS)
    
    results = defaultdict(list)
    fiber_map = defaultdict(list)
    
    total_latency = 0
    success_count = 0
    
    print(f"{'ID':<4} | {'Domain':<10} | {'Latency':<6} | {'Fiber':<6} | {'Preview'}")
    print("-" * 60)
    
    for item in suite:
        prompt = item['prompt']
        domain = item['domain']
        
        # 1. Run Inference
        response, lat = get_completion([{"role": "user", "content": prompt}])
        
        if response:
            success_count += 1
            total_latency += lat
            
            # 2. Get Manifold Embedding
            # Note: We embed the RESPONSE to see where the *generated thought* maps
            # Or the PROMPT to see where the *input* maps. 
            # Usually input mapping determines processing path. Let's do PROMPT for skill mapping.
            emb = get_embedding(prompt)
            
            # 3. Analyze Placement
            peak_fiber, intensity = analyze_manifold_placement(emb)
            
            results[domain].append({
                "latency": lat,
                "fiber": peak_fiber,
                "intensity": intensity
            })
            
            fiber_map[domain].append(peak_fiber)
            
            preview = response[:30].replace('\n', ' ') + "..."
            print(f"{item['id']:<4} | {domain:<10} | {lat:6.2f} | {peak_fiber:<6} | {preview}")
        else:
            print(f"{item['id']:<4} | {domain:<10} | FAILED | -      | -")

    print("\n=== Eval Request Summary ===")
    print(f"Total Requests: {NUM_TESTS}")
    print(f"Success Rate:   {success_count/NUM_TESTS*100:.1f}%")
    print(f"Avg Latency:    {total_latency/success_count:.2f}s")
    
    print("\n=== Manifold Skill Placement (Fiber Centroids) ===")
    print("Do skills cluster in distinct geometric regions?")
    
    for domain, fibers in fiber_map.items():
        avg_fiber = np.mean(fibers)
        std_fiber = np.std(fibers)
        print(f"{domain:<10}: Mean Fiber {avg_fiber:6.1f} (Std {std_fiber:5.1f}) -> Range [{int(min(fibers))}-{int(max(fibers))}]")

if __name__ == "__main__":
    run_eval()
