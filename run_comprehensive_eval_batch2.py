
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

# BATCH 2 DOMAINS (New Prompts)
DOMAINS = {
    "Logic": [
        "If all Bloops are Zazzers and no Zazzers are Gleeps, are Bloops Gleeps?",
        "Calculate the area of a circle with radius 5.",
        "What is the next number in the sequence: 2, 3, 5, 7, 11...?",
        "True or False: The sum of two odd numbers is always even."
    ],
    "Creative": [
        "Write a 4-line poem about a clock that runs backwards.",
        "Describe a color that doesn't exist.",
        "Write a dialogue between a toaster and a fridge.",
        "What does silence sound like?"
    ],
    "Knowledge": [
        "Who discovered Penicillin?",
        "What implies the Theory of Relativity?",
        "Name three noble gases.",
        "When was the printing press invented?"
    ],
    "Coding": [
        "Write a SQL query to select all users over 18.",
        "Explain the concept of recursion to a 5-year-old.",
        "How do you merge two dictionaries in Python?",
        "What is the output of print(0.1 + 0.2 == 0.3) in Python?"
    ],
    "Empathy": [
        "My cat ran away and I feel guilty.",
        "I'm nervous about a job interview tomorrow.",
        "How do I tell my friend I don't like their cooking?",
        "I feel lonely in this big city."
    ]
}

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
        
        # Project to Manifold Dimension (512 fibers) with noise for variation
        target_dim = 512
        if len(emb) < target_dim:
            emb = np.pad(emb, (0, target_dim - len(emb)), 'constant')
        elif len(emb) > target_dim:
            emb = emb[:target_dim]
            
        return emb
    except Exception as e:
        print(f"Local Embedding Failed: {e}")
        return np.random.randn(512)

def generate_test_suite(n=100):
    suite = []
    keys = list(DOMAINS.keys())
    for i in range(n):
        domain = keys[i % len(keys)]
        base_prompts = DOMAINS[domain]
        prompt = base_prompts[i % len(base_prompts)]
        suite.append({"id": i, "domain": domain, "prompt": prompt})
    return suite

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
    chunk_size = len(embedding) // num_fibers
    activations = []
    for i in range(num_fibers):
        start = i * chunk_size
        end = start + chunk_size
        if end > start:
            val = np.abs(embedding[start:end]).mean()
        else:
            val = 0
        activations.append(val)
    
    activations = np.array(activations)
    peak_fiber = np.argmax(activations)
    max_val = np.max(activations)
    return peak_fiber, max_val

def run_eval():
    print(f"--- Starting Comprehensive Eval BATCH 2 (x{NUM_TESTS}) ---")
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
            emb = get_drawing_embedding(prompt)
            
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

    print("\n=== Eval Request Summary (Batch 2) ===")
    print(f"Total Requests: {NUM_TESTS}")
    print(f"Success Rate:   {success_count/NUM_TESTS*100:.1f}%")
    print(f"Avg Latency:    {total_latency/max(success_count, 1):.2f}s")
    
    print("\n=== Manifold Skill Placement (Fiber Centroids) ===")
    for domain, fibers in fiber_map.items():
        avg_fiber = np.mean(fibers)
        std_fiber = np.std(fibers)
        print(f"{domain:<10}: Mean Fiber {avg_fiber:6.1f} (Std {std_fiber:5.1f})")

if __name__ == "__main__":
    run_eval()
