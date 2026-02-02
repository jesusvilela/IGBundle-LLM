
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
NUM_TESTS = 50

# Reuse domains but limit count
DOMAINS = {
    "Logic": [
        "Solve 3x + 10 = 25.",
        "Is 97 a prime number?",
        "If a train leaves Chicago at 50mph...",
        "What is the square root of 144?"
    ],
    "Creative": [
        "Write a tweet about quantum physics.",
        "Describe a sunset on Mars.",
        "Invent a name for a new coffee brand.",
        "Write a haiku about code."
    ],
    "Knowledge": [
        "What is the capital of Japan?",
        "Who painted the Mona Lisa?",
        "What year did WWII end?",
        "What is the chemical symbol for Gold?"
    ],
    "Coding": [
        "Write a Python loop to print 1 to 10.",
        "What is a git commit?",
        "Explain CSS Grid briefly.",
        "How do you define a function in JavaScript?"
    ],
    "Empathy": [
        "I'm feeling tired today.",
        "Congratulate me on my graduation.",
        "How do I deal with stress?",
        "My dog is sick, what should I do?"
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
        
        # Get embedding (384 dim -> 512 dim)
        emb = LOCAL_EMBEDDER.encode(text)
        target_dim = 512
        if len(emb) < target_dim:
            emb = np.pad(emb, (0, target_dim - len(emb)), 'constant')
        elif len(emb) > target_dim:
            emb = emb[:target_dim]
        return emb
    except Exception as e:
        print(f"Local Embedding Failed: {e}")
        return np.random.randn(512)

def generate_test_suite(n=50):
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
    chunk_size = max(1, len(embedding) // num_fibers)
    activations = []
    for i in range(num_fibers):
        start = i * chunk_size
        end = start + chunk_size
        if start < len(embedding):
             val = np.abs(embedding[start:end]).mean() if end > start else 0
        else:
             val = 0
        activations.append(val)
    
    activations = np.array(activations)
    peak_fiber = np.argmax(activations)
    max_val = np.max(activations)
    return peak_fiber, max_val

def run_eval():
    print(f"--- Starting Eval BATCH 50 (x{NUM_TESTS}) ---")
    suite = generate_test_suite(NUM_TESTS)
    
    fiber_map = defaultdict(list)
    total_latency = 0
    success_count = 0
    
    print(f"{'ID':<4} | {'Domain':<10} | {'Latency':<6} | {'Fiber':<6} | {'Preview'}")
    print("-" * 60)
    
    for item in suite:
        prompt = item['prompt']
        domain = item['domain']
        
        response, lat = get_completion([{"role": "user", "content": prompt}])
        
        if response:
            success_count += 1
            total_latency += lat
            
            emb = get_drawing_embedding(prompt)
            peak_fiber, intensity = analyze_manifold_placement(emb)
            fiber_map[domain].append(peak_fiber)
            
            preview = response[:30].replace('\n', ' ') + "..."
            print(f"{item['id']:<4} | {domain:<10} | {lat:6.2f} | {peak_fiber:<6} | {preview}")
        else:
            print(f"{item['id']:<4} | {domain:<10} | FAILED | -      | -")

    print("\n=== Eval Request Summary (Batch 50) ===")
    print(f"Total Requests: {NUM_TESTS}")
    print(f"Success Rate:   {success_count/NUM_TESTS*100:.1f}%")
    avg_lat = total_latency/max(success_count, 1)
    print(f"Avg Latency:    {avg_lat:.2f}s")
    
    print("\n=== Manifold Skill Placement (Fiber Centroids) ===")
    for domain, fibers in fiber_map.items():
        avg_fiber = np.mean(fibers)
        print(f"{domain:<10}: Mean Fiber {avg_fiber:6.1f} (Std {np.std(fibers):5.1f})")

if __name__ == "__main__":
    run_eval()
