
import requests
import json
import time
import numpy as np
import os
import sys
import re
from collections import defaultdict

# Configuration
API_BASE = "http://192.168.56.1:1234/v1"
MODEL_ID = "ig-bundlellmv2-hamiltonians"
NUM_TESTS = 25

# Domains
DOMAINS = ["Logic", "Math", "Coding", "Strategy", "Physics"]

def generate_suite(n=25):
    suite = []
    for i in range(n):
        domain = DOMAINS[i % len(DOMAINS)]
        prompt = f"Question {i}: Solve a concise problem in {domain}. Let's think step by step."
        
        # Specifics
        if domain == "Math":
            a, b = np.random.randint(2, 20, 2)
            prompt = f"What is {a} * {b}? step by step."
        elif domain == "Logic":
            prompt = f"If A is true and B is false, is A or B true? step by step."
            
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
        resp = requests.post(url, json=payload, timeout=30)
        latency = time.time() - start
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            return content, latency
        return None, 0
    except Exception as e:
        print(f"API Error: {e}")
        return None, 0

def broadcast_state(data):
    """Atomic write to prevent race conditions."""
    temp_file = "dashboard_state.tmp"
    target_file = "dashboard_state.json"
    try:
        with open(temp_file, "w") as f:
            json.dump(data, f)
        os.replace(temp_file, target_file)
    except Exception as e:
        print(f"Broadcast Error: {e}")

def run_eval():
    print(f"--- Starting Quick QA (x{NUM_TESTS}) ---")
    suite = generate_suite(NUM_TESTS)
    
    # Init Embedder
    from sentence_transformers import SentenceTransformer
    try:
        broadcaster_model = SentenceTransformer('all-MiniLM-L6-v2')
    except:
        broadcaster_model = None
        print("Warning: Local embedder failed. Dashboard will rely on fallback.")
    
    print(f"{'ID':<4} | {'Domain':<10} | {'Lat':<6} | {'Steps':<5} | {'Preview'}")
    print("-" * 60)
    
    for item in suite:
        prompt = item['prompt']
        domain = item['domain']
        
        # START
        if broadcaster_model:
            emb = list(broadcaster_model.encode([prompt])[0].astype(float))
        else: emb = None
        
        broadcast_state({
            "status": "processing",
            "id": item['id'],
            "domain": domain,
            "prompt": prompt,
            "embedding": emb,
            "timestamp": time.time()
        })
        
        response, lat = get_completion([{"role": "user", "content": prompt}])
        
        if response:
            step_list = re.split(r'[.\n]+', response)
            step_list = [s.strip() for s in step_list if len(s.strip()) > 5]
            steps = len(step_list)
            
            preview = response[:30].replace('\n', ' ') + "..."
            print(f"{item['id']:<4} | {domain:<10} | {lat:6.2f} | {steps:<5} | {preview}")
            
            # ANIMATE STEPS (Quickly: 0.5s per step)
            for idx, step_text in enumerate(step_list):
                step_emb = None
                if broadcaster_model:
                     step_emb = list(broadcaster_model.encode([step_text])[0].astype(float))
                     
                broadcast_state({
                    "status": "processing",
                    "id": item['id'],
                    "domain": domain,
                    "prompt": f"[Step {idx+1}/{steps}] {step_text}",
                    "embedding": step_emb,
                    "timestamp": time.time()
                })
                time.sleep(0.5) # Quick pace
            
            # FINISH
            broadcast_state({
                "status": "complete",
                "id": item['id'],
                "domain": domain,
                "prompt": f"ANSWER: {preview}",
                "embedding": list(broadcaster_model.encode([response[:50]])[0].astype(float)) if broadcaster_model else None,
                "latency": lat,
                "timestamp": time.time()
            })
            time.sleep(1.0)
            
        else:
            print(f"{item['id']:<4} | {domain:<10} | FAILED | -     | -")
            
    # Done
    broadcast_state({"status": "idle", "prompt": "Quick QA Completed.", "embedding": None})

if __name__ == "__main__":
    run_eval()
