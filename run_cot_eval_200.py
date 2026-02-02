
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
NUM_TESTS = 200

# Domains focusing on Reasoning (CoT)
DOMAINS = ["Logic", "Math", "Coding", "Strategy", "Physics"]

# Template generator
TEMPLATES = [
    "If {0} implies {1}, and {1} implies {2}, does {0} imply {2}?",
    "Solve for x: {0}x + {1} = {2}.",
    "Write a function to {0} in {1}.",
    "How would you design a system to {0}?",
    "Explain why {0} happens when {1}."
]

VARS = {
    "Logic": [("A", "B", "C"), ("rain", "wet", "slippery"), ("fire", "smoke", "alarm")],
    "Math": [(2, 5, 15), (10, -2, 48), (3, 7, 22), (5, 5, 30)],
    "Coding": [("sort a list", "Python"), ("fetch data", "JS"), ("allocate memory", "C++")],
    "Strategy": [("win chess", "opening"), ("optimize traffic", "lights")],
    "Physics": [("gravity", "mass"), ("friction", "heat")]
}

def generate_cot_suite(n=200):
    suite = []
    for i in range(n):
        domain = DOMAINS[i % len(DOMAINS)]
        # simplified generation for volume
        prompt = f"Question {i}: Solve a complex problem in {domain}. Let's think step by step."
        
        # Inject some variance
        if domain == "Math":
            a, b, c = np.random.randint(1, 100, 3)
            prompt = f"Solve for x: {a}x + {b} = {c}. Let's think step by step."
        elif domain == "Logic":
            prompt = f"If all A are B and some B are C, are some A C? Explain step by step."
            
        suite.append({"id": i, "domain": domain, "prompt": prompt})
    return suite

def get_completion(messages):
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256 # Allow room for CoT
    }
    try:
        start = time.time()
        resp = requests.post(url, json=payload)
        latency = time.time() - start
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            return content, latency
        return None, 0
    except:
        return None, 0

def run_eval():
    print(f"--- Starting CoT Eval (x{NUM_TESTS}) ---")
    suite = generate_cot_suite(NUM_TESTS)
    
    total_latency = 0
    success_count = 0
    total_steps = 0
    
    print(f"{'ID':<4} | {'Domain':<10} | {'Lat':<6} | {'Steps':<5} | {'Preview'}")
    print("-" * 60)
    
    # Init Local Embedder for Broadcasting
    from sentence_transformers import SentenceTransformer
    broadcaster_model = None
    for i in range(5):
        try:
            print(f"Loading embedder (Attempt {i+1})...")
            broadcaster_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            break
        except Exception as e:
            print(f"Embedder Load Error: {e}. Retrying...")
            time.sleep(2)
            
    if not broadcaster_model:
        print("Warning: Failed to load embedder. Dashboard visualization may be limited.")
    
    def broadcast_state(data):
        """Atomic write to prevent race conditions in specific dashboard readers."""
        temp_file = "dashboard_state.tmp"
        target_file = "dashboard_state.json"
        try:
            with open(temp_file, "w") as f:
                json.dump(data, f)
            os.replace(temp_file, target_file)
        except Exception as e:
            print(f"Broadcast Error: {e}")

    for item in suite:
        prompt = item['prompt']
        domain = item['domain']
        
        # Broadcast Start State
        broadcast_state({
            "status": "processing",
            "id": item['id'],
            "domain": domain,
            "prompt": prompt,
            "embedding": list(broadcaster_model.encode([prompt])[0].astype(float)),
            "timestamp": time.time()
        })
        
        response, lat = get_completion([{"role": "user", "content": prompt}])
        
        if response:
            success_count += 1
            total_latency += lat
            
            # Count steps
            step_list = re.split(r'[.\n]+', response)
            step_list = [s.strip() for s in step_list if len(s.strip()) > 10] # Filter empty/short
            steps = len(step_list)
            total_steps += steps
            
            preview = response[:30].replace('\n', ' ') + "..."
            print(f"{item['id']:<4} | {domain:<10} | {lat:6.2f} | {steps:<5} | {preview}")
            
            # --- BROADCAST STEPS ANIMATION ---
            # We want to visualize the flow of thought.
            print(f"    > Replaying {steps} steps to Braintop...")
            for idx, step_text in enumerate(step_list):
                # Calculate embedding for this specific step
                step_emb = None
                if broadcaster_model:
                    try:
                        step_emb = list(broadcaster_model.encode([step_text])[0].astype(float))
                    except: pass
                
                broadcast_state({
                    "status": "processing", # Keep 'processing' to show active thinking
                    "id": item['id'],
                    "domain": domain,
                    "prompt": f"[Step {idx+1}/{steps}] {step_text}",
                    "embedding": step_emb,
                    "timestamp": time.time()
                })
                time.sleep(1.5) # Pace the animation (1.5s per step)
            
            # Final Result State
            broadcast_state({
                "status": "complete",
                "id": item['id'],
                "domain": domain,
                "prompt": f"FINAL ANSWER: {preview}",
                "response_preview": preview,
                "latency": lat,
                "steps": steps,
                "embedding": list(broadcaster_model.encode([response[:100]])[0].astype(float)) if broadcaster_model else None,
                "timestamp": time.time()
            })
            time.sleep(2.0) # Pause before next question
            
        else:
            print(f"{item['id']:<4} | {domain:<10} | FAILED | -     | -")
            
    # Final cleanup
    broadcast_state({"status": "idle", "prompt": "Eval Completed.", "embedding": None})


    print("\n=== CoT Eval Summary (x200) ===")
    print(f"Success Rate: {success_count/NUM_TESTS*100:.1f}%")
    print(f"Avg Latency:  {total_latency/max(1, success_count):.2f}s")
    print(f"Avg Steps:    {total_steps/max(1, success_count):.1f}")

if __name__ == "__main__":
    run_eval()
