
import requests
import json
import time

# Configuration
API_URL = "http://192.168.56.1:1234/v1/chat/completions"
# Alternate URL if the above fails
# API_URL = "http://localhost:1234/v1/chat/completions" 

MODEL_ID = "ig-bundlellmv2-hamiltonians" # As specified by user

def test_prompt(name, system_prompt, user_prompt):
    print(f"\n--- Testing: {name} ---")
    print(f"User > {user_prompt}")
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    try:
        start_t = time.time()
        response = requests.post(API_URL, json=payload)
        end_t = time.time()
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"Assistant > {content}")
            print(f"[Time: {end_t - start_t:.2f}s]")
            return content
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Connection Failed: {e}")
        return None

if __name__ == "__main__":
    print(f"Connecting to LM Studio at {API_URL}...")
    
    # 1. Stability Check (Basic)
    test_prompt(
        "Stability Check",
        "You are a helpful assistant.",
        "Hello! Who are you and what can you do?"
    )
    
    # 2. Geometric/Thesis Principle Check (Metric Awareness)
    test_prompt(
        "Thesis Check: Geometric Intuition",
        "You are a rigorous mathematician specializing in Riemannian geometry.",
        "Explain the concept of 'Parallel Transport' using a metaphor involving a shield on a curved planet. How does the curvature affect the shield's orientation when it returns to the start?"
    )
    
    # 3. Logic/Reasoning (Chain of Thought)
    test_prompt(
        "Reasoning Check",
        "You are a logical solver.",
        "If I have a 3-liter jug and a 5-liter jug, how can I measure exactly 4 liters of water? precision and step-by-step."
    )
