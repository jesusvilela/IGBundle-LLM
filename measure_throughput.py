import argparse
import subprocess
import time
import requests
import json
import sys
import os
import signal
from contextlib import contextmanager

# Configuration
PYTHON_EXE = sys.executable
SERVER_EXE = r"h:\LLM-MANIFOLD\igbundle-llm\llama.cpp\build\bin\Release\llama-server.exe"
MODEL_PATH = r"h:\LLM-MANIFOLD\igbundle-llm\igbundle_qwen7b_Q4_K_M.gguf"

TEST_PROMPT = "Write a very long and detailed essay about the history of artificial intelligence, covering its origins, the golden age, the AI winter, and the modern deep learning revolution. Ensure you generate at least 500 tokens."

CONFIGS = [
    {
        "name": "Baseline (Auto)",
        "args": ["--n-gpu-layers", "auto"]
    },
    {
        "name": "Max Offload",
        "args": ["--n-gpu-layers", "99"]
    },
    {
        "name": "Max Offload + FlashAttn",
        "args": ["--n-gpu-layers", "99", "--flash-attn", "on"]
    },
    {
        "name": "Optimized Batching",
        "args": ["--n-gpu-layers", "99", "--batch-size", "4096", "--ubatch-size", "1024"]
    }
]

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def wait_for_server(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/health", timeout=1)
            return True
        except:
            time.sleep(0.5)
    return False

def run_test(config):
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    
    cmd = [
        PYTHON_EXE, "-m", "llama_cpp.server",
        "--model", MODEL_PATH,
        "--port", str(port),
        "--n_ctx", "4096"
    ] + config["args"]
    
    print(f"--- Running Config: {config['name']} ---")
    print(f"Cmd: {' '.join(cmd)}")
    
    # Start Server
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
    except FileNotFoundError:
        print(f"Error: Server executable not found at {SERVER_EXE}")
        return None

    try:
        if not wait_for_server(base_url):
            print("Server failed to allow connection.")
            return None
        
        # Warmup with retry for loading
        print("Warming up...")
        for _ in range(30): # Wait up to 30*2 = 60s for load
            resp = requests.post(f"{base_url}/completion", json={"prompt": "Warmup", "n_predict": 10})
            if resp.status_code == 503:
                time.sleep(2)
                continue
            elif resp.status_code == 200:
                break
            else:
                print(f"Warmup failed: {resp.status_code}")
                return None
        else:
            print("Model failed to load in time.")
            return None
        
        # Test
        payload = {
            "prompt": TEST_PROMPT,
            "n_predict": 500,
            "temperature": 0.0,
            "stream": False
        }
        
        start_time = time.time()
        resp = requests.post(f"{base_url}/completion", json=payload)
        end_time = time.time()
        
        if resp.status_code == 200:
            data = resp.json()
            # Llama.cpp returns timings in the response typically if not disabled
            # But measuring wall clock client side is safer for 'user experience' speed
            
            # Extract content length roughly
            content = data.get("content", "")
            # Or use 'timings' if available
            timings = data.get("timings", {})
            predicted_ms = timings.get("predicted_ms", 0)
            predicted_n = timings.get("predicted_n", 0)
            
            if predicted_ms > 0 and predicted_n > 0:
                tps = predicted_n / (predicted_ms / 1000.0)
                print(f"Server Reported Speed: {tps:.2f} t/s")
                return tps
            else:
                # Fallback calculation
                duration = end_time - start_time
                # Estimate tokens (rude)
                tokens = len(content) / 3.0 # Rough est
                tps = tokens / duration
                print(f"Client Estimated Speed: {tps:.2f} t/s (approx)")
                return tps
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
            return None
            
    finally:
        proc.terminate()
        proc.wait()

def main():
    results = {}
    for conf in CONFIGS:
        tps = run_test(conf)
        if tps:
            results[conf["name"]] = tps
            
    print("\n=== FINAL RESULTS ===")
    winner = None
    max_tps = 0
    for name, tps in results.items():
        print(f"{name}: {tps:.2f} t/s")
        if tps > max_tps:
            max_tps = tps
            winner = name
            
    if winner:
        print(f"\nWinner: {winner} ({max_tps:.2f} t/s)")
        # Ideally save this recommendation
        with open("throughput_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
