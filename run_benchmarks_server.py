import argparse
import json
import os
import sys
import time
import subprocess
import requests
import socket
from contextlib import closing

# Use standard lm_eval or subprocess to call it if import is tricky
import lm_eval
from lm_eval import simple_evaluate
# from lm_eval.models.gguf import GGUFLLM # verify import path
from transformers import Qwen2TokenizerFast

# Monkey patch for Qwen2TokenizerFast to support encode_batch which lm_eval expects
def encode_batch(self, text, **kwargs):
    # This is a naive implementation to satisfy lm_eval
    if isinstance(text, str):
        text = [text]
    return self.batch_encode_plus(text, **kwargs)['input_ids']

if not hasattr(Qwen2TokenizerFast, "encode_batch"):
    print("Patching Qwen2TokenizerFast with encode_batch...")
    Qwen2TokenizerFast.encode_batch = encode_batch

# Monkey patch eot_token
if not hasattr(Qwen2TokenizerFast, "eot_token"):
    print("Patching Qwen2TokenizerFast with eot_token...")
    # Map eot_token to eos_token_id (or actual token string if needed, but lm_eval often uses id or decoding of it)
    # The error says: self.tokenizer.decode([self.tokenizer.eot_token])
    # So eot_token should be an ID.
    @property
    def eot_token(self):
        return self.eos_token_id
    Qwen2TokenizerFast.eot_token = eot_token

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def wait_for_server(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(f"{url}/health")
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--server_path", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu"])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    
    # Task mappings
    task_map = {
        "mmlu-pro": "mmlu", # Fallback as mmlu-pro not found in grep
        "mmlu": "mmlu",
        "aime": "aime25",
        "aime25": "aime25",
        "gpqa": "leaderboard_gpqa_main", # Use leaderboard version (usually open)
        "arc": "arc_challenge",
        "human-eval": "humaneval", # might need checking
        "gsm8k": "gsm8k_cot_zeroshot"
    }

    selected_tasks = []
    for b in args.benchmarks:
        if b in task_map:
            selected_tasks.append(task_map[b])
        else:
            selected_tasks.append(b) # Try direct pass
            
    print(f"Selected tasks: {selected_tasks}")

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    
    print(f"Starting llama-server on port {port}...")
    server_cmd = [
        args.server_path,
        "-m", args.model_path,
        "--port", str(port),
        "-ngl", "99", # GPU layers
        "--ctx-size", "8192", # Context size
        "--parallel", "1"
    ]
    
    # Start server
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL, # Mute stdout/stderr to keep console clean? or KEEP for debug
        stderr=subprocess.PIPE
    ) # We might want to see stderr

    try:
        print("Waiting for server to become ready...")
        # Check /health or just wait for TCP
        ready = False
        for i in range(30):
            try:
                # llama.cpp server usually has /health or just /
                requests.get(f"{base_url}/health", timeout=1)
                ready = True
                break
            except:
                time.sleep(1)
        
        if not ready:
            print("Server failed to start or is not responding.")
            # Read stderr
            # print(server_process.stderr.read())
            return

        print("Server is ready. Running benchmarks...")
        
        # We use 'local-chat-completions' or 'openai-completions' depending on lm_eval version
        # Current lm_eval might use 'openai' with base_url
        # base_url needs to be the FULL endpoint for openai-completions in this version of lm_eval
        # We spoof 'davinci-002' because lm_eval asserts model name must be in [babbage-002, davinci-002] for loglikelihood
        model_args = f"model=davinci-002,base_url={base_url}/v1/completions,tokenizer=Qwen/Qwen2.5-7B" 
        # Note: tokenizer arg might be needed or it infers 'gpt2' if not set. 
        # But for 'openai', it often uses tiktoken.
        # Ideally we tell it to use the HuggingFace tokenizer.
        # But this is complicated.
        
        # Simplified: Just run MMLU
        # we might need to list tasks to find the right name
        
        results = simple_evaluate(
            model="openai-completions", # or 'openai-chat-completions'
            model_args=model_args,
            tasks=selected_tasks,
            limit=args.limit
        )
        
        print("\n=== RESULTS ===")
        print(json.dumps(results["results"], indent=2))
        
        with open("server_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
