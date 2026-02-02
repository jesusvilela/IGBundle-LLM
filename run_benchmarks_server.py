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
from transformers import Qwen2TokenizerFast
from lm_eval.models.openai_completions import OpenAICompletionsAPI

# Monkey patch OpenAICompletions to use max_tokens=1 instead of 0 for loglikelihood
# This is required for llama-cpp-python server compatibility
original_create_payload = OpenAICompletionsAPI._create_payload

def patched_create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
    payload = original_create_payload(self, messages, generate, gen_kwargs, seed, eos, **kwargs)
    
    # --- BRAINTOP TELEMETRY INJECTION ---
    try:
        # We broadcast the current benchmarking prompt to the dashboard
        # This allows visualization of MMLU/GSM8K solving in real-time
        prompt_content = None
        if isinstance(messages, str): prompt_content = messages
        elif isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict) and 'content' in messages[0]:
                prompt_content = messages[-1]['content'] # Last user message
            else:
                prompt_content = str(messages)
        
        if prompt_content:
            # Atomic Write
            t_file = "dashboard_state_lmstudio.tmp"
            r_file = "dashboard_state_lmstudio.json"
            with open(t_file, "w") as f:
                json.dump({
                    "status": "processing",
                    "id": "BENCH",
                    "domain": "Benchmark",
                    "prompt": prompt_content[:200] + "...", # Truncate for display
                    "embedding": None, # Dashboard will calc or skip
                    "timestamp": time.time()
                }, f)
            os.replace(t_file, r_file)
    except Exception as e:
        pass # Silent fail to not interrupt benchmarks
    # ------------------------------------

    if not generate and payload.get("echo") is True:
        # Loglikelihood request
        # OpenAI uses max_tokens=0 to mean "echo prompt", but llama-cpp-python needs 1 to generate at least one token?
        # Actually, if we set it to 1, we get prompt + 1 token. 
        # lm_eval slices [:-1], so it works perfectly.
        if payload.get("max_tokens") == 0:
            payload["max_tokens"] = 1
            
        # CRITICAL FIX: Ensure logprobs are requested for llama-cpp-python
        # Otherwise it returns None and crashes lm_eval
        if not payload.get("logprobs"):
            payload["logprobs"] = 1

    # Patch for llama-cpp-python server: it doesn't accept tokens in prompt, only strings
    prompt = payload.get("prompt")
    if prompt and isinstance(prompt, list):
        if len(prompt) > 0:
            first = prompt[0]
            if isinstance(first, int):
                # Single sequence of tokens -> decode to string
                payload["prompt"] = self.tokenizer.decode(prompt)
            elif isinstance(first, list) and len(first) > 0 and isinstance(first[0], int):
                # Batch of token sequences -> batch decode to list of strings
                payload["prompt"] = self.tokenizer.batch_decode(prompt)
                
    # Debug logging for validation errors
    # print(f"DEBUG PAYLOAD: {json.dumps(payload, default=str)}")
    
    # Ensure 'stop' is a list or string, not None
    if payload.get("stop") is None:
        payload.pop("stop", None)
        
    return payload

OpenAICompletionsAPI._create_payload = patched_create_payload
print("Patched OpenAICompletionsAPI._create_payload to use max_tokens=1 and Braintop Telemetry.")


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
    @property
    def eot_token(self):
        return self.eos_token_id
    Qwen2TokenizerFast.eot_token = eot_token

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

# Global context
CURRENT_BENCHMARK = None

# benchmark specific chat skeletons (Jinja2-like or simple format)
# This maps the benchmark name to a function that wraps the raw lm_eval prompt
BENCHMARK_SKELETONS = {
    "gsm8k": lambda p: f"<|im_start|>system\nYou are a helpful assistant. Solve the math problem step by step.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
    "mmlu": lambda p: p, # MMLU usually works best with raw few-shot prompts
    "aime25": lambda p: f"<|im_start|>system\nYou are a genius mathematician. Solve the problem.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
    "arc_challenge": lambda p: f"<|im_start|>system\nYou are an intelligent puzzle solver.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
    # Default fallback
    "default": lambda p: p 
}

def apply_skeleton(prompt, benchmark_name):
    skeleton_func = BENCHMARK_SKELETONS.get(benchmark_name, BENCHMARK_SKELETONS["default"])
    # Some heuristics to prevent double-templating if lm_eval already formatted it (unlikely for openai-completions)
    if "<|im_start|>" in prompt:
        return prompt
    return skeleton_func(prompt)

def save_incremental_results(results):
    try:
        if os.path.exists("server_benchmark_results.json"):
            with open("server_benchmark_results.json", "r") as f:
                existing = json.load(f)
        else:
            existing = {"results": {}}
        
        # Merge
        if "results" in results:
            existing["results"].update(results["results"])
            
        with open("server_benchmark_results.json", "w") as f:
            json.dump(existing, f, indent=2, default=str)
    except Exception as e:
        print(f"Failed to save incremental results: {e}")

def main():
    global CURRENT_BENCHMARK
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--server_path", required=False) 
    parser.add_argument("--base_url", required=False, help="Existing server URL")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu"])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    
    # Task mappings
    task_map = {
        "mmlu-pro": "mmlu", "mmlu": "mmlu",
        "aime": "aime25", "aime25": "aime25",
        "gpqa": "leaderboard_gpqa_main", "arc": "arc_challenge",
        "human-eval": "humaneval", "gsm8k": "gsm8k_cot_zeroshot"
    }

    selected_tasks = []
    for b in args.benchmarks:
        selected_tasks.append(task_map.get(b, b))
            
    print(f"Selected tasks: {selected_tasks}")

    # Reset Monitor State
    try:
        init_state = {
            "tps": 0.0,
            "latest_q": "Initializing benchmarks...",
            "latest_a": "Please wait...",
            "timestamp": time.time()
        }
        with open("monitor_state_lmstudio.json", "w") as f:
            json.dump(init_state, f)
    except: pass

    server_process = None
    
    # Server Launch Logic
    if not args.base_url:
        port = find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        print(f"Starting llama-cpp-python server on port {port} for GGUF/Model...")
        
        # Args for server
        server_cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", args.model_path, "--port", str(port),
            "--n_gpu_layers", "99", "--n_ctx", "4096",
            "--n_threads", "4", "--n_batch", "128"
        ]
        
        server_process = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr) 
        
        print("Waiting for server...")
        for i in range(120):
            try:
                if requests.get(f"{base_url}/v1/models", timeout=1).status_code == 200:
                    print("Server Ready.")
                    break
            except: pass
            time.sleep(1)
    else:
        base_url = args.base_url
            
    try:
        # Normalize Base URL
        if base_url.endswith("/"):
            base_url = base_url[:-1]
            
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
            
        # Set Env Vars for openai client
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_API_KEY"] = "sk-ollama"
        
        # --- BRAINTOP TELEMETRY (LM BASE CLASS PATCH) ---
        from lm_eval.api.model import LM
        if not hasattr(LM, "_braintop_patched"):
            original_loglikelihood = LM.loglikelihood
            
            def patched_loglikelihood(self, requests, disable_tqdm=False):
                try:
                    if requests and len(requests) > 0:
                        # Inspect first request tuple: (context, continuation)
                        ctx = requests[0][0] if isinstance(requests[0], tuple) else str(requests[0])
                        t_file = "dashboard_state_lmstudio.tmp"
                        r_file = "dashboard_state_lmstudio.json"
                        with open(t_file, "w") as f:
                            json.dump({
                                "status": "processing",
                                "id": "BENCH_LL",
                                "domain": f"Benchmark ({CURRENT_BENCHMARK})",
                                "prompt": ctx[-200:], 
                                "embedding": None,
                                "timestamp": time.time()
                            }, f)
                        os.replace(t_file, r_file)
                except: pass
                return original_loglikelihood(self, requests, disable_tqdm)
            
            LM.loglikelihood = patched_loglikelihood
            LM._braintop_patched = True
            print("Patched lm_eval.api.model.LM for Braintop Telemetry.")
        # ------------------------------------------------

        # Backend Selection
        if args.model_path.lower().endswith(".gguf") and os.path.isfile(args.model_path):
            print(f"[GGUF] Using 'gguf' backend via Local Server.")
            backend = "gguf"
            
            # Fix: sse_starlette/gguf backend likely appends /v1 itself
            # We should pass the root URL (e.g. http://localhost:1234)
            root_url = base_url
            if root_url.endswith("/v1"):
                root_url = root_url[:-3]
                
            model_args = f"base_url={root_url}"
        else:
            # Fallback to local-completions for other models (non-GGUF or remote)
            backend = "local-completions"
            compl_url = f"{base_url}/completions"
            model_args = f"model={args.model_path},base_url={compl_url},api_base={compl_url},tokenizer=Qwen/Qwen2.5-7B,num_concurrent=1,tokenized_requests=False" 
 
        # Define tasks
        # Use the user-selected tasks from args
        print(f"Executing tasks: {selected_tasks}")
        
        for task in selected_tasks:
            CURRENT_BENCHMARK = task
            print(f"\nStarting Task: {task}")
            
            try:
                # Use selected backend
                results = simple_evaluate(
                    model=backend,
                    model_args=model_args,
                    tasks=[task],
                    num_fewshot=0,
                    batch_size=1, # FORCE batch size 1
                    limit=args.limit,
                    log_samples=True,
                    cache_requests=False 
                )
                print(f"Task {task} Complete.")
                if results is not None:
                    save_incremental_results(results)
                    # Save Samples for Debugging
                    if "samples" in results:
                        try:
                            with open(f"debug_samples_{task}.json", "w") as f:
                                json.dump(results["samples"], f, indent=2, default=str)
                            print(f"Saved samples to debug_samples_{task}.json")
                        except Exception as e:
                            print(f"Failed to save samples: {e}")
                else:
                    print(f"Warning: Task {task} returned No Results.")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error evaluating {task}: {e}")
                if "NoneType" in str(e) and "subscriptable" in str(e):
                    print(f"CRITICAL: Task {task} likely failed due to missing 'logprobs' from server.")
                    print("Skipping to next task...")
                continue

    finally:
        if server_process:
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()