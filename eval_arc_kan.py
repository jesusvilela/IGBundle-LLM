
import os
import sys
import json
import glob
import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

# Setup Paths
sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter

# Constants
# We use the Base Model + Trained Adapter approach
BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
ADAPTER_PATH = "igbundle-llm/igbundle_unified_training/final/adapter_weights.pt"
CHECKPOINT_DIR = "igbundle-llm/igbundle_unified_training"

def find_latest_adapter():
    if os.path.exists("adapter_weights.pt"): return "adapter_weights.pt"
    if os.path.exists(ADAPTER_PATH): return ADAPTER_PATH
    
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
    if not checkpoints:
        # Debug fallback
        debug_path = os.path.join(CHECKPOINT_DIR, "debug_init", "adapter_weights.pt")
        if os.path.exists(debug_path): return debug_path
        raise FileNotFoundError("No checkpoints found.")
    
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(CHECKPOINT_DIR, latest, "adapter_weights.pt")

def setup_kan_pipeline(manifold_type="kan"):
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16
    )
    
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Initializing Geometric Adapter ({manifold_type})...")
    config = IGBundleConfig(
        hidden_size=3584, # Qwen 7B
        num_components=8,
        latent_dim=64,
        num_categories=16, 
        use_dynamics=True,
        use_geodesic_attn=True,
        manifold_type=manifold_type, # KAN ACTIVATION
        supported_modalities=["vision", "text"]
    )
    adapter = create_geometric_adapter(config).to("cuda")
    
    # Load Weights
    weight_path = find_latest_adapter()
    print(f"Loading Weights from: {weight_path}")
    state_dict = torch.load(weight_path)
    adapter.load_state_dict(state_dict, strict=False) # Strict=False because KAN might have new params?
    # KAN params (kan_flow) are not in the pretrained weights. 
    # They will be random initialized. This is expected for "Validation of Stability".
    # If we wanted PRETRAINED KAN, we'd need to train it.
    # For now, we test if the *structure* holds.
    
    return llm, tokenizer, adapter

def inject_adapter(llm, adapter):
    target_layer_idx = 12
    layers = llm.model.layers
    target_layer = layers[target_layer_idx]
    original_forward = target_layer.forward
    
    def adapter_hook(hidden_states, *args, **kwargs):
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype
        
        # Move to Float32 for Manifold Ops
        h_in = h.to("cuda").to(torch.float32)
        
        # Adapter Forward
        with torch.no_grad():
             # pass None for pixel_values
             adapted_out, _ = adapter(h_in, pixel_values=None)
        
        adapted = adapted_out.to(orig_dtype)
        
        if isinstance(out, tuple):
            return (adapted,) + out[1:]
        return adapted
        
    target_layer.forward = adapter_hook
    print("Adapter Injected at Layer 12.")

# --- ARC EVAL LOGIC ---
def format_grid(grid):
    return str(grid).replace(" ", "")

def create_prompt(task_data):
    prompt = "You are solving an ARC-AGI puzzle. Detect the pattern in the Input grids and generate the correct Output grid.\n\n"
    # 1-shot
    if len(task_data['train']) > 0:
        pair = task_data['train'][-1]
        prompt += f"Input:\n{format_grid(pair['input'])}\nOutput:\n{format_grid(pair['output'])}\n\n"
    
    test_case = task_data['test'][0] 
    prompt += f"Input:\n{format_grid(test_case['input'])}\nOutput:\n"
    return prompt, test_case['output']

def eval_arc_kan(data_dir, limit=None):
    # 1. Setup KAN
    llm, tokenizer, adapter = setup_kan_pipeline(manifold_type="kan")
    inject_adapter(llm, adapter)
    
    files = glob.glob(os.path.join(data_dir, "*.json"))
    if limit:
        files = files[:limit]
    
    possible_files = [f for f in files if os.path.exists(f)]
    if not possible_files:
        print(f"No files found in {data_dir}")
        return

    print(f"Running KAN-ARC Evaluation on {len(possible_files)} tasks...")
    
    correct = 0
    total = 0
    valid_syntax = 0
    results = []

    for fpath in tqdm(possible_files):
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        prompt, ground_truth = create_prompt(data)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        try:
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False, # Deterministic for eval
                    pad_token_id=tokenizer.eos_token_id,
                    stop_strings=["\n\n", "Input:", "Output:"], 
                    tokenizer=tokenizer
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction_text = generated_text[len(prompt):].strip()
            prediction_text = prediction_text.split("Input:")[0].strip()
            
            # Sanity Check KAN didn't explode
            if "NaN" in prediction_text or len(prediction_text) == 0:
                print(f"Warning: Model output suspicious for {os.path.basename(fpath)}")
            
            is_correct = False
            is_valid = False
            try:
                pred_grid = json.loads(prediction_text)
                if isinstance(pred_grid, list):
                    is_valid = True
                    if pred_grid == ground_truth:
                        is_correct = True
                        correct += 1
            except:
                pass
                
            if is_valid: valid_syntax += 1
            total += 1
            
            results.append({
                "file": os.path.basename(fpath),
                "correct": is_correct,
                "valid": is_valid,
                "pred": prediction_text
            })
            
        except Exception as e:
            print(f"Error executing task {fpath}: {e}")
            
    print("-" * 30)
    print(f"KAN-ARC Results (Limit {limit}):")
    print(f"Total: {total}")
    print(f"Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"Valid Syntax: {valid_syntax} ({valid_syntax/total*100:.2f}%)")
    
    with open("kan_arc_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Test limit")
    args = parser.parse_args()
    
    eval_arc_kan("igbundle-llm/data/arc_agi", args.limit)
