import os
import sys
import torch
import json
import time
import argparse
import random
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    SiglipVisionModel,
    SiglipImageProcessor
)

# Setup Paths
sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter

# Constants
BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
ADAPTER_PATH = "igbundle_unified_training/final/adapter_weights.pt"
DASHBOARD_STATE_FILE = "dashboard_state.json"

def update_dashboard(status, prompt, domain="Logic", embedding=None):
    """Broadcast state to Braintop Dashboard."""
    state = {
        "status": status,
        "prompt": prompt,
        "domain": domain,
        "timestamp": time.time(),
        "embedding": embedding.tolist() if embedding is not None else None
    }
    try:
        with open(DASHBOARD_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Dashboard Update Failed: {e}")

def load_gsm8k_samples(path, n=3):
    with open(path, "r") as f:
        data = json.load(f)
    # The file structure is {"gsm8k_cot_zeroshot": [ ... ]}
    samples = data.get("gsm8k_cot_zeroshot", [])[:n]
    parsed = []
    for s in samples:
        parsed.append({
            "type": "text",
            "prompt": s["doc"]["question"],
            "target": s["doc"]["answer"]
        })
    return parsed

def load_visual_samples():
    base_dir = "data/geometric_shapes"
    samples = [
        {
            "type": "visual",
            "image": os.path.join(base_dir, "physics_ball.png"),
            "prompt": "Describe the physics principles shown in this image, specifically regarding kinematics and gravity."
        },
        {
            "type": "visual",
            "image": os.path.join(base_dir, "geometric_torus.png"),
            "prompt": "Analyze the topology of this object. Is it simply connected? Explain using the concept of genus."
        }
    ]
    return samples

def setup_pipeline():
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
    
    print("Loading SigLIP Vision Model...")
    vision_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        device_map="cuda", 
        torch_dtype=torch.float16
    )
    
    print("Loading Geometric Adapter...")
    config = IGBundleConfig(
        hidden_size=3584,
        num_components=8,
        latent_dim=64,
        num_categories=16,
        use_dynamics=True,
        use_geodesic_attn=True,
        supported_modalities=["vision", "text"]
    )
    adapter = create_geometric_adapter(config).to("cuda")
    
    if os.path.exists(ADAPTER_PATH):
        print(f"Loading Weights: {ADAPTER_PATH}")
        state_dict = torch.load(ADAPTER_PATH)
        adapter.load_state_dict(state_dict)
    else:
        print("WARNING: ADAPTER NOT FOUND. Using Random Weights.")

    return llm, tokenizer, vision_model, vision_processor, adapter

def inject_adapter(llm, adapter, vision_feats_holder):
    target_layer_idx = 12
    layers = llm.model.layers
    target_layer = layers[target_layer_idx]
    original_forward = target_layer.forward
    
    def adapter_hook(hidden_states, *args, **kwargs):
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype
        
        h_in = h.to("cuda").to(torch.float32)
        
        vis_feats = None
        if vision_feats_holder['feats'] is not None:
             vis_feats = vision_feats_holder['feats'].to("cuda").to(torch.float32)

        with torch.no_grad():
             adapted_out, geo_state = adapter(h_in, pixel_values=vis_feats)
        
        # --- DASHBOARD HOOK ---
        # Extract mean activation from the manifold (x_proj or similar)
        # geo_state is a named tuple or object depending on implementation.
        # Assuming geo_state has 'x' (projected coordinates) or we use the output itself.
        # For visualization, we'll take the mean of the adapted output for now to represent 'thought state'.
        
        embedding = adapted_out.mean(dim=1).squeeze().cpu().numpy() # [Hidden]
        # Downsample for dashboard efficiency (if needed, but dashboard renderer handles PCA/UMAP usually)
        # Actually dashboard expects embedding compatible with 'all-MiniLM-L6-v2' (384 dim) or uses it for projection.
        # But wait, run_braintop_dashboard.py attempts to load sentence transformer if embedding is None.
        # If we provide embedding, BraintopRenderer.render_frame needs to handle it.
        # Ideally we project to 384 or something manageable. 
        # But let's just send a slice to simulate a "thought vector".
        
        if len(embedding.shape) > 0:
             dashboard_emb = embedding[:384] # Slice to match standard size roughly
             # Broadcast!
             # We only update periodically to avoid IO flooding
             if random.random() < 0.1: # 10% chance per token (approx every 10 tokens)
                  update_dashboard("active", "Thinking...", domain="Processing", embedding=dashboard_emb)

        adapted = adapted_out.to(orig_dtype)
        
        if isinstance(out, tuple):
            return (adapted,) + out[1:]
        return adapted
        
    target_layer.forward = adapter_hook
    print("Adapter Injected w/ Dashboard Hook.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_text", type=int, default=1)
    parser.add_argument("--n_vis", type=int, default=1)
    args = parser.parse_args()
    
    llm, tokenizer, vision_model, processor, adapter = setup_pipeline()
    
    vision_holder = {'feats': None}
    inject_adapter(llm, adapter, vision_holder)
    
    # Load Tasks
    text_tasks = load_gsm8k_samples("h:/LLM-MANIFOLD/debug_samples_gsm8k_cot_zeroshot.json", n=args.n_text)
    vis_tasks = load_visual_samples()[:args.n_vis]
    
    all_tasks = text_tasks + vis_tasks
    
    print(f"Starting Benchmark Suite: {len(all_tasks)} Tasks")
    
    for i, task in enumerate(all_tasks):
        print(f"\n--- Task {i+1} [{task['type'].upper()}] ---")
        prompt = task['prompt']
        print(f"Prompt: {prompt}")
        
        # Update Dashboard (Start)
        update_dashboard("active", prompt, domain="Logic" if task['type']=='text' else "Physics")
        
        # Vision Processing
        vision_holder['feats'] = None # Reset
        if task['type'] == 'visual':
            try:
                image_path = task['image']
                print(f"Loading Image: {image_path}")
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                pixel_values = inputs.pixel_values.to("cuda").to(torch.float16)
                with torch.no_grad():
                     vis_out = vision_model(pixel_values)
                     vision_holder['feats'] = vis_out.last_hidden_state
            except Exception as e:
                print(f"Vision Error: {e}")
        
        # Generation
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # We wrap generation in a try/finally to ensure status is updated
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    do_sample=True, 
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                 response = response[len(prompt):].strip()
                 
            print(f"Response: {response}")
            update_dashboard("complete", prompt, domain="Result")
            
        except Exception as e:
            print(f"Generation Error: {e}")
            
        time.sleep(2) # Pause for visual effect

if __name__ == "__main__":
    main()
