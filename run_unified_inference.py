import os
import sys
import torch
import argparse
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    SiglipVisionModel,
    SiglipImageProcessor
)
import warnings
warnings.filterwarnings("ignore")

# Setup Paths
sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter, GeometricState

# Constants
BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
ADAPTER_PATH = "igbundle_unified_training/final/adapter_weights.pt"
CHECKPOINT_DIR = "igbundle_unified_training"

def find_latest_adapter():
    # 0. Check Local (Deployment Mode)
    if os.path.exists("adapter_weights.pt"):
        return "adapter_weights.pt"

    # 1. Check Training Path
    if os.path.exists(ADAPTER_PATH):
        return ADAPTER_PATH
    
    # Else find latest checkpoint
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
    if not checkpoints:
        # Final Fallback
        debug_path = os.path.join(CHECKPOINT_DIR, "debug_init", "adapter_weights.pt")
        if os.path.exists(debug_path):
            print("WARNING: Using DEBUG INITIALIZED weights.")
            return debug_path
        raise FileNotFoundError("No checkpoints found in igbundle_unified_training")
    
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(CHECKPOINT_DIR, latest, "adapter_weights.pt")

    # Final Fallback
    debug_path = os.path.join(CHECKPOINT_DIR, "debug_init", "adapter_weights.pt")
    if os.path.exists(debug_path):
        print("WARNING: Using DEBUG INITIALIZED weights.")
        return debug_path
        
    raise FileNotFoundError("No checkpoints found in igbundle_unified_training")

def setup_pipeline():
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16 # Use float16 for inference
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
        hidden_size=3584, # Qwen 7B
        num_components=8,
        latent_dim=64,
        num_categories=16,
        use_dynamics=True,
        use_geodesic_attn=True,
        supported_modalities=["vision", "text"]
    )
    adapter = create_geometric_adapter(config).to("cuda")
    
    # Load Weights
    weight_path = find_latest_adapter()
    print(f"Loading Adapter Weights from: {weight_path}")
    state_dict = torch.load(weight_path)
    adapter.load_state_dict(state_dict)
    
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
        
        # Move to Float32 for Manifold Ops
        h_in = h.to("cuda").to(torch.float32)
        
        # Get Vision Features
        vis_feats = None
        if vision_feats_holder['feats'] is not None:
             vis_feats = vision_feats_holder['feats'].to("cuda").to(torch.float32)

        # Adapter Forward
        with torch.no_grad():
             adapted_out, _ = adapter(h_in, pixel_values=vis_feats)
        
        adapted = adapted_out.to(orig_dtype)
        
        if isinstance(out, tuple):
            return (adapted,) + out[1:]
        return adapted
        
    target_layer.forward = adapter_hook
    print("Adapter Injected at Layer 12.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text query")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()
    
    llm, tokenizer, vision_model, processor, adapter = setup_pipeline()
    
    # Shared state for hook
    vision_holder = {'feats': None}
    inject_adapter(llm, adapter, vision_holder)
    
    # Process Image if present
    if args.image:
        print(f"Processing Image: {args.image}")
        try:
            image = Image.open(args.image).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to("cuda").to(torch.float16)
            with torch.no_grad():
                 vis_out = vision_model(pixel_values)
                 vision_holder['feats'] = vis_out.last_hidden_state # (1, N, 1152)
            print("Vision Features Extracted.")
        except Exception as e:
            print(f"Error loading image: {e}")
            
    # Generate
    print(f"\nQuery: {args.text}")
    print("-" * 40)
    
    inputs = tokenizer(args.text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=args.max_tokens, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strip prompt
    if response.startswith(args.text):
        response = response[len(args.text):]
        
    print(f"Response: {response.strip()}")

if __name__ == "__main__":
    main()
