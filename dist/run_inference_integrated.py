import sys
import os
import argparse
import time

# Add the local library path
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, "igbundle_lib")
sys.path.append(lib_path)

def run_fast_inference(model_path, prompt, max_tokens=128):
    """Refined Mode (GGUF): Uses llama.cpp for high performance, no fiber adaptation."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: 'llama-cpp-python' is not installed. Required for 'fast' mode.")
        print("Install via: pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading GGUF model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1, # Offload all to GPU
        verbose=False
    )
    
    print(f"\nGenerating (Fast Mode)...\nPrompt: {prompt}\n")
    start = time.time()
    output = llm(
        prompt,
        max_tokens=max_tokens,
        echo=True
    )
    dt = time.time() - start
    
    result = output['choices'][0]['text']
    print(f"\n--- Result ({dt:.2f}s) ---\n{result}\n-----------------------")

def run_refined_inference(base_model_path, adapter_weight_path, prompt, max_tokens=128):
    """Refined Mode (PyTorch): Uses generic Transformers + Fiber Adapter for full capabilities."""
    print("Initializing Refined Mode (PyTorch + GeometricAdapter)...")
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from igbundle.modules.geometric_adapter import create_geometric_adapter, GeometricIGBundleAdapter
    from igbundle.core.config import IGBundleConfig
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Base Model (Quantized)
    print(f"Loading Base Model (4-bit)...")
    base_model_id = "Qwen/Qwen2.5-7B" # Fallback if path not provided, but we expect the user to have it
    # We assume 'base_model_path' here might be the HF repo name if not local, 
    # but for this package we might need to rely on the user having internet or the cache.
    # In the export script, we don't copy the whole HF model (too big), we rely on cache or path.
    # We'll use a hardcoded default or an arg.
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", 
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False # Required for some adapter gradients, but inference? Safe to keep false for correctness.
    
    # 2. Initialize Adapter
    print("Loading Geometric Fiber Adapter...")
    config = IGBundleConfig(
        hidden_size=3584,
        num_components=8,
        latent_dim=64,
        num_categories=16,
        use_dynamics=True
    )
    adapter = create_geometric_adapter(config).to(device)
    
    if os.path.exists(adapter_weight_path):
        adapter.load_state_dict(torch.load(adapter_weight_path), strict=False)
        print("Adapter weights loaded successfully.")
    else:
        print(f"Warning: Adapter weights not found at {adapter_weight_path}. Using random init.")

    # 3. Inject Hook (Effect Discipline)
    def adapter_hook(module, args, output):
        # args is typically (hidden_states, ...)
        # output is the result of vertical forward
        # We hook onto the last layer or specific layer.
        # Check training script; it replaces forward. Here we use register_forward_hook for simplicity if possible?
        # Training script replaced .forward on layer 12. We should match that.
        pass

    target_layer_idx = 12
    layers = model.model.layers
    target_layer = layers[target_layer_idx]
    original_forward = target_layer.forward
    
    def manual_hook(hidden_states, *args, **kwargs):
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype
        # Adapter Pass
        h_in = h.to(device).to(torch.float32)
        adapted_out, _ = adapter(h_in)
        adapted = adapted_out.to(orig_dtype)
        
        if isinstance(out, tuple):
             return (adapted,) + out[1:]
        return adapted
        
    target_layer.forward = manual_hook
    print(f"Adapter injected at layer {target_layer_idx}.")
    
    # 4. Generate
    print(f"\nGenerating (Refined Mode)...\nPrompt: {prompt}\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        start = time.time()
        # Refinement: We could call adapter.refine_latents() here if we had an interactive loop.
        # For this script, we just run the forward pass with the loaded latents.
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        dt = time.time() - start
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n--- Result ({dt:.2f}s) ---\n{result}\n-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGBundle Refined Inference Runner")
    parser.add_argument("--mode", choices=["fast", "refined"], default="refined", help="Inference mode")
    parser.add_argument("--prompt", type=str, default="Explain the geometry of a fiber bundle.", help="Input prompt")
    parser.add_argument("--gguf_model", type=str, default="base_model.gguf", help="Path to GGUF file")
    parser.add_argument("--adapter_path", type=str, default="adapter_refined.pt", help="Path to adapter weights")
    
    args = parser.parse_args()
    
    if args.mode == "fast":
        if not os.path.exists(args.gguf_model):
            print(f"Error: GGUF model not found at {args.gguf_model}")
            sys.exit(1)
        run_fast_inference(args.gguf_model, args.prompt)
    else:
        run_refined_inference(None, args.adapter_path, args.prompt)
