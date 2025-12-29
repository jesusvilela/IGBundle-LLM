import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse

def export_to_gguf_ready(base_model_id, checkpoint_path, output_dir):
    print(f"Loading Base: {base_model_id} (CPU)")
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    import subprocess
    
    # 1. Load Base Model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu",
        torch_dtype=torch.float16, # Use float16 to save RAM
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # 2. Load Adapter
    print(f"Loading Adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path, device_map="cpu")
    
    # 3. Merge
    print("Merging adapters...")
    model = model.merge_and_unload()
    
    # 4. Save Merged HF Model
    merged_dir = "igbundle_merged_hf"
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    # 5. Convert to GGUF using local script
    print("Converting to GGUF using llama.cpp/convert_hf_to_gguf.py...")
    gguf_file = f"{output_dir}.gguf" # e.g. igbundle_qwen7b.gguf
    
    # Use sys.executable to ensure we use the same environment
    cmd = [
        sys.executable, "llama.cpp/convert_hf_to_gguf.py", 
        merged_dir, 
        "--outfile", gguf_file, 
        "--outtype", "f16"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"Success! GGUF saved to {gguf_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint folder")
    parser.add_argument("--output", default="igbundle_qwen7b") # Filename without extension
    args = parser.parse_args()
    
    export_to_gguf_ready(args.base, args.checkpoint, args.output)
