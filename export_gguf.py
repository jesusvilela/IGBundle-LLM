import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse

def export_to_gguf_ready(base_model_id, checkpoint_path, output_dir):
    print(f"Loading base model: {base_model_id}")
    # We must load in float16 for merging, not 4bit
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    print(f"Loading adapter from {checkpoint_path}")
    try:
        model = PeftModel.from_pretrained(model, checkpoint_path)
    except Exception as e:
        print(f"Failed to load PEFT adapter: {e}")
        return

    print("Merging LoRA weights into Base Model...")
    # This merges the LoRA parameters (A*B) into the base weights
    model = model.merge_and_unload()
    
    # NOTE: IGBundle specific bottleneck weights (if separate from LoRA)
    # cannot be merged into the base Linear layers if they have non-linearities.
    # They are preserved in the topological structure but ignored by standard GGUF.
    # However, since we trained with LoRA, the LoRA weights capture the 'projection' 
    # effects influenced by the Sheaf Loss.
    
    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n=== GGUF CONVERSION INSTRUCTIONS ===")
    print(f"1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
    print(f"2. Install requirements: pip install -r llama.cpp/requirements.txt")
    print(f"3. Run conversion:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile {output_dir}/igbundle_qwen.gguf --outtype f16")
    print(f"4. Quantize (Optional):")
    print(f"   ./llama.cpp/llama-quantize {output_dir}/igbundle_qwen.gguf {output_dir}/igbundle_qwen_q4_k_m.gguf q4_k_m")
    
    # Helper script generator
    with open(os.path.join(output_dir, "convert.bat"), "w") as f:
        f.write(f"python ../../../llama.cpp/convert_hf_to_gguf.py . --outfile igbundle_qwen.gguf --outtype f16\n")
        f.write("pause")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint folder")
    parser.add_argument("--output", default="igbundle_merged")
    args = parser.parse_args()
    
    if os.path.exists(args.output):
        print(f"Warning: Output directory {args.output} exists.")
    
    export_to_gguf_ready(args.base, args.checkpoint, args.output)
