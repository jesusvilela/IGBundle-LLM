
import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    base_model_id = "unsloth/Qwen2.5-7B-Instruct"
    adapter_path = "output/igbundle_qwen7b_riemannian/checkpoint-100"
    output_path = "output/igbundle_qwen7b_riemannian_merged"

    print(f"Loading base model: {base_model_id}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu", # Load on CPU to avoid OOM before merge if possible, or "auto" if enough VRAM
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print(f"Loading adapter from: {adapter_path}")
    if not os.path.exists(adapter_path):
        print(f"Error: Adapter path {adapter_path} does not exist.")
        # Fallback to checkpoint-600 if 100 is wrong, but we confirmed 100 exists
        return

    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return

    print("Merging adapter...")
    model = model.merge_and_unload()
    
    # Cast to bfloat16 explicitly to ensure consistency
    model = model.to(dtype=torch.bfloat16)

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Merge complete.")

if __name__ == "__main__":
    main()
