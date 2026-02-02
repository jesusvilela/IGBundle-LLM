from unsloth import FastLanguageModel
import torch

def export_gguf():
    # Load Merged Model
    # Load V2 Checkpoint 300 (User Specified)
    model_path = r"output/qwen25_7b_igbundle_v2/checkpoint-300"
    print(f"Loading checkpoint from {model_path} for GGUF export...")
    
    # We explicitly trust remote code for Qwen
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        load_in_4bit=False, 
        trust_remote_code=True,
        device_map="cpu" # Crucial: CPU only to avoid OOM with running training
    )
    
    # Export to GGUF
    print("Exporting to GGUF (q4_k_m)...")
    # Using 'q4_k_m' as standard balanced quantization
    model.save_pretrained_gguf(
        "igbundle_v2_cp300", 
        tokenizer, 
        quantization_method="q4_k_m"
    )
    print("GGUF Export Complete: igbundle_v2_cp300-unsloth.gguf")
    print("GGUF Export Complete: igbundle_qwen7b_riemannian-unsloth.gguf")

if __name__ == "__main__":
    export_gguf()
