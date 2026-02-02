
from unsloth import FastLanguageModel
import torch
import os

def export_merged():
    # Load V2 Checkpoint 300
    model_path = r"output/qwen25_7b_igbundle_v2/checkpoint-300"
    output_dir = "output/igbundle_v2_cp300_merged"
    
    print(f"Loading checkpoint from {model_path}...")
    
    # Load model (CPU to be safe, or GPU if VRAM allows - Training is running so CPU implies RAM usage)
    # Using 'cpu' map might be slow but safe for VRAM. RAM might be tight (16GB+ needed).
    device_map = "cpu" 
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            load_in_4bit=False, 
            trust_remote_code=True,
            device_map=device_map
        )
        
        print(f"Merging and Saving to {output_dir}...")
        model.save_pretrained_merged(
            output_dir,
            tokenizer,
            save_method="merged_16bit", # Standard 16-bit export
        )
        print("Export Success!")
        
    except Exception as e:
        print(f"Export Failed: {e}")

if __name__ == "__main__":
    export_merged()
