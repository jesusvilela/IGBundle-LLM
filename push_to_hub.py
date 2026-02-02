
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def push_to_hub():
    model_path = "output/igbundle_qwen7b_riemannian_merged"
    repo_id = "jesusvilela/igbundle-qwen2.5-7b-riemannian"
    
    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Pushing to Hub: {repo_id}...")
    try:
        model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id, private=False)
        print("✅ Model pushed successfully!")

        from huggingface_hub import HfApi
        api = HfApi()
        print("Pushing README.md...")
        api.upload_file(
            path_or_fileobj="README_model_card.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ README pushed successfully!")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")

if __name__ == "__main__":
    push_to_hub()
