from safetensors.torch import load_file
import sys

def list_keys(path):
    print(f"Keys in {path}:")
    sd = load_file(path)
    for k in sorted(sd.keys()):
        print(f"  {k} - {sd[k].shape}")

if __name__ == "__main__":
    list_keys("h:/LLM-MANIFOLD/igbundle-llm/output/igbundle_qwen7b/checkpoint-50/adapter_model.safetensors")
