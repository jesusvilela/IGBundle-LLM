
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "unsloth/Qwen2.5-7B-Instruct"
print(f"Testing load of {model_id}...")

try:
    # Try loading just the config first
    print("Loading config...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Try loading model (cpu is fine for integrity check, but might be slow)
    # We'll use device_map="auto" to use GPU if available for speed
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("✅ Model loaded successfully. Integrity appears OK.")
except Exception as e:
    print(f"❌ Load failed: {e}")
    
    # If failed, try to find the cache path to suggest deletion
    from transformers.utils.hub import TRANSFORMERS_CACHE
    print(f"Cache location: {TRANSFORMERS_CACHE}")
    # In Windows it might be different, let's print typical huggingface home
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
