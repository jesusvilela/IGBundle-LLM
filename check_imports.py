try:
    import lm_eval
    print("lm_eval imported successfully")
except ImportError as e:
    print(f"lm_eval import failed: {e}")

try:
    import llama_cpp
    print("llama_cpp imported successfully")
except ImportError as e:
    print(f"llama_cpp import failed: {e}")
