import argparse
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector

# Representative MT-Bench Questions
MT_BENCH_QUESTIONS = [
    {"category": "reasoning", "question": "If I have 3 apples and eat one, how many oranges do I have?"},
    {"category": "coding", "question": "Write a Python function to find the nth Fibonacci number using recursion."},
    {"category": "roleplay", "question": "Act as a 19th-century pirate captain negotiating for more rum."},
    {"category": "writing", "question": "Draft a professional email declining a job offer politely."},
    {"category": "math", "question": "Solve for x: 2x + 5 = 15."},
    {"category": "extraction", "question": "Extract the names from this text: 'John and Mary went to Paris within 3 days.'"},
    {"category": "stem", "question": "Explain the concept of osmosis to a 5-year-old."},
    {"category": "humanities", "question": "What were the main causes of the French Revolution?"}
]

def generate_arena_answers(model_id, checkpoint_path):
    print(f"Loading Model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if checkpoint_path:
        print(f"Loading adapter: {checkpoint_path}")
        # Patch for IGBundle stats
        # We wrap it to see if we can extract sigma metrics during generation too
        class DictConfig:
            def __init__(self, d):
                for k,v in d.items(): setattr(self, k, v)
        
        # Default config assumption
        adapter_config = {
                'hidden_size': model.config.hidden_size,
                'num_components': 4,
                'num_categories': 16,
                'bottleneck_dim': 256,
                'latent_dim': 128,
                'adapter_scale': 0.1,
                'dropout': 0.05,
                'alpha': 1.0, 
                'beta': 1.0,
                'eta_f': 0.1,
                'eta_b': 0.01,
                'eta_w': 0.01
        }
        model = wrap_hf_candidate(model, DictConfig(adapter_config))
        
        # Load Weights
        ig_weights = os.path.join(checkpoint_path, "adapter_weights.pt")
        if os.path.exists(ig_weights):
            print("Loading IGBundle explicit weights...")
            model.load_state_dict(torch.load(ig_weights, map_location=device), strict=False)
            
        try:
            model = PeftModel.from_pretrained(model, checkpoint_path)
            # model = model.merge_and_unload() # Don't merge if we want to keep the wrapping dynamic
        except Exception as e:
            print(f"Peft load warning: {e}")

    model.eval()
    
    # Attach collector
    collector = StateCollector()
    collector.attach(model)
    
    results = []
    
    print("=== MT-Bench (Arena) Generation ===")
    for item in MT_BENCH_QUESTIONS:
        q = item['question']
        cat = item['category']
        print(f"\n[Category: {cat}] Q: {q}")
        
        # Format for Qwen (ChatML usually)
        # <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        
        collector.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            
        ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"A: {ans[:100]}...")
        
        # Metrics
        sigma = 0.0
        if collector.states:
            sigma = sum([s.sigma.mean().item() for s in collector.states]) / len(collector.states)
            print(f"  -> Curvature (Sigma): {sigma:.4f}")
            
        results.append({
            "question_id": len(results)+1,
            "category": cat,
            "prompt": q,
            "answer": ans,
            "sigma": sigma,
            "model_id": "igbundle-qwen-7b"
        })
        
    # Save
    with open("arena_answers.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved answers to arena_answers.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    
    generate_arena_answers("Qwen/Qwen2.5-7B", args.checkpoint)
