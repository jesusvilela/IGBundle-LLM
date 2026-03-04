import argparse
import gc
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser(description="Run Neural Glass qualitative salvo probes")
parser.add_argument("--max-tokens", type=int, default=512,
                    help="Max new tokens per generation (default: 512)")
args = parser.parse_args()

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Fast 4-bit load to fit in VRAM along with any adapters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
# We apply CPU swapping just in case the memory gets tight
max_memory = {0: "5GiB", "cpu": "30GiB"}
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory,
    trust_remote_code=True
)

# Apply simple ChatML template
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"

# For IGBundle adapter (Optional but the endpoint uses it so we try to load if present)
checkpoint_path = "./igbundle_phase9_odyssey/checkpoint-3000"
if os.path.exists("./igbundle_phase9_odyssey/checkpoint-3000"):
    checkpoint_path = "./igbundle_phase9_odyssey/checkpoint-3000"

try:
    from igbundle.integrations.hf_patch import wrap_hf_candidate
    from igbundle.core.config import IGBundleConfig

    # Try to load the trained geometric adapter if available
    adapter_paths = [
        os.path.join(checkpoint_path, "geometric_adapter_weights.pt"),
        os.path.join(checkpoint_path, "adapter_weights.pt")
    ]

    for path in adapter_paths:
        if os.path.exists(path):
            print(f"Loading IGBundle Geometric Adapter from {path}...")
            # Configure Adapter (defaults matched from app.py)
            cfg = {
                "hidden_size": model.config.hidden_size,
                "num_components": 8,
                "latent_dim": 64,
                "num_categories": 16,
                "use_dynamics": True,
                "use_geodesic_attn": True
            }
            class DictConfig:
                def __init__(self, d):
                    for k,v in d.items(): setattr(self, k, v)

            model = wrap_hf_candidate(model, DictConfig(cfg))
            model.load_state_dict(torch.load(path, map_location=model.device), strict=False)
            print("Adapter Loaded Successfully.")
            break
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Warning: IGBundle not found or failed to load. Running Base Qwen2.5-7B. Error: {e}")

model.eval()


def get_vram_usage():
    """Return current VRAM usage string."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return f"VRAM: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "VRAM: N/A (no CUDA)"


salvos = {
    "Salvo 1 - Abstract Reasoning": [
        "A sequence follows this rule: each term is the sum of the digits of the previous term squared. Starting from 39, what are the next 5 terms?",
        "If all Bloops are Razzles, and all Razzles are Lazzles, but no Lazzles are Wazzles, what can we conclude about Bloops and Wazzles?",
        "A 4x4 grid has numbers 1-16 placed so each row, column, and 2x2 quadrant sums to the same value. What is that value, and what constraint does it place on the center 2x2?"
    ],
    "Salvo 2 - Multi-step Math": [
        "Maria has 3x as many apples as João. After João buys 12 more and Maria gives away a third of hers, they have the same amount. How many did Maria start with?",
        "A train travels from A to B at 80 km/h and returns at 120 km/h. What is the average speed for the round trip, and why is it not 100 km/h?",
        "If you invest €1,000 at 5% annual compound interest, how many years until it doubles? Give the exact formula and an approximation."
    ],
    "Salvo 3 - MMLU Knowledge": [
        "What is the difference between a tensor, a vector, and a scalar in the context of differential geometry? Give a concrete example of each.",
        "Explain why the Fisher information matrix is always positive semi-definite. What does it mean when it's singular?",
        "What is the second law of thermodynamics, and how does it relate to the arrow of time? Name one system that appears to violate it and explain why it doesn't."
    ],
    "Salvo 4 - Multi-hop Reasoning": [
        "The philosopher who wrote the Critique of Pure Reason was born in the same city where he died. That city is now in which country, and what language is spoken there today?",
        "The element with atomic number 79 shares a group in the periodic table with the element used in most electrical wiring. What are both elements and what property do they share?",
        "The author of Brave New World was the grandson of a scientist famous for defending a controversial 19th century theory. What was that theory, and who originally proposed it?"
    ],
    "Salvo 5 - Stress Tests": [
        "Describe consciousness from the perspective of information theory, thermodynamics, and differential geometry simultaneously. Where do the three frameworks agree?",
        "Write a step-by-step proof that there are infinitely many primes, then explain the proof as if speaking to a 10-year-old, then to a mathematics PhD student.",
        "What would a valid counterexample to the Riemann Hypothesis look like? Why would finding one be catastrophic for number theory?"
    ]
}

os.makedirs("salvos_results", exist_ok=True)
output_file = "salvos_results/probe_results.md"

prompt_count = 0
total_prompts = sum(len(v) for v in salvos.values())

with open(output_file, "w", encoding="utf-8") as f:
    f.write("# Neural Glass Qualitative Probe\n\n")

    for salvo_name, prompts in salvos.items():
        print(f"\n--- {salvo_name} ---")
        f.write(f"## {salvo_name}\n\n")

        for prompt in prompts:
             prompt_count += 1
             print(f"[{prompt_count}/{total_prompts}] Prompt: {prompt[:80]}...")
             f.write(f"**Prompt:** {prompt}\n\n")

             messages = [
                 {"role": "system", "content": "You are Neural Glass, an advanced neurosymbolic AI assistant. Think step-by-step and provide clear, precise answers."},
                 {"role": "user", "content": prompt}
             ]

             try:
                 prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

                 with torch.no_grad():
                      outputs = model.generate(
                          **inputs,
                          max_new_tokens=args.max_tokens,
                          temperature=0.7,
                          top_p=0.9,
                          repetition_penalty=1.05,
                          pad_token_id=tokenizer.eos_token_id
                      )

                 response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
                 response_text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

                 f.write(f"**Response:**\n{response_text}\n\n---\n\n")
                 print(f"  Generated ({len(response_tokens)} tokens). {get_vram_usage()}")

                 # Cleanup tensors and KV cache between prompts
                 del outputs, response_tokens, inputs
                 torch.cuda.empty_cache()
                 gc.collect()

             except torch.cuda.OutOfMemoryError:
                 print(f"  CUDA OOM on prompt {prompt_count}! Clearing cache and continuing...")
                 f.write("**Response:** [CUDA OOM — skipped]\n\n---\n\n")
                 torch.cuda.empty_cache()
                 gc.collect()
             except Exception as e:
                 print(f"  Error on prompt {prompt_count}: {e}")
                 f.write(f"**Response:** [Error: {e}]\n\n---\n\n")
                 torch.cuda.empty_cache()
                 gc.collect()

print(f"\nProbe complete! Results saved to {output_file}")
print(f"Final {get_vram_usage()}")
