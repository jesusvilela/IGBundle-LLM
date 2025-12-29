import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import math
import os
import glob
from tqdm import tqdm
from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector
from igbundle.utils import triton_fix

class ValidationBenchmark:
    def __init__(self, model_id, checkpoint_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Model
        print(f"Loading base: {model_id} on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if checkpoint_path:
            print(f"Loading adapter: {checkpoint_path}")
            # Patch for IGBundle
            # We need to manually recreate the 'ig_adapter' config or infer it.
            # Ideally we read it from the saved config.
            # For now, we hardcode the standard config for reliability in this script.
            class DictConfig:
                def __init__(self, d):
                    for k,v in d.items(): setattr(self, k, v)
            
            # TODO: Load this from config file in checkpoint if exists
            adapter_config = {
                'hidden_size': self.model.config.hidden_size,
                'num_components': 4,
                'num_categories': 16,
                'bottleneck_dim': 256,
                'latent_dim': 128,
                'adapter_scale': 0.1,
                'dropout': 0.05
            }
            self.model = wrap_hf_candidate(self.model, DictConfig(adapter_config))
            
            # Load Weights
            # Try loading adapter_weights.pt first (IGBundle specific)
            ig_weights = os.path.join(checkpoint_path, "adapter_weights.pt")
            if os.path.exists(ig_weights):
                print(f"Loading IGBundle weights: {ig_weights}")
                sd = torch.load(ig_weights, map_location=self.device)
                self.model.load_state_dict(sd, strict=False)
            
            # Try LoRA
            try:
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            except Exception as e:
                print(f"LoRA load warning: {e}")

        self.model.eval()

    def perplexity(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.max_position_embeddings
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating Perplexity"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                # NEGATIVE_LOG_LIKELIHOOD
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    def evaluate_topology(self, prompts):
        collector = StateCollector()
        collector.attach(self.model)
        
        results = []
        
        for p in prompts:
            collector.clear()
            inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
            
            if collector.states:
                mean_sigma = sum([s.sigma.mean().item() for s in collector.states]) / len(collector.states)
                results.append(mean_sigma)
            else:
                results.append(0.0)
                
        return sum(results) / len(results)

def run_benchmarks(checkpoint_path):
    print("=== IG-Bundle Industry Validation ===")
    
    # 1. WikiText-2 (Snippet) for Perplexity
    # We use a built-in snippet to avoid downloading large datasets
    wiki_snippet = """
    The game began with a start by the offense. 
    The operational attributes of the unit were considered highly efficient.
    Mathematical topology explores the properties of space that are preserved under continuous deformation.
    Deep learning models utilize vast amounts of data to approximate functions.
    """ * 20 # Repeat to get some length
    
    bench = ValidationBenchmark("Qwen/Qwen2.5-7B", checkpoint_path)
    
    print("\n[Metric 1] Perplexity (WikiText Proxy)")
    ppl = bench.perplexity(wiki_snippet)
    print(f"Perplexity: {ppl:.2f}")
    
    print("\n[Metric 2] Topological Curvature (Sigma)")
    topo_score = bench.evaluate_topology([
        "The curvature of spacetime is determined by mass.",
        "A manifold is a topological space.",
        "Love is a complex emotion."
    ])
    print(f"Mean Sigma: {topo_score:.4f}")
    
    # Update README or log
    with open("benchmark_results.md", "a") as f:
        f.write(f"| Checkpoint | Perplexity | Sigma |\n")
        f.write(f"| {os.path.basename(checkpoint_path)} | {ppl:.2f} | {topo_score:.4f} |\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    
    run_benchmarks(args.checkpoint)
