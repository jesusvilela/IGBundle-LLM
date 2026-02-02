import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import json
import logging
import argparse

# Path Setup
sys.path.append(os.path.abspath("src"))

from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.training.losses import RetrospectionLoss

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase3Trainer")

class ARCDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        import glob
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    content = json.load(f)
                    if "train" in content:
                        for pair in content["train"]:
                            inp = str(pair["input"])
                            out = str(pair["output"])
                            text = f"Pattern Analysis:\nInput Grid: {inp}\n\nTask: Derive the underlying geometric transformation logic.\n\nOutput Grid should be: {out}"
                            self.data.append(text)
            except Exception:
                pass
        logger.info(f"Loaded {len(self.data)} ARC samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length")
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": enc.input_ids.squeeze(0)
        }

class Phase3PotentialTrainer:
    def __init__(self, args):
        self.args = args
        self.config = IGBundleConfig(
            hidden_size=3584, # Qwen 2.5 7B
            latent_dim=128,
            num_components=8,
            use_dynamics=True,
            num_leapfrog_steps=4
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def setup(self):
        # 1. Load Model
        logger.info(f"Loading Model: {self.args.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        
        # 2. Init Geometric Adapter (With Potential)
        logger.info("Initializing Geometric Adapter with Neural Potential...")
        self.adapter = create_geometric_adapter(self.config).to(self.device)
        
        # 3. Inject into Layer 15
        self._inject_adapter(layer_idx=15)
        
        # 4. Losses
        # Note: We need access to the Hamiltonian integrator inside the adapter
        # Just passing the integrator object might be enough if it's stateful or if we pass (q,p) manually
        # Ideally RetrospectionLoss is stateless and takes (q_final, p_final, integrator) or we use the adapter's hook to capture state.
        # We will capture state in the hook.
        
    def _inject_adapter(self, layer_idx):
        layers = self.llm.model.layers
        target_layer = layers[layer_idx]
        original_forward = target_layer.forward
        
        self._geo_state = None
        self._geo_inputs = None # (q_in, p_in) for retro check?
        
        def adapter_hook(hidden_states, *args, **kwargs):
            # 1. Original Layer
            out = original_forward(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            
            # 2. Adapter
            # Cast to float32 for Physics
            h_in = h.to(self.device).to(torch.float32)
            adapted_out, geo_state = self.adapter(h_in)
            
            # Capture state for Loss calculation in training loop
            self._geo_state = geo_state
            
            # 3. Residual + Cast back
            # adapted_out is already residualized in some versions, check adapter code.
            # Usually adapter returns *modification*.
            # Let's assume it returns the TRANSFORMED hidden state.
            
            h_final = adapted_out.to(h.dtype)
            
            # Debug Hook
            # print(f"HOOK: Layer {layer_idx} | h_in grad: {h.requires_grad} | h_out grad: {h_final.requires_grad}", flush=True)
            
            if isinstance(out, tuple):
                return (h_final,) + out[1:]
            return h_final
            
        target_layer.forward = adapter_hook
        logger.info(f"Injected Adapter at Layer {layer_idx}")

    def train(self):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=2e-4) # Train Adapter Only
        dataset = ARCDataset(self.args.data_dir, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        retro_loss_fn = RetrospectionLoss(self.adapter.integrator, lambda_reg=0.5)
        
        logger.info("Starting Training Loop...")
        self.llm.train()
        
        for step, batch in enumerate(dataloader):
            if step >= self.args.max_steps: break
            
            try:
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward Pass (End-to-End)
                outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # 1. Task Loss
                loss_task = outputs.loss
                
                # Manual Log
                with open("C:/Users/HAL900/debug_phase3.txt", "a") as f:
                     f.write(f"Step {step} | Task Loss: {loss_task.item() if loss_task is not None else 'None'} | Grad: {loss_task.requires_grad}\n")
                     f.write(f"Adapter Params Require Grad: {any(p.requires_grad for p in self.adapter.parameters())}\n")
                     
                # 2. Retro Loss
                B = input_ids.shape[0]
                q_dream = torch.randn(B, 64, 8, 128, device=self.device, requires_grad=True) * 0.1
                p_dream = torch.randn_like(q_dream) * 0.1
                loss_retro = retro_loss_fn(q_dream, p_dream)

                with open("C:/Users/HAL900/debug_phase3.txt", "a") as f:
                     f.write(f"Retro Loss Grad: {loss_retro.requires_grad}\n")

                total_loss = loss_task + 0.1 * loss_retro
                total_loss.backward()
                optimizer.step()
                
                if step % 5 == 0:
                    logger.info(f"Step {step}: Total={total_loss.item():.4f}")
                    
            except Exception as e:
                import traceback
                with open("C:/Users/HAL900/debug_phase3.txt", "a") as f:
                    f.write(f"ERROR: {e}\n")
                    f.write(traceback.format_exc())
                raise e

        torch.save(self.adapter.state_dict(), "output/phase3_adapter_potential.pt")
        logger.info("Saved Phase 3 Adapter")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir", type=str, default="data/arc_agi")
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()
    
    trainer = Phase3PotentialTrainer(args)
    trainer.setup()
    trainer.train()
