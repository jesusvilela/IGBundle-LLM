import torch
from unsloth import FastLanguageModel
import argparse
import os
import time
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from igbundle.modules.geometric_adapter import create_geometric_adapter, IGBundleConfig
from igbundle.training.losses import RetrospectionLoss
# # import wandb
import json
from torch.utils.data import Dataset, DataLoader

# Dataset Class copied from train_unified.py for standalone utility
class ARCDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Walk through ARC directory
        import glob
        # Assuming data_dir is relative to script
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        if not json_files:
             print(f"Warning: No valid JSON files found in {data_dir}. Using dummy data if empty.")
             
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    content = json.load(f)
                    # Train pairs
                    if "train" in content:
                        for pair in content["train"]:
                            inp = str(pair["input"])
                            out = str(pair["output"])
                            text = f"Pattern Analysis:\nInput Grid: {inp}\n\nTask: Derive the underlying geometric transformation logic.\n\nOutput Grid should be: {out}"
                            self.data.append(text)
            except Exception as e:
                print(f"Error parsing ARC file {jf}: {e}")
                
        print(f"Loaded {len(self.data)} ARC samples from {data_dir}")

    def __len__(self):
        return max(len(self.data), 100) # Ensure at least some length for loop

    def __getitem__(self, idx):
        if not self.data:
             # Fallback
             text = "Input: [[0]], Output: [[0]]"
        else:
             text = self.data[idx % len(self.data)]
             
        enc = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            padding="max_length"
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": enc.input_ids.squeeze(0)
        }

def train_phase2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir", type=str, default="data/arc_agi")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--retro_lambda", type=float, default=0.5, help="Weight for Retrospection Loss")
    args = parser.parse_args()
    
    # 1. Load Model (Unsloth 4-bit)
    print("Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    
    # 2. Inject Geometric Adapter (Phase 2)
    print("Injecting Geometric Adapter (Phase 2)...")
    config = IGBundleConfig(
        hidden_size=3584, # Qwen 2.5 7B hidden size is 3584 not 4096? Let's check config or trust default.
        # Actually Qwen2.5-7B hidden size IS 3584. 4096 was for Llama/Mistral usually.
        # Checking loaded model config is safer if we can access it.
        # But IGBundleConfig needs it at init.
        # Let's set it to proper Qwen value: 3584
        latent_dim=128,
        num_components=8,
        use_dynamics=True,
        num_leapfrog_steps=4
    )
    # Patch config hidden size if possible
    # config.hidden_size = model.config.hidden_size
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        use_gradient_checkpointing = True,
    )
    
    # Instantiate OUR Geometric Adapter
    geometric_adapter = create_geometric_adapter(config).to("cuda")
    
    # 3. Setup Retrospection Loss
    # We need access to the adapter's integrator
    retro_loss_fn = RetrospectionLoss(geometric_adapter.integrator, lambda_reg=args.retro_lambda)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(geometric_adapter.parameters()), 
        lr=2e-4
    )
    
    # 5. Data Loader
    print(f"Loading ARC Data from {args.data_dir}...")
    dataset = ARCDataset(args.data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 6. Training Loop
    print("Starting Phase 2 Training (Hamiltonian + Retrospection on ARC Data)...")
    
    model.train()
    # We need to manually handle the training loop since we have a custom loss component
    # that depends on the adapter state, which SFTTrainer hides.
    
    step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            
            # Forward Pass: Run through LLM to get embeddings at adapter layer?
            # Or assume we pass input_ids to adapter? No, adapter takes Hidden States.
            # We need to hook the model or run partial forward.
            # For simplicity in this script, we'll assume we attached the adapter 
            # to the model via a hook (like in train_unified.py) OR we simulate the input.
            
            # For exact correctness without defining the hook class again:
            # We will grab the embeddings from the model and pass them to adapter.
            # This is a bit of an approximation (Adapter usually at middle layer locally).
            # But for testing MECHANICS, let's use the model's *input embeddings*.
            with torch.no_grad():
                inputs_embeds = model.get_input_embeddings()(input_ids)
            
            # Adapter Forward (Approximating layer input as raw embeddings for demo)
            # In production, use the Hook method.
            # Scaling to correct dimension if needed (embeddings match hidden size 3584)
            x = inputs_embeds.to(torch.float32)
            
            out, state = geometric_adapter(x)
            
            # Retro Step: Sample random points on manifold ("Dreaming") 
            # or use the state's coordinates if we want to enforce reversibility on *actual* thoughts.
            # Let's enforce it on random points for global stability first.
            B_size, T_size = x.shape[:2]
            q_dream = torch.randn(B_size, T_size, config.num_components, config.latent_dim, device="cuda") * 0.1
            p_dream = torch.randn_like(q_dream) * 0.1
            
            loss_retro = retro_loss_fn(q_dream, p_dream)
            
            # Geometric Structure Loss
            # CAUTION: compute_geometric_losses involves expensive Curvature tensor (OOM risk)
            # loss_geo = geometric_adapter.compute_geometric_losses(state)
            # total_geo_loss = sum(loss_geo.values()) if loss_geo else 0
            total_geo_loss = 0.0
            
            # Total Loss
            loss = loss_retro + total_geo_loss
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}: Loss={loss.item():.4f} (Retro={loss_retro.item():.4f})")
            
            step += 1
            if step >= 100: break # Limit for this run
        if step >= 100: break
            
    print("Training Complete. Saving Phase 2 Adapter...")
    torch.save(geometric_adapter.state_dict(), "output/phase2_adapter_arc.pt")
    
if __name__ == "__main__":
    train_phase2()
