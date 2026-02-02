
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training
import logging
import json
from datasets import load_dataset

sys.path.append(os.path.abspath("src"))  
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter

# Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_deepplanning.log")
    ]
)
logger = logging.getLogger("DeepPlanningTrainer")

# Config
MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
DATASET_PATH = "igbundle-llm/data/deepplanning_train.jsonl"
OUTPUT_DIR = "igbundle_deepplanning_training"
MAX_STEPS = 50 # Pilot Run Limit
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
DEVICE = "cuda"

class DeepPlanningTrainer:
    def __init__(self):
        self.config = IGBundleConfig(
            hidden_size=3584,
            num_components=8,
            latent_dim=64,
            num_categories=16,
            use_dynamics=True,
            use_geodesic_attn=True,
            manifold_type="kan" # KAN Geometry
        )
        self.tokenizer = None
        self.llm = None
        self.adapter = None
        
    def setup_model(self):
        logger.info(f"Loading Base Model: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        self.llm.config.use_cache = False
        
        logger.info("Initializing KAN Adapter...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # Inject
        target_layer = self.llm.model.layers[12]
        original_forward = target_layer.forward
        self._current_geo_state = None
        
        def adapter_hook(hidden_states, *args, **kwargs):
            out = original_forward(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            orig_dtype = h.dtype
            h_in = h.to(DEVICE).to(torch.float32)
            adapted_out, geo_state = self.adapter(h_in, pixel_values=None)
            self._current_geo_state = geo_state
            adapted = adapted_out.to(orig_dtype)
            if isinstance(out, tuple):
                return (adapted,) + out[1:]
            return adapted
            
        target_layer.forward = adapter_hook
        logger.info("Adapter Injected.")
        
    def prepare_dataset(self):
        logger.info(f"Loading Dataset: {DATASET_PATH}")
        ds = load_dataset("json", data_files=DATASET_PATH, split="train")
        
        def process(item):
            # Prompt format: System + Input -> Output
            system = item.get('system', '')
            user = item['input']
            assistant = item['output']
            
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            full_text = prompt + assistant + "<|im_end|>"
            
            enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Mask user part for labels? Naive for now: learn all
            ids = enc.input_ids.squeeze(0)
            labels = ids.clone()
            # Simple masking: mask until "assistant"
            # Skipping advanced masking for pilot
            
            return {
                "input_ids": ids,
                "attention_mask": enc.attention_mask.squeeze(0),
                "labels": labels
            }
            
        self.dataset = ds.map(process)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
    def train(self):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=LEARNING_RATE)
        train_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        logger.info("Starting Pilot Training...")
        self.llm.train()
        
        step = 0
        epoch = 0
        while step < MAX_STEPS:
            for batch in train_loader:
                if step >= MAX_STEPS: break
                
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                outputs = self.llm(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss_llm = outputs.loss
                
                # Geometric Loss
                loss_geo = 0.0
                if self._current_geo_state:
                    losses = self.adapter.compute_geometric_losses(self._current_geo_state)
                    loss_geo = sum(losses.values())
                
                total_loss = loss_llm + 0.1 * loss_geo
                total_loss.backward()
                
                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                    optimizer.step()
                
                if step % 5 == 0:
                    logger.info(f"Step {step}: Total={total_loss.item():.4f} LLM={loss_llm.item():.4f} Geo={loss_geo:.4f}")
                    
                step += 1
                
            epoch += 1
            
        # Save Pilot
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(self.adapter.state_dict(), os.path.join(OUTPUT_DIR, "adapter_pilot.pt"))
        logger.info(f"Pilot saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    trainer = DeepPlanningTrainer()
    trainer.setup_model()
    trainer.prepare_dataset()
    trainer.train()
