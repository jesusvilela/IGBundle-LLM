
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training
import random
import logging
import json
from typing import Dict, List, Optional, Tuple, Iterator
from datasets import load_dataset
import numpy as np

sys.path.append(os.path.abspath("src"))  
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter

# Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_refined_hf.log")
    ]
)
logger = logging.getLogger("RefinedTrainer")

# Configuration
MODEL_ID = "../igbundle_qwen7b_cp600" 
OUTPUT_DIR = "igbundle_refined_training"
MAX_STEPS = 500 # Refinement phase usually shorter
BATCH_SIZE = 1 
GRAD_ACCUM = 16 
LEARNING_RATE = 1e-4 # Lower LR for refinement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_SOURCE = "igbundle_diverse_training/checkpoint-1000/adapter_weights.pt"

class CompositeStreamingDataset(IterableDataset):
    def __init__(self, datasets, weights=None, transform=None):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        self.transform = transform
        
    def __iter__(self):
        iterators = [iter(d) for d in self.datasets]
        while True:
            idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
            try:
                item = next(iterators[idx])
                if self.transform:
                    yield self.transform(item, dataset_idx=idx)
                else:
                    yield item
            except StopIteration:
                 iterators[idx] = iter(self.datasets[idx])
                 continue
            except Exception as e:
                logger.warning(f"Error yielding from dataset {idx}: {e}")
                continue

class RefinedTrainer:
    def __init__(self):
        self.config = IGBundleConfig(
            hidden_size=3584, # Qwen 7B
            num_components=8, 
            latent_dim=64,    
            num_categories=16,
            use_dynamics=True,
            use_geodesic_attn=True,
            supported_modalities=["vision", "text"]
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
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        self.llm.config.use_cache = False
        
        logger.info("Initializing Geometric IGBundle Adapter (With Fiber Refinement)...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # Load Checkpoint and strictly handle mismatches (new modules vs old checkpoint)
        if os.path.exists(CHECKPOINT_SOURCE):
            logger.info(f"RESUMING from Diverse Checkpoint: {CHECKPOINT_SOURCE}")
            # strict=False allows loading weights even if fiber_latents are new/missing in checkpoint
            # We explicitly want to keep new random initialization for fiber_store if it wasn't in checkpoint.
            keys = self.adapter.load_state_dict(torch.load(CHECKPOINT_SOURCE), strict=False)
            logger.info(f"Missing Keys (New Modules): {keys.missing_keys}")
            logger.info(f"Unexpected Keys: {keys.unexpected_keys}")
        else:
            logger.warning(f"Checkpoint {CHECKPOINT_SOURCE} not found! Starting fresh.")
        
        logger.info("Injecting Adapter Hook at Layer 12...")
        self._inject_adapter()
        
    def _inject_adapter(self):
        target_layer_idx = 12
        layers = self.llm.model.layers
        target_layer = layers[target_layer_idx]
        original_forward = target_layer.forward
        self._current_geo_state = None
        
        def adapter_hook(hidden_states, *args, **kwargs):
            out = original_forward(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            orig_dtype = h.dtype
            h_in = h.to(DEVICE).to(torch.float32)
            # No visual features for this text-heavy run
            adapted_out, geo_state = self.adapter(h_in, pixel_values=None)
            self._current_geo_state = geo_state
            adapted = adapted_out.to(orig_dtype)
            if isinstance(out, tuple):
                return (adapted,) + out[1:]
            return adapted
            
        target_layer.forward = adapter_hook
        
    def prepare_datasets(self):
        logger.info("Initializing HF Streaming Datasets (Expanded Mix)...")
        
        # 1. Cosmopedia (Textbooks / Knowledge) - 40%
        ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True)
        # Note: 'stories' subset is small/fast, 'stanford' or 'web_samples' might be better for textbook quality. 
        # Let's use 'auto_math_text' if specific, or default. 
        # Using 'stories' for now as it's safe.
        
        # 2. Orca Math (Logic/Reasoning) - 30%
        ds_orca = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
        
        # 3. FineWeb-Edu (General Quality) - 20%
        ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        # 4. OpenAssistant (Dialogue Alignment) - 10%
        ds_dialog = load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train", streaming=True)
        
        def tokenize_and_format(item, dataset_idx):
            text = ""
            modality = "text"
            
            if dataset_idx == 0: # Cosmopedia
                text = item.get('text', '') or item.get('prompt', '') # Check structure
            elif dataset_idx == 1: # Orca
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            elif dataset_idx == 2: # FineWeb
                text = item['text']
            elif dataset_idx == 3: # OASST
                text = item['text']

            enc = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding="max_length"
            )
            
            return {
                "input_ids": enc.input_ids.squeeze(0),
                "attention_mask": enc.attention_mask.squeeze(0),
                "labels": enc.input_ids.squeeze(0),
                "modality": modality
            }

        self.train_ds = CompositeStreamingDataset(
            [ds_cosmo, ds_orca, ds_edu, ds_dialog],
            weights=[0.4, 0.3, 0.2, 0.1],
            transform=tokenize_and_format
        )
        
    def train(self):
        # Ensure fiber parameters are included in optimization
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=LEARNING_RATE)
        self.prepare_datasets()
        
        train_loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE)
        iter_loader = iter(train_loader)
        
        logger.info(f"Starting Refined Training Loop (Target Steps: {MAX_STEPS})...")
        self.llm.train()
        
        for step in range(MAX_STEPS):
            optimizer.zero_grad()
            try:
                batch = next(iter_loader)
            except Exception as e:
                logger.error(f"Batch Error: {e}")
                iter_loader = iter(train_loader) 
                batch = next(iter_loader)
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss_llm = outputs.loss
            loss_geo = 0.0
            
            if self._current_geo_state:
                geo_losses = self.adapter.compute_geometric_losses(self._current_geo_state)
                loss_geo = sum(geo_losses.values())
                
                # Active Index Logging & Telemetry
                if step % 5 == 0:
                     active_idx = []
                     if getattr(self._current_geo_state, 'active_indices', None) is not None:
                         active_idx = self._current_geo_state.active_indices[0].tolist()
                         logger.info(f"Step {step} Active: {active_idx}")
                     
                     # Write state for dashboard
                     try:
                         dash_state = {
                             "status": "training",
                             "id": str(step),
                             "domain": "Refined-HF",
                             "prompt": f"Step {step} | Loss {loss_llm.item():.4f} | Geo {loss_geo:.2f}",
                             "curvature": loss_geo * 100, 
                             "entropy": len(active_idx) if active_idx else 0,
                             "active_fiber": str(active_idx),
                             "embedding": [0.0]*32
                         }
                         with open("dashboard_state.json", "w") as f:
                             json.dump(dash_state, f)
                     except: pass
            
            total_loss = loss_llm + 0.1 * loss_geo
            total_loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 10 == 0:
                logger.info(f"Step {step}: Total Loss = {total_loss.item():.4f} (LLM={loss_llm.item():.4f})")
            
            if (step + 1) % 100 == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step+1}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.adapter.state_dict(), os.path.join(save_path, "adapter_weights.pt"))
                logger.info(f"Saved checkpoint to {save_path}")

        logger.info("Refined Training Complete.")
        
if __name__ == "__main__":
    trainer = RefinedTrainer()
    trainer.setup_model()
    trainer.train()
