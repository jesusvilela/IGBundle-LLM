
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from datasets import load_dataset
import logging
print("DEBUG: Imports done.")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
    print("DEBUG: BNB Found.")
except ImportError:
    HAS_BNB = False
    print("DEBUG: BNB NOT Found.")
    logger = logging.getLogger("RefinedTrainer")
    logger.warning("BitsAndBytes not found. Switching to CPU Offload Strategy (Slower but compatible).")

from peft import prepare_model_for_kbit_training
import random

import json
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np
import gc

sys.path.append(os.path.abspath("src"))  
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.modules.regularization import LipschitzPenalty, spectral_normalize_module

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
MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600" 
OUTPUT_DIR = "igbundle_phase8_training"
MAX_STEPS = 5000 # Full Scale Phase 8 Run
BATCH_SIZE = 1 
GRAD_ACCUM = 16 
LEARNING_RATE = 5e-5 # Lower LR for stability in long run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_SOURCE = "igbundle_phase8_training/checkpoint-3900" # Resume from Step 3900

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
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        print("DEBUG: Tokenizer initialized.")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # BNB / Quantization Logic
        if HAS_BNB:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "max_memory": {0: "7GB"}, # Leave headroom
                "trust_remote_code": True
            }
        else:
            # Fallback: Float16 with Offloading
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "max_memory": {0: "7GB", "cpu": "32GB"}, # Allow CPU offload
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
        
        print("DEBUG: Loading Model...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            **model_kwargs
        )
        print("DEBUG: Model Loaded.")
        
        if HAS_BNB:
             print("DEBUG: Preparing for kbit training...")
             self.llm = prepare_model_for_kbit_training(self.llm)
        else:
             # Enable Gradient Checkpointing manually for memory saving
             self.llm.gradient_checkpointing_enable() 

        self.llm.config.use_cache = False
        
        print("DEBUG: Initializing Adapter...")
        logger.info("Initializing Geometric IGBundle Adapter (With Fiber Refinement)...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # DEBUG after to(DEVICE)
        if torch.isnan(self.adapter.output_proj.weight).any():
             print(f"CRITICAL: Adapter weights corrupted after .to({DEVICE})")
        else:
             print(f"DEBUG: Adapter weights CLEAN after .to({DEVICE})")
        
        # Apply Spectral Normalization for Kirszbraun Guarantee (Epic 38)
        # DISABLE spectral norm for now as it causes NaN with Zero-Init output_proj
        # spectral_normalize_module(self.adapter)
        print("DEBUG: Spectral Normalization DISABLED.")
        
        # RESUME LOGIC
        self.start_step = 0
        latest_ckpt = self._find_latest_checkpoint()
        
        if latest_ckpt:
             logger.info(f"RESUMING from Latest Refined Checkpoint: {latest_ckpt}")
             self.adapter.load_state_dict(torch.load(latest_ckpt), strict=False)
             # Parse step from path .../checkpoint-100/adapter_weights.pt
             try:
                 step_str = os.path.basename(os.path.dirname(latest_ckpt)).split("-")[1]
                 self.start_step = int(step_str)
                 logger.info(f"Resuming at Step {self.start_step}")
             except:
                 logger.warning("Could not parse step from checkpoint path, starting at 0.")
        elif CHECKPOINT_SOURCE and os.path.exists(CHECKPOINT_SOURCE):
             logger.info(f"Loading Source Weights from: {CHECKPOINT_SOURCE}")
             try:
                 state_dict = torch.load(CHECKPOINT_SOURCE)
                 self.adapter.load_state_dict(state_dict, strict=False)
                 print("DEBUG: Source Adapter Weights Loaded.")
             except Exception as e:
                 logger.error(f"Failed to load source checkpoint: {e}")
        else:
             logger.info("Starting FRESH (Random Initialization)")
             print("DEBUG: Fresh Adapter Initialized.")
        
        logger.info("Injecting Adapter Hook at Layer 12...")
        self._inject_adapter()

    def _find_latest_checkpoint(self):
        if not os.path.exists(OUTPUT_DIR):
            return None
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if not checkpoints:
            return None
        # Sort by number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest = checkpoints[-1]
        ckpt_path = os.path.join(OUTPUT_DIR, latest, "adapter_weights.pt")
        if os.path.exists(ckpt_path):
             return ckpt_path
        return None

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
        
        if not hasattr(self, 'start_step'): self.start_step = 0
        
        # Adjust iter_loader if needed? No, streaming dataset is infinite usually or shuffled.
        # Just skipping steps in loader is hard for streaming.
        # We assume independent batches.
        
        logger.info(f"Starting Refined Training Loop (Steps {self.start_step} -> {MAX_STEPS})...")
        self.llm.train()
        
        for step in range(self.start_step, MAX_STEPS):
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
                
                # Kirszbraun Lipschitz Penalty (Epic 38)
                # Compute penalty on input batch -> output manifold projection
                # We need access to the projector within adapter. 
                # Assuming adapter has 'fiber_projector' or similar. 
                # For now, we apply it if 'fiber_projector' exists.
                loss_lip = 0.0
                if hasattr(self.adapter, 'fiber_projector'):
                     # Use input hidden states (approximate as we don't have them handy here without hook interception)
                     # Actually, `compute_geometric_losses` might be too late.
                     # But we can use the 'base_coords' from _current_geo_state if available?
                     # Let's assume _current_geo_state has 'base_coords' (B, T, D)
                     if hasattr(self._current_geo_state, 'base_coords'):
                         loss_lip = LipschitzPenalty.compute_penalty(
                             self.adapter.fiber_projector, 
                             self._current_geo_state.base_coords,
                             c=self.config.manifold_curvature
                         )
                
                loss_geo += loss_lip
                
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
                # GRADIENT CLINIC: Sanitize gradients before step
                grad_norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
                
                # Check for NaN/Inf in gradients and zero them out if found
                found_nan_grad = False
                for p in self.adapter.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            # logger.warning(f"NaN/Inf gradient detected in {p.shape}. Zeroing grad.")
                            p.grad = None # Zero out (skip update for this param)
                            found_nan_grad = True
                
                if found_nan_grad:
                     logger.warning(f"Step {step}: NaN gradients detected! Skipped update for corrupted params.")
                
                if not found_nan_grad:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # THERMAL THROTTLE: Sleep briefly to let GPU cool
                import time
                if step % 10 == 0:
                    time.sleep(1.0)
            
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
