
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
from datasets import load_dataset
import logging
import random
import json
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np
import time

sys.path.append(os.path.abspath("src"))  
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.modules.regularization import LipschitzPenalty
from igbundle.optimization.symplectic import SymplecticSPIDER
from igbundle.modules.spectral import SpectrallyNormalizedLinear

# Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_odyssey.log")
    ]
)
logger = logging.getLogger("OdysseyTrainer")

# Configuration
MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600" 
OUTPUT_DIR = "igbundle_phase9_odyssey"
MAX_STEPS = 3001 # Phase B: Retrain to checkpoint-3000 with entropy gradient unfrozen
BATCH_SIZE = 1 
GRAD_ACCUM = 16 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_SOURCE = "igbundle_phase9_odyssey" 

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

class OdysseyTrainer:
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
        self.optimizer = None
        self.train_ds = None
        self._current_geo_state = None # For Geometry Loss

    def inject_spectral_norm(self, model):
        """
        Replaces Linear layers with SpectrallyNormalizedLinear for Kirszbraun consistency.
        """
        logger.info("Injecting SpectrallyNormalizedLinear (Kirszbraun Extension)...")
        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # PHASE 8 FIX: Do NOT apply SN to output_proj (Zero-Init causes 0/0 NaN)
                if "vision_proj" in name: 
                    replacements.append((name, module))
        
        for name, module in replacements:
            sn_layer = SpectrallyNormalizedLinear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            with torch.no_grad():
                sn_layer.linear.weight.copy_(module.weight)
                if module.bias is not None:
                     sn_layer.linear.bias.copy_(module.bias)
            
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], sn_layer)
            logger.info(f"Replaced {name} with SpectrallyNormalizedLinear")

    def _inject_adapter_hook(self):
        """Injects the adapter into the LLM forward pass."""
        # Find target layer (usually layer 12 for Qwen mid-stack injection)
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
            
            # Forward through Geometric Adapter
            # Returns (adapted_hidden, geometric_state)
            adapted_out, geo_state = self.adapter(h_in, pixel_values=None)
            self._current_geo_state = geo_state
            
            adapted = adapted_out.to(orig_dtype)
            if isinstance(out, tuple):
                return (adapted,) + out[1:]
            return adapted
            
        target_layer.forward = adapter_hook
        logger.info(f"Injected Adapter Hook at Layer {target_layer_idx}")

    def prepare_datasets(self):
        logger.info("Initializing Expanded Reasoning Dataset (full_scale_train.jsonl)...")
        # Load local JSONL directly
        data_path = "data/full_scale_train.jsonl"
        ds_reasoning = load_dataset("json", data_files={"train": data_path}, split="train", streaming=True)
        
        def tokenize_and_format(item, dataset_idx):
            # item has 'input' and 'output'
            text = f"Question: {item.get('input', '')}\nAnswer: {item.get('output', '')}"

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
                "labels": enc.input_ids.squeeze(0)
            }

        self.train_ds = CompositeStreamingDataset(
            [ds_reasoning],
            weights=[1.0],
            transform=tokenize_and_format
        )

    def setup(self):
        logger.info("Initializing Tokenizer & Model...")
        # torch.autograd.set_detect_anomaly(True) # Debug NaN gradients (Disabled due to hang)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.llm.requires_grad_(False) 
        
        logger.info("Initializing GeometricAdapter...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # Load Phase 9 Checkpoint if available
        self.start_step = 0
        try:
             if os.path.exists(CHECKPOINT_SOURCE):
                 dirs = [d for d in os.listdir(CHECKPOINT_SOURCE) if d.startswith("checkpoint-")]
                 if dirs:
                     latest = sorted(dirs, key=lambda x: int(x.split("-")[1]))[-1]
                     self.start_step = int(latest.split("-")[1])
                     ckpt_path = os.path.join(CHECKPOINT_SOURCE, latest, "adapter_weights.pt")
                     if os.path.exists(ckpt_path):
                         logger.info(f"Loading weights from {ckpt_path} (Resuming from step {self.start_step})")
                         idx = torch.load(ckpt_path, map_location=DEVICE)
                         self.adapter.load_state_dict(idx, strict=False)
        except Exception as e:
             logger.warning(f"Failed to load checkpoint: {e}")
        
        self.inject_spectral_norm(self.adapter)
        self.adapter.to(DEVICE) # Ensure new SN layers are on device
        self._inject_adapter_hook()
        self.adapter.train()
        
        # Optimizer Setup
        base_params = []
        fiber_params = []
        for name, p in self.adapter.named_parameters():
            if not p.requires_grad: continue
            if "fiber_store" in name or "latent_store" in name:
                base_params.append(p)
                p._is_base = True
            else:
                fiber_params.append(p)
                p._is_base = False
                
        logger.info(f"Optimizer Groups: Base={len(base_params)} params, Fiber={len(fiber_params)} params")
        
        self.optimizer = SymplecticSPIDER(
            [
                {'params': base_params, 'is_base': True, 'base_lr': 1e-4, 'base_momentum': 0.0},
                {'params': fiber_params, 'is_base': False, 'fiber_lr': 5e-3, 'fiber_momentum': 0.9}
            ],
            c=5.0,
            period=100
        )
        
        self.prepare_datasets()

    def train(self):
        logger.info("Starting The Odyssey Real Training...")
        train_loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE)
        iter_loader = iter(train_loader)
        
        step = getattr(self, "start_step", 0)
        optimizer = self.optimizer
        
        while step < MAX_STEPS:
            try:
                batch = next(iter_loader)
            except Exception as e:
                logger.error(f"Batch Error: {e}")
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Forward Pass
            # Adapter hook will trigger and populate self._current_geo_state
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss_llm = outputs.loss
            loss_geo = 0.0
            
            if self._current_geo_state and hasattr(self.adapter, 'compute_geometric_losses'):
                geo_losses = self.adapter.compute_geometric_losses(self._current_geo_state)

                # Phase B sub-homotopy: ramp Phase A entropy losses over 200 steps after resume
                # to avoid gradient shock on weights trained without entropy gradient flow.
                # Steps 2000-2200: entropy_ramp goes 0.0 → 1.0
                resume_step = getattr(self, 'start_step', 0)
                entropy_ramp = min(1.0, max(0.0, (step - resume_step) / 200.0)) if resume_step > 0 else 1.0

                loss_geo = 0.0
                for k, v in geo_losses.items():
                    if k in ('fiber_diversity', 'fiber_entropy'):
                        loss_geo += v * entropy_ramp
                    else:
                        loss_geo += v

            # Homotopy Schedule: Soft constraints early, harder constraints later
            # lambda_t scales from 0.0 to 0.1 over the first 500 steps to prevent Phase II collapse.
            lambda_t = 0.1 * min(1.0, step / 500.0)
            total_loss = loss_llm + lambda_t * loss_geo
            
            # Backward
            total_loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                # No external clip_grad_norm_ — SymplecticSPIDER has built-in relativistic
                # velocity limiter (Lorentz gamma = sqrt(1 + ||m||^2/c^2) caps at c).
                # External clipping was killing fiber_section_proj gradients:
                # base_coord_proj curvature grad (~6M) dominated group norm, reducing
                # fiber grads to ~0.07 effective → S frozen at ln(16).
                grad_norm = sum(p.grad.norm().item()**2 for p in self.adapter.parameters()
                                if p.grad is not None) ** 0.5
                
                # Surgically zero ONLY the NaN/Inf entries, keep valid gradients
                nan_param_count = 0
                total_nan_entries = 0
                for name, p in self.adapter.named_parameters():
                    if p.grad is not None:
                        nan_mask = torch.isnan(p.grad) | torch.isinf(p.grad)
                        if nan_mask.any():
                            nan_param_count += 1
                            total_nan_entries += nan_mask.sum().item()
                            p.grad[nan_mask] = 0.0  # Zero only bad entries
                
                # ALWAYS step (the whole point - don't skip valid gradients)
                optimizer.step()
                optimizer.zero_grad()

                # Phase B: Clear CUDA cache to prevent OOM accumulation on long runs
                if step % 64 == 0:
                    torch.cuda.empty_cache()
                
                # Report
                status = ""
                if nan_param_count > 0:
                    status = f" | NaN-zeroed: {nan_param_count} params ({int(total_nan_entries)} entries)"

                # Phase B telemetry: per-loss breakdown + actual entropy value
                geo_detail = ""
                if self._current_geo_state and geo_losses:
                    parts = []
                    for k, v in sorted(geo_losses.items()):
                        val = v.item() if hasattr(v, 'item') else float(v)
                        parts.append(f"{k}={val:.3f}")
                    geo_detail = f" | [{', '.join(parts)}]"

                    # Actual manifold entropy S (not loss, the raw value)
                    if self._current_geo_state.fiber_sections is not None:
                        with torch.no_grad():
                            p = self._current_geo_state.fiber_sections.clamp(min=1e-8)
                            S_actual = -(p * p.log()).sum(dim=-1).mean().item()
                            geo_detail += f" | S={S_actual:.4f}"

                    if resume_step > 0:
                        geo_detail += f" | e_ramp={entropy_ramp:.2f}"

                logger.info(f"Step {step} | Loss: {loss_llm.item():.4f} | Geo: {loss_geo:.2f}{geo_detail}{status}")

            step += 1
            if step % 200 == 0 or step == MAX_STEPS: # Phase B: checkpoint every 200 steps
                 # Checkpoint
                 ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
                 os.makedirs(ckpt_dir, exist_ok=True)
                 torch.save(self.adapter.state_dict(), os.path.join(ckpt_dir, "adapter_weights.pt"))
                 logger.info(f"Saved Checkpoint to {ckpt_dir}")

if __name__ == "__main__":
    trainer = OdysseyTrainer()
    try:
        trainer.setup()
        trainer.train() 
        logger.info("Odyssey Training Complete.")
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        # Emergency checkpoint save on crash
        if trainer.adapter is not None:
            try:
                step = getattr(trainer, 'start_step', 0)
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-emergency-{int(time.time())}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(trainer.adapter.state_dict(), os.path.join(ckpt_dir, "adapter_weights.pt"))
                logger.info(f"Emergency checkpoint saved to {ckpt_dir}")
            except Exception:
                pass
        raise
