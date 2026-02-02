
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    SiglipVisionModel, 
    SiglipImageProcessor
)
from peft import prepare_model_for_kbit_training
import json
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
# import wandb
import logging
from typing import Dict, List, Optional, Tuple

# Path Setup
# Assumes script is in h:\LLM-MANIFOLD\igbundle-llm\
sys.path.append(os.path.abspath("src"))  # Ensure src/igbundle is in path

# IGBundle Imports
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter, GeometricState
# from igbundle.modules.vision import VisionProjector # Integrated in Adapter
# from igbundle.modules.consensus import SheafConsensus # Integrated in Adapter

# ------------------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} text samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text'] # Expects "Instruction: ... Response: ..." format
        
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
            "labels": enc.input_ids.squeeze(0), # Causal LM
            "modality": "text"
        }

class VisualTextDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, processor, max_length=512):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.data = []
        base_dir = os.path.dirname(jsonl_path)
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if "image" in entry:
                         # Fix: JSON paths are already relative to root (data/...)
                         # If we join with base_dir (data), we get data/data/...
                         if os.path.exists(entry["image"]):
                             entry["image_path"] = entry["image"]
                         else:
                             entry["image_path"] = os.path.join(base_dir, entry["image"])
                    self.data.append(entry)
        print(f"Loaded {len(self.data)} visual samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Process Text
        text = item['text'].replace("<image>", "") # Remove token if present
        enc = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            padding="max_length"
        )
        
        # 2. Process Image
        pixel_values = None
        if "image_path" in item and os.path.exists(item["image_path"]):
            try:
                image = Image.open(item["image_path"]).convert("RGB")
                # Processor returns dict with pixel_values
                inputs = self.processor(images=image, return_tensors="pt")
                # Resize if needed? Siglip processor handles resizing
                pixel_values = inputs.pixel_values.squeeze(0) # (3, H, W)
            except Exception as e:
                print(f"Error loading image {item.get('image_path')}: {e}")
                pixel_values = torch.zeros(3, 378, 378) # Siglip size
        else:
             pixel_values = torch.zeros(3, 378, 378)
             
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": enc.input_ids.squeeze(0),
            "pixel_values": pixel_values,
            "modality": "visual"
        }

class CompositeDataset(Dataset):
    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        self.total_len = sum(len(d) for d in datasets)
        
    def __len__(self):
        return self.total_len # Approximate
        
    def __getitem__(self, idx):
        # Weighted sampling
        dataset = random.choices(self.datasets, weights=self.weights, k=1)[0]
        rand_idx = random.randint(0, len(dataset)-1)
        return dataset[rand_idx]

# ------------------------------------------------------------------------------
# Extended Datasets (Epic 20)
# ------------------------------------------------------------------------------
class ARCDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Walk through ARC directory
        import glob
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
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
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
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
            "labels": enc.input_ids.squeeze(0),
            "modality": "arc",
            "pixel_values": torch.zeros(3, 378, 378) # ARC is text-based grid representation here
        }

class GSM8KDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        try:
            with open(json_path, 'r') as f:
                content = json.load(f)
                # Structure: {"gsm8k_cot_zeroshot": [{"doc": {"question": "...", "answer": "..."}}]}
                samples = content.get("gsm8k_cot_zeroshot", [])
                for s in samples:
                    q = s["doc"]["question"]
                    a = s["doc"]["answer"]
                    text = f"Math Problem:\n{q}\n\nReasoning:\n{a}"
                    self.data.append(text)
        except Exception as e:
            print(f"Error parsing GSM8K {json_path}: {e}")
            
        print(f"Loaded {len(self.data)} GSM8K samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
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
            "labels": enc.input_ids.squeeze(0),
            "modality": "gsm8k",
            "pixel_values": torch.zeros(3, 378, 378)
        }

# Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_unified_extended.log")
    ]
)
logger = logging.getLogger("UnifiedTrainer")

# Configuration
# Note: Relative path from igbundle-llm directory
MODEL_ID = "../igbundle_qwen7b_cp600" # Relative to script
OUTPUT_DIR = "igbundle_unified_training"
MAX_STEPS = 500
BATCH_SIZE = 1 # Small batch for 7B
GRAD_ACCUM = 8 # High accumulation
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UnifiedManifoldTrainer:
    def __init__(self):
        self.config = IGBundleConfig(
            hidden_size=3584, # Qwen 7B
            num_components=8, # 8 Manifold Patches
            latent_dim=64,    # Dimension D
            num_categories=16,# K Categorical fibers
            use_dynamics=True,
            use_geodesic_attn=True,
            supported_modalities=["vision", "text", "logic", "swarm"]
        )
        
        self.tokenizer = None
        self.llm = None
        self.adapter = None
        
    def setup_model(self):
        logger.info(f"Loading Base Model: {MODEL_ID}")
        
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. Load 4-bit Quantized LLM
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
        
        # 2b. Load SigLip Vision Tower (for Epic 5)
        logger.info("Loading SigLIP Vision Model...")
        self.vision_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.vision_model = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            device_map=DEVICE,
            torch_dtype=torch.float16
        )
        
        # 3. Initialize Geometric Adapter (The Unified Component)
        logger.info("Initializing Geometric IGBundle Adapter...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # Check for existing weights to RESUME
        # RECOVERY: Prioritize checkpoint-100 from crash recovery
        recovery_ckpt = os.path.join(OUTPUT_DIR, "checkpoint-100", "adapter_weights.pt")
        final_ckpt = os.path.join(OUTPUT_DIR, "final", "adapter_weights.pt")
        
        if os.path.exists(recovery_ckpt):
            logger.info(f"RESUMING (RECOVERY): Loading existing weights from {recovery_ckpt}")
            self.adapter.load_state_dict(torch.load(recovery_ckpt))
        elif os.path.exists(final_ckpt):
            logger.info(f"RESUMING: Loading existing weights from {final_ckpt}")
            self.adapter.load_state_dict(torch.load(final_ckpt))
        
        # 4. Inject Adapter Hook
        logger.info("Injecting Adapter Hook at Layer 12...")
        self._inject_adapter()
        
    def _inject_adapter(self):
        target_layer_idx = 12
        layers = self.llm.model.layers
        target_layer = layers[target_layer_idx]
        
        original_forward = target_layer.forward
        
        # Thread-local storage for GeometricState capture
        self._current_geo_state = None
        
        def adapter_hook(hidden_states, *args, **kwargs):
            # Forward pass through original layer
            out = original_forward(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            
            # --- Adapter Injection ---
            # Capture Dtype
            orig_dtype = h.dtype
            
            # Move to Float32 for Manifold Ops
            h_in = h.to(DEVICE).to(torch.float32)
            
            # Forward pass through Geometric Adapter
            # Retrieve vision features from trainer state
            vis_feats = getattr(self, '_current_vision_features', None)
            
            adapted_out, geo_state = self.adapter(h_in, pixel_values=vis_feats)
            
            # Capture State for Loss Computation
            self._current_geo_state = geo_state
            
            # Restore Dtype
            adapted = adapted_out.to(orig_dtype)
            
            if isinstance(out, tuple):
                return (adapted,) + out[1:]
            return adapted
            
        target_layer.forward = adapter_hook
        
    def train(self):
        # Setup Optimizers
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=LEARNING_RATE)
        
        # Setup Datasets
        logger.info("Setting up Multi-Modal Datasets...")
        
        # 1. Text (Unrestricted)
        text_ds = TextDataset("unrestricted_long_context_data.jsonl", self.tokenizer)
        
        # 2. Physics & Geometry (Visual)
        # Note: Paths relative to igbundle-llm
        physics_ds = VisualTextDataset("data/physics_dynamics.jsonl", self.tokenizer, self.vision_processor)
        geo_ds = VisualTextDataset("data/geometric_visual_reasoning.jsonl", self.tokenizer, self.vision_processor)
        
        # 3. ARC (Abstract Logic)
        arc_ds = ARCDataset("data/arc_agi", self.tokenizer)

        # 4. GSM8K (Math)
        # Assuming path is in root of workspace, so ../ relative to igbundle-llm if run from there
        # Or hardcode absolute path for safety as we verify it exists
        gsm8k_ds = GSM8KDataset("h:/LLM-MANIFOLD/debug_samples_gsm8k_cot_zeroshot.json", self.tokenizer)

        # 5. Composite
        # Bias: Text(0.3), Vis(0.3), ARC(0.2), GSM8K(0.2)
        train_ds = CompositeDataset(
            [text_ds, physics_ds, geo_ds, arc_ds, gsm8k_ds], 
            weights=[0.3, 0.15, 0.15, 0.2, 0.2]
        )
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        iter_loader = iter(train_loader)
        
        logger.info(f"Starting Training Loop (Max Steps: {MAX_STEPS})...")
        self.llm.train()
        
        for step in range(MAX_STEPS):
            optimizer.zero_grad()
            
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
            
            # Move to device
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            modality = batch["modality"][0]
            
            # Handle Vision
            self._current_vision_features = None
            if modality == "visual":
                pixel_values = batch["pixel_values"].to(DEVICE).to(torch.float16)
                # Compute Features: (B, N, V)
                with torch.no_grad():
                    vis_out = self.vision_model(pixel_values)
                    # SigLIP returns pooler_output by default? No, last_hidden_state.
                    # Cast to Float32 for Adapter compatibility
                    self._current_vision_features = vis_out.last_hidden_state.to(torch.float32)
            else:
                # Ensure other modalities don't use stale vision features
                self._current_vision_features = None
            
            # Forward Pass
            # Pass labels for CausalLM loss computation
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute Losses
            # 1. LLM Loss
            loss_llm = outputs.loss
            
            # 2. Geometric Loss (from captured state)
            loss_geo = 0.0
            loss_geo_sum = 0.0
            if self._current_geo_state:
                geo_losses = self.adapter.compute_geometric_losses(self._current_geo_state)
                loss_geo_sum = sum(geo_losses.values())
                
                # Log components
                if step % 10 == 0:
                    logger.info(f"Step {step} [{modality}] Geo Losses: { {k: f'{v.item():.4f}' for k, v in geo_losses.items()} }")
            
            # Total Loss
            total_loss = loss_llm + 0.1 * loss_geo_sum
            
            # Backward
            total_loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 10 == 0:
                logger.info(f"Step {step} [{modality}]: Total Loss = {total_loss.item():.4f} (LLM={loss_llm.item():.4f})")
                
                logger.info(f"Step {step} [{modality}]: Total Loss = {total_loss.item():.4f} (LLM={loss_llm.item():.4f})")
            
            # Save Checkpoint
            if (step + 1) % 50 == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step+1}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.adapter.state_dict(), os.path.join(save_path, "adapter_weights.pt"))
                logger.info(f"Saved checkpoint to {save_path}")

        logger.info("Training Complete.")
        # Save Final
        final_path = os.path.join(OUTPUT_DIR, "final")
        os.makedirs(final_path, exist_ok=True)
        torch.save(self.adapter.state_dict(), os.path.join(final_path, "adapter_weights.pt"))
        logger.info(f"Saved final model to {final_path}")
        
if __name__ == "__main__":
    trainer = UnifiedManifoldTrainer()
    trainer.setup_model()
    trainer.train()
