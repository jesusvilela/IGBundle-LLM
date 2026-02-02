
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
import logging
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath("src"))  
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter, GeometricState

# ------------------------------------------------------------------------------
# Datasets (Reused from train_unified.py)
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
        text = item['text']
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
        text = item['text'].replace("<image>", "")
        enc = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            padding="max_length"
        )
        
        pixel_values = None
        if "image_path" in item and os.path.exists(item["image_path"]):
            try:
                image = Image.open(item["image_path"]).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs.pixel_values.squeeze(0)
            except Exception as e:
                print(f"Error loading image {item.get('image_path')}: {e}")
                pixel_values = torch.zeros(3, 378, 378)
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
        return self.total_len
        
    def __getitem__(self, idx):
        dataset = random.choices(self.datasets, weights=self.weights, k=1)[0]
        rand_idx = random.randint(0, len(dataset)-1)
        return dataset[rand_idx]

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
            "pixel_values": torch.zeros(3, 378, 378)
        }

class GSM8KDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        try:
            with open(json_path, 'r') as f:
                content = json.load(f)
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
        logging.FileHandler("train_symmetry.log")
    ]
)
logger = logging.getLogger("SymmetryTrainer")

# Configuration
MODEL_ID = "../igbundle_qwen7b_cp600" 
OUTPUT_DIR = "igbundle_symmetry_training"
MAX_STEPS = 2000 # Extended Horizon
BATCH_SIZE = 1 
GRAD_ACCUM = 8 
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SymmetryTrainer:
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
        
        logger.info("Loading SigLIP Vision Model...")
        self.vision_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.vision_model = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            device_map=DEVICE,
            torch_dtype=torch.float16
        )
        
        logger.info("Initializing Geometric IGBundle Adapter...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # Load from Previous Unified Training Checkpoint
        unified_final = "igbundle_unified_training/final/adapter_weights.pt"
        if os.path.exists(unified_final):
            logger.info(f"RESUMING from Unified Training: {unified_final}")
            self.adapter.load_state_dict(torch.load(unified_final))
        else:
            logger.warning("No unified training checkpoint found! Starting fresh (Warning: This may take longer to break symmetry)")
        
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
            vis_feats = getattr(self, '_current_vision_features', None)
            adapted_out, geo_state = self.adapter(h_in, pixel_values=vis_feats)
            self._current_geo_state = geo_state
            adapted = adapted_out.to(orig_dtype)
            if isinstance(out, tuple):
                return (adapted,) + out[1:]
            return adapted
            
        target_layer.forward = adapter_hook
        
    def train(self):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=LEARNING_RATE)
        
        logger.info("Setting up Multi-Modal Datasets...")
        text_ds = TextDataset("unrestricted_long_context_data.jsonl", self.tokenizer)
        physics_ds = VisualTextDataset("data/physics_dynamics.jsonl", self.tokenizer, self.vision_processor)
        geo_ds = VisualTextDataset("data/geometric_visual_reasoning.jsonl", self.tokenizer, self.vision_processor)
        arc_ds = ARCDataset("data/arc_agi", self.tokenizer)
        gsm8k_ds = GSM8KDataset("h:/LLM-MANIFOLD/debug_samples_gsm8k_cot_zeroshot.json", self.tokenizer)

        # Higher weight on Physics/Geometry to force symmetry breaking in manifold
        train_ds = CompositeDataset(
            [text_ds, physics_ds, geo_ds, arc_ds, gsm8k_ds], 
            weights=[0.2, 0.25, 0.25, 0.15, 0.15] 
        )
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        iter_loader = iter(train_loader)
        
        logger.info(f"Starting Symmetry Breaking Loop (Target Steps: {MAX_STEPS})...")
        self.llm.train()
        
        symmetry_broken = False
        
        for step in range(MAX_STEPS):
            optimizer.zero_grad()
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            modality = batch["modality"][0]
            
            self._current_vision_features = None
            if modality == "visual":
                pixel_values = batch["pixel_values"].to(DEVICE).to(torch.float16)
                with torch.no_grad():
                    vis_out = self.vision_model(pixel_values)
                    self._current_vision_features = vis_out.last_hidden_state.to(torch.float32)
            
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
                
                # --- SYMMETRY MONITORING ---
                consensus = geo_losses.get('consensus_agreement', torch.tensor(0.0)).item()
                energy = geo_losses.get('hamiltonian_energy', torch.tensor(0.0)).item()
                
                if consensus > 2.0 and not symmetry_broken:
                    logger.info(f">>> SYMMETRY BREAK DETECTED at Step {step}: Consensus {consensus:.4f} > 2.0 <<<")
                    symmetry_broken = True
                
                if consensus > 2.0 and energy < 0.002:
                     logger.info(f">>> CRITICAL PHASE TRANSITION at Step {step}: Consensus {consensus:.4f} / Energy {energy:.6f} <<<")

                if step % 10 == 0:
                    logger.info(f"Step {step} [{modality}] Geo Losses: { {k: f'{v.item():.4f}' for k, v in geo_losses.items()} }")
            
            total_loss = loss_llm + 0.1 * loss_geo
            total_loss.backward()
            
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 10 == 0:
                logger.info(f"Step {step} [{modality}]: Total Loss = {total_loss.item():.4f} (LLM={loss_llm.item():.4f})")
            
            if (step + 1) % 50 == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step+1}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.adapter.state_dict(), os.path.join(save_path, "adapter_weights.pt"))
                logger.info(f"Saved checkpoint to {save_path}")

        logger.info("Training Complete.")
        final_path = os.path.join(OUTPUT_DIR, "final")
        os.makedirs(final_path, exist_ok=True)
        torch.save(self.adapter.state_dict(), os.path.join(final_path, "adapter_weights.pt"))
        logger.info(f"Saved final model to {final_path}")
        
if __name__ == "__main__":
    trainer = SymmetryTrainer()
    trainer.setup_model()
    trainer.train()
