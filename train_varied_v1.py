
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    SiglipVisionModel, 
    SiglipImageProcessor
)
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
import logging
import argparse

# Path Setup
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
        logging.FileHandler("train_varied.log")
    ]
)
logger = logging.getLogger("VariedTrainer")

# Constants
MODEL_ID = "../igbundle_qwen7b_cp600"
OUTPUT_DIR = "igbundle_varied_training"
MAX_STEPS = 200
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4 # Reduced for stability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HFStreamDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer, config_name=None, split="train", text_column="text", prefix="", max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = prefix
        self.text_column = text_column
        self.dataset_name = dataset_name
        
        logger.info(f"Initializing Stream: {dataset_name} ({config_name or 'default'})")
        # Load streaming dataset with optional config
        if config_name:
            self.dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        
    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                item = next(iterator)
                # Text Extraction Logic
                text = ""
                if self.dataset_name == "microsoft/orca-math-word-problems-200k":
                     q = item.get("question", "")
                     a = item.get("answer", "")
                     text = f"Math Problem:\n{q}\n\nSolution:\n{a}"
                elif self.dataset_name == "OpenAssistant/oasst_top1_2023-08-25":
                     text = item.get("text", "") 
                else:
                     text = item.get(self.text_column, "")
                
                text = self.prefix + text
                
                # Tokenization
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length"
                )
                
                yield {
                    "input_ids": enc.input_ids.squeeze(0),
                    "attention_mask": enc.attention_mask.squeeze(0),
                    "labels": enc.input_ids.squeeze(0),
                    "modality": self.dataset_name
                }
            except StopIteration:
                return # End of stream
            except Exception as e:
                logger.warning(f"Error processing item from {self.dataset_name}: {e}")
                continue

class MixedStreamLoader:
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        self.iterators = [iter(ds) for ds in datasets]
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # Weighted selection
        choice_idx = torch.multinomial(torch.tensor(self.weights), 1).item()
        try:
            return next(self.iterators[choice_idx])
        except StopIteration:
            # Restart iterator if exhausted (infinite stream)
            self.iterators[choice_idx] = iter(self.datasets[choice_idx])
            return next(self.iterators[choice_idx])

class VariedTrainer:
    def __init__(self):
        self.config = IGBundleConfig(
            hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
            use_dynamics=True, use_geodesic_attn=True, supported_modalities=["text", "logic"]
        )
        self.tokenizer = None
        self.llm = None
        self.adapter = None
        
    def setup(self):
        logger.info(f"Loading Base Model: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
        self.llm = prepare_model_for_kbit_training(self.llm)
        self.llm.config.use_cache = False
        
        logger.info("Initializing Geometric Adapter...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # Resume from previous best if available
        # But user wants varied training, maybe start fresh or continue? 
        # Typically continue. Let's look for `igbundle_unified_training/final`
        prev_ckpt = "igbundle_unified_training/final/adapter_weights.pt"
        if os.path.exists(prev_ckpt):
            logger.info(f"Loading Base Adapter State from {prev_ckpt}")
            # Use strict=False because we dropped 'vision' modality but checkpoint has it
            keys = self.adapter.load_state_dict(torch.load(prev_ckpt), strict=False)
            logger.info(f"Loaded keys: {keys}")
            
        self._inject_adapter()
        
    def _inject_adapter(self):
        target = self.llm.model.layers[12]
        orig_fwd = target.forward
        self._geo_state = None
        
        def hook(hidden_states, *args, **kwargs):
            out = orig_fwd(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            orig_dtype = h.dtype
            h_in = h.to(DEVICE).to(torch.float32)
            
            adapted_out, geo_state = self.adapter(h_in)
            self._geo_state = geo_state
            
            adapted = adapted_out.to(orig_dtype)
            if isinstance(out, tuple): return (adapted,) + out[1:]
            return adapted
            
        target.forward = hook
        
    def train(self, dry_run=False):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=LEARNING_RATE)
        
        # Setup Streams
        # Cosmopedia requires a config. Using 'stories' for high-quality synthetic narrative.
        ds_cosmo = HFStreamDataset("HuggingFaceTB/cosmopedia", self.tokenizer, config_name="stories", text_column="text")
        ds_orca = HFStreamDataset("microsoft/orca-math-word-problems-200k", self.tokenizer)
        ds_oasst = HFStreamDataset("OpenAssistant/oasst_top1_2023-08-25", self.tokenizer)
        
        loader = MixedStreamLoader(
            datasets=[ds_cosmo, ds_orca, ds_oasst],
            weights=[0.4, 0.3, 0.3]
        )
        
        steps = 10 if dry_run else MAX_STEPS
        logger.info(f"Starting Training. Steps={steps}, DryRun={dry_run}")
        
        loss_ema = None
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Since BATCH_SIZE=1 and it's a stream, we fetch manually
            # To support BATCH_SIZE > 1, needs Collate.
            # For simplicity, logic inside loop assumes BS=1 or we accumulate.
            
            batch = next(loader)
            input_ids = batch["input_ids"].to(DEVICE).unsqueeze(0) # Add batch dim
            attention = batch["attention_mask"].to(DEVICE).unsqueeze(0)
            labels = batch["labels"].to(DEVICE).unsqueeze(0)
            modality = batch["modality"]
            
            outputs = self.llm(input_ids=input_ids, attention_mask=attention, labels=labels)
            
            loss_llm = outputs.loss
            
            # Geo Loss (Capped)
            loss_geo = 0.0
            if self._geo_state:
                geo_losses = self.adapter.compute_geometric_losses(self._geo_state)
                # Clamp geo loss to prevent spikes from out-of-distribution inputs
                raw_geo_sum = sum(geo_losses.values())
                loss_geo = torch.clamp(raw_geo_sum, max=10.0) 
                
            total_loss = loss_llm + 0.1 * loss_geo
            
            if torch.isnan(total_loss):
                 logger.error("NaN Loss detected! Skipping step.")
                 continue
                 
            total_loss.backward()
            
            # Gradient Clipping (Anti-Spike)
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
            
            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            # Logging
            if loss_ema is None: loss_ema = total_loss.item()
            else: loss_ema = 0.9 * loss_ema + 0.1 * total_loss.item()
            
            if step % 10 == 0:
                logger.info(f"Step {step} [{modality}] Loss={total_loss.item():.4f} (EMA={loss_ema:.4f})")
                
            # Checkpoint
            if not dry_run and (step + 1) % 50 == 0:
                s_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step+1}")
                os.makedirs(s_path, exist_ok=True)
                torch.save(self.adapter.state_dict(), os.path.join(s_path, "adapter_weights.pt"))
                
        if not dry_run:
            f_path = os.path.join(OUTPUT_DIR, "final")
            os.makedirs(f_path, exist_ok=True)
            torch.save(self.adapter.state_dict(), os.path.join(f_path, "adapter_weights.pt"))
            logger.info("Varied Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    trainer = VariedTrainer()
    trainer.setup()
    trainer.train(dry_run=args.dry_run)
