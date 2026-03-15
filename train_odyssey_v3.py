"""
Epic 15: Foundational Multimodal Training Pipeline (train_odyssey_v3)

3-stage training pipeline for IGBundle with multimodal support:
  Stage 1 (Alignment): Frozen LLM + frozen vision encoder, train adapter + projector only
  Stage 2 (Instruction Tuning): QLoRA rank-32 on LLM, train adapter + projector
  Stage 3 (Domain Fine-Tune): Continue QLoRA + full adapter with geometric loss ramp

Supports:
  - Text-only, image-text, and mixed batches via CompositeStreamingDataset
  - SigLIP2 so400m vision encoder (1152-dim, 27x27=729 patches)
  - Delta-net fiber dynamics (use_delta_fiber flag)
  - SymplecticSPIDER optimizer with base/fiber param groups
  - Homotopy geometric loss scheduling
  - VRAM-aware gradient checkpointing for 5-8GB GPUs
  - IACS telemetry reporting (optional)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from datasets import load_dataset
import logging
import random
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import numpy as np

sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.modules.regularization import LipschitzPenalty
from igbundle.optimization.symplectic import SymplecticSPIDER
from igbundle.modules.spectral import SpectrallyNormalizedLinear

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_odyssey_v3.log"),
    ],
)
logger = logging.getLogger("OdysseyV3")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Vision encoder defaults (SigLIP2 so400m)
VISION_MODEL_ID = "google/siglip2-so400m-patch14-384"
VISION_DIM = 1152
VISION_PATCHES = 729  # 27x27

# ---------------------------------------------------------------------------
# Stage config
# ---------------------------------------------------------------------------
@dataclass
class StageConfig:
    name: str
    max_steps: int
    geo_lambda_max: float       # max homotopy weight for geo loss
    geo_ramp_steps: int         # steps to ramp geo loss 0 -> geo_lambda_max
    use_qlora: bool             # unfreeze LLM via QLoRA
    qlora_rank: int
    qlora_alpha: int
    base_lr: float
    fiber_lr: float
    text_weight: float          # sampling weight for text-only data
    multimodal_weight: float    # sampling weight for image-text data
    grad_accum: int
    max_seq_len: int
    checkpoint_every: int


STAGE_CONFIGS = {
    "alignment": StageConfig(
        name="alignment",
        max_steps=2000,
        geo_lambda_max=0.05,
        geo_ramp_steps=300,
        use_qlora=False,
        qlora_rank=0,
        qlora_alpha=0,
        base_lr=1e-4,
        fiber_lr=5e-3,
        text_weight=0.4,
        multimodal_weight=0.6,
        grad_accum=16,
        max_seq_len=512,
        checkpoint_every=200,
    ),
    "instruction": StageConfig(
        name="instruction",
        max_steps=4000,
        geo_lambda_max=0.1,
        geo_ramp_steps=500,
        use_qlora=True,
        qlora_rank=32,
        qlora_alpha=64,
        base_lr=5e-5,
        fiber_lr=3e-3,
        text_weight=0.5,
        multimodal_weight=0.5,
        grad_accum=16,
        max_seq_len=768,
        checkpoint_every=200,
    ),
    "domain": StageConfig(
        name="domain",
        max_steps=2000,
        geo_lambda_max=0.1,
        geo_ramp_steps=200,
        use_qlora=True,
        qlora_rank=32,
        qlora_alpha=64,
        base_lr=2e-5,
        fiber_lr=1e-3,
        text_weight=0.6,
        multimodal_weight=0.4,
        grad_accum=16,
        max_seq_len=512,
        checkpoint_every=200,
    ),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultimodalStreamingDataset(IterableDataset):
    """
    Composite streaming dataset that yields text-only and image-text samples
    with configurable weights. Each source can have a different transform.
    """

    def __init__(self, sources: List[dict], weights: List[float]):
        """
        Args:
            sources: list of dicts with keys:
                - 'dataset': HF IterableDataset or list-like
                - 'transform': callable(item) -> dict with input_ids, labels, pixel_values (optional)
                - 'name': str for logging
            weights: sampling weights per source
        """
        self.sources = sources
        self.weights = weights

    def __iter__(self):
        iterators = [iter(s["dataset"]) for s in self.sources]
        while True:
            idx = random.choices(range(len(self.sources)), weights=self.weights, k=1)[0]
            try:
                item = next(iterators[idx])
                transformed = self.sources[idx]["transform"](item)
                if transformed is not None:
                    yield transformed
            except StopIteration:
                iterators[idx] = iter(self.sources[idx]["dataset"])
            except Exception as e:
                logger.warning(f"Error from source '{self.sources[idx].get('name', idx)}': {e}")


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
def multimodal_collate(batch: List[dict]) -> dict:
    """Collate that handles optional pixel_values."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Collect pixel_values if any sample has them
    pv_list = [b.get("pixel_values") for b in batch]
    if any(pv is not None for pv in pv_list):
        # For samples without images, use zeros
        ref = next(pv for pv in pv_list if pv is not None)
        filled = [pv if pv is not None else torch.zeros_like(ref) for pv in pv_list]
        result["pixel_values"] = torch.stack(filled)
        # Mask: which samples actually have images
        result["has_image"] = torch.tensor(
            [pv is not None for pv in pv_list], dtype=torch.bool
        )

    return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class OdysseyV3Trainer:
    def __init__(self, args):
        self.args = args
        self.stage_cfg: StageConfig = STAGE_CONFIGS[args.stage]

        self.config = IGBundleConfig(
            hidden_size=3584,
            num_components=8,
            latent_dim=64,
            num_categories=16,
            use_dynamics=True,
            use_geodesic_attn=True,
            supported_modalities=["vision", "text"],
            use_delta_fiber=args.use_delta_fiber,
            delta_mem_dim=64,
            delta_num_heads=4,
        )

        self.tokenizer = None
        self.llm = None
        self.adapter = None
        self.optimizer = None
        self.vision_model = None
        self.vision_processor = None
        self._current_geo_state = None
        self.start_step = 0

    # --- Vision encoder ---------------------------------------------------
    def _load_vision_encoder(self):
        """Load SigLIP2 vision encoder on CPU (moved to GPU per-batch)."""
        if not self.args.enable_vision:
            logger.info("Vision disabled — skipping SigLIP2 load")
            return

        try:
            from transformers import SiglipModel, SiglipImageProcessor

            logger.info(f"Loading vision encoder: {VISION_MODEL_ID}")
            # Use SiglipModel directly — AutoModel can misroute for SigLIP2
            # Only need the vision tower, but loading the full model gives us
            # get_image_features() which handles the projection
            try:
                self.vision_processor = SiglipImageProcessor.from_pretrained(VISION_MODEL_ID)
            except Exception:
                self.vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
            self.vision_model = SiglipModel.from_pretrained(
                VISION_MODEL_ID, torch_dtype=torch.float16
            )
            self.vision_model.eval()
            self.vision_model.requires_grad_(False)
            # Stay on CPU — move to GPU per-batch to save VRAM
            logger.info(
                f"Vision encoder loaded on CPU "
                f"({sum(p.numel() for p in self.vision_model.parameters()) / 1e6:.1f}M params)"
            )
        except Exception as e:
            logger.warning(f"Failed to load vision encoder: {e}. Falling back to text-only.")
            self.vision_model = None
            self.vision_processor = None

    # --- Spectral norm injection ------------------------------------------
    def _inject_spectral_norm(self, model):
        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "vision_proj" in name:
                replacements.append((name, module))

        for name, module in replacements:
            sn_layer = SpectrallyNormalizedLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
            )
            with torch.no_grad():
                sn_layer.linear.weight.copy_(module.weight)
                if module.bias is not None:
                    sn_layer.linear.bias.copy_(module.bias)

            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], sn_layer)
            logger.info(f"SpectralNorm: {name}")

    # --- Adapter hook -----------------------------------------------------
    def _inject_adapter_hook(self):
        target_layer_idx = 12
        target_layer = self.llm.model.layers[target_layer_idx]
        original_forward = target_layer.forward
        self._current_geo_state = None

        def adapter_hook(hidden_states, *args, **kwargs):
            out = original_forward(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            orig_dtype = h.dtype
            h_in = h.to(DEVICE).to(torch.float32)

            # Pass pixel_values if available (set per-batch in train loop)
            pv = getattr(self, "_batch_pixel_values", None)
            adapted_out, geo_state = self.adapter(h_in, pixel_values=pv)
            self._current_geo_state = geo_state

            adapted = adapted_out.to(orig_dtype)
            if isinstance(out, tuple):
                return (adapted,) + out[1:]
            return adapted

        target_layer.forward = adapter_hook
        logger.info(f"Adapter hook injected at layer {target_layer_idx}")

    # --- Checkpoint loading -----------------------------------------------
    def _load_checkpoint(self):
        ckpt_dir = self.args.checkpoint_dir
        if not ckpt_dir or not os.path.exists(ckpt_dir):
            return

        import re
        # Match both checkpoint-STEP and checkpoint-STAGE-STEP patterns
        dirs = [d for d in os.listdir(ckpt_dir)
                if re.match(r"^checkpoint-(\w+-)?(\d+)$", d)]
        if not dirs:
            return

        def extract_step(name):
            m = re.match(r"^checkpoint-(?:\w+-)?(\d+)$", name)
            return int(m.group(1)) if m else 0

        latest = sorted(dirs, key=extract_step)[-1]
        self.start_step = extract_step(latest)
        ckpt_path = os.path.join(ckpt_dir, latest, "adapter_weights.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming from {ckpt_path} (step {self.start_step})")
            state = torch.load(ckpt_path, map_location=DEVICE)
            self.adapter.load_state_dict(state, strict=False)

    # --- QLoRA setup ------------------------------------------------------
    def _setup_qlora(self):
        """Apply QLoRA adapters to frozen LLM for Stage 2/3."""
        if not self.stage_cfg.use_qlora:
            return

        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=self.stage_cfg.qlora_rank,
                lora_alpha=self.stage_cfg.qlora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm = get_peft_model(self.llm, lora_config)
            trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llm.parameters())
            logger.info(
                f"QLoRA applied: rank={self.stage_cfg.qlora_rank}, "
                f"trainable={trainable/1e6:.2f}M / {total/1e6:.1f}M "
                f"({100*trainable/total:.2f}%)"
            )
        except ImportError:
            logger.error("peft not installed — QLoRA unavailable. pip install peft")
            raise

    # --- Dataset preparation ----------------------------------------------
    def _make_text_transform(self, max_len: int):
        """Returns transform for text-only Q&A JSONL data."""
        def transform(item):
            text = f"Question: {item.get('input', '')}\nAnswer: {item.get('output', '')}"
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding="max_length",
            )
            return {
                "input_ids": enc.input_ids.squeeze(0),
                "attention_mask": enc.attention_mask.squeeze(0),
                "labels": enc.input_ids.squeeze(0),
            }
        return transform

    def _make_multimodal_transform(self, max_len: int):
        """Returns transform for LLaVA-Instruct / ShareGPT4V format.

        Format: {"image": "path.jpg", "conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
        The <image> tag in conversation text is stripped (vision features injected separately).
        """
        def transform(item):
            convs = item.get("conversations", [])
            if not convs:
                text = item.get("text", "")
            else:
                parts = []
                for c in convs:
                    role = "Question" if c.get("from") == "human" else "Answer"
                    val = c.get("value", "").replace("<image>\n", "").replace("<image>", "")
                    parts.append(f"{role}: {val}")
                text = "\n".join(parts)

            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_len, padding="max_length",
            )
            result = {
                "input_ids": enc.input_ids.squeeze(0),
                "attention_mask": enc.attention_mask.squeeze(0),
                "labels": enc.input_ids.squeeze(0),
            }

            if self.vision_model is not None and self.vision_processor is not None:
                image = item.get("image")
                if image is not None:
                    try:
                        from PIL import Image as PILImage
                        if isinstance(image, str):
                            image = PILImage.open(image).convert("RGB")
                        inputs = self.vision_processor(images=image, return_tensors="pt")
                        result["pixel_values"] = inputs["pixel_values"].squeeze(0)
                    except Exception:
                        pass
            return result

        return transform

    def _make_cauldron_transform(self, max_len: int):
        """Returns transform for The Cauldron (ChartQA) format.

        Format: {"image_paths": ["path.png"], "texts": [{"user":"...","assistant":"..."}]}
        """
        def transform(item):
            texts = item.get("texts", [])
            if texts:
                t = texts[0]
                text = f"Question: {t.get('user', '')}\nAnswer: {t.get('assistant', '')}"
            else:
                text = ""

            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_len, padding="max_length",
            )
            result = {
                "input_ids": enc.input_ids.squeeze(0),
                "attention_mask": enc.attention_mask.squeeze(0),
                "labels": enc.input_ids.squeeze(0),
            }

            if self.vision_model is not None and self.vision_processor is not None:
                paths = item.get("image_paths", [])
                if paths:
                    try:
                        from PIL import Image as PILImage
                        img = PILImage.open(paths[0]).convert("RGB")
                        inputs = self.vision_processor(images=img, return_tensors="pt")
                        result["pixel_values"] = inputs["pixel_values"].squeeze(0)
                    except Exception:
                        pass
            return result

        return transform

    def _prepare_datasets(self):
        sc = self.stage_cfg
        sources = []
        weights = []

        # Text-only source: full_scale_train.jsonl
        text_path = "data/full_scale_train.jsonl"
        if os.path.exists(text_path):
            ds_text = load_dataset(
                "json", data_files={"train": text_path}, split="train", streaming=True
            )
            sources.append({
                "dataset": ds_text,
                "transform": self._make_text_transform(sc.max_seq_len),
                "name": "text_reasoning",
            })
            weights.append(sc.text_weight)
            logger.info(f"Text source: {text_path} (weight={sc.text_weight})")

        # Geometric visual reasoning (small, text-only format but geometric domain)
        geo_path = "data/geometric_visual_reasoning.jsonl"
        if os.path.exists(geo_path):
            ds_geo = load_dataset(
                "json", data_files={"train": geo_path}, split="train", streaming=True
            )
            sources.append({
                "dataset": ds_geo,
                "transform": self._make_text_transform(sc.max_seq_len),
                "name": "geometric_reasoning",
            })
            # Small dataset, low weight
            weights.append(0.05)
            logger.info(f"Geometric source: {geo_path} (weight=0.05)")

        # Physics dynamics (small, text-only)
        phys_path = "data/physics_dynamics.jsonl"
        if os.path.exists(phys_path):
            ds_phys = load_dataset(
                "json", data_files={"train": phys_path}, split="train", streaming=True
            )
            sources.append({
                "dataset": ds_phys,
                "transform": self._make_text_transform(sc.max_seq_len),
                "name": "physics_dynamics",
            })
            weights.append(0.05)
            logger.info(f"Physics source: {phys_path} (weight=0.05)")

        # Multimodal sources — prepared by Epic 18 (Gemini)
        # Weight is split across available multimodal sources
        mm_sources_found = 0
        if self.args.enable_vision:
            mm_configs = [
                ("data/multimodal/llava_instruct_subset.jsonl", "llava_instruct",
                 self._make_multimodal_transform(sc.max_seq_len)),
                ("data/multimodal/sharegpt4v_subset.jsonl", "sharegpt4v",
                 self._make_multimodal_transform(sc.max_seq_len)),
                ("data/multimodal/cauldron_chartqa_subset.jsonl", "cauldron_chartqa",
                 self._make_cauldron_transform(sc.max_seq_len)),
            ]
            for mm_path, mm_name, mm_transform in mm_configs:
                if os.path.exists(mm_path):
                    ds_mm = load_dataset(
                        "json", data_files={"train": mm_path}, split="train", streaming=True
                    )
                    sources.append({
                        "dataset": ds_mm,
                        "transform": mm_transform,
                        "name": mm_name,
                    })
                    # Split multimodal weight evenly across found sources
                    weights.append(sc.multimodal_weight / 3.0)
                    mm_sources_found += 1
                    logger.info(f"Multimodal source: {mm_path} ({mm_name})")

            if mm_sources_found == 0:
                logger.warning(
                    "No multimodal data found in data/multimodal/. "
                    "Run Epic 18 (data/prepare_multimodal.py) first. Proceeding text-only."
                )

        if not sources:
            raise RuntimeError("No training data found in data/")

        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        self.train_ds = MultimodalStreamingDataset(sources, weights)
        logger.info(f"Dataset ready: {len(sources)} sources, weights={[f'{w:.2f}' for w in weights]}")

    # --- Vision feature extraction ----------------------------------------
    def _extract_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract patch features from SigLIP2 on GPU, then move back to save VRAM."""
        if self.vision_model is None:
            return None

        with torch.no_grad():
            # Move to GPU for forward pass
            pv = pixel_values.to(DEVICE).half()
            vm = self.vision_model.to(DEVICE)
            features = vm.get_image_features(pixel_values=pv)
            # Move vision model back to CPU
            self.vision_model.to("cpu")
            torch.cuda.empty_cache()
            return features.float()  # (B, N_patches, vision_dim)

    # --- Optimizer setup --------------------------------------------------
    def _setup_optimizer(self):
        sc = self.stage_cfg
        base_params = []
        fiber_params = []

        for name, p in self.adapter.named_parameters():
            if not p.requires_grad:
                continue
            if "fiber_store" in name or "latent_store" in name:
                base_params.append(p)
                p._is_base = True
            else:
                fiber_params.append(p)
                p._is_base = False

        param_groups = [
            {"params": base_params, "is_base": True, "base_lr": sc.base_lr, "base_momentum": 0.0},
            {"params": fiber_params, "is_base": False, "fiber_lr": sc.fiber_lr, "fiber_momentum": 0.9},
        ]

        # Add QLoRA params if active
        if sc.use_qlora:
            qlora_params = [p for p in self.llm.parameters() if p.requires_grad]
            if qlora_params:
                param_groups.append({
                    "params": qlora_params,
                    "is_base": True,
                    "base_lr": sc.base_lr * 0.5,  # lower LR for LLM LoRA
                    "base_momentum": 0.0,
                })
                logger.info(f"QLoRA param group: {len(qlora_params)} params, lr={sc.base_lr * 0.5}")

        logger.info(
            f"Optimizer: base={len(base_params)}, fiber={len(fiber_params)} params"
        )

        self.optimizer = SymplecticSPIDER(param_groups, c=5.0, period=100)

    # --- IACS telemetry ---------------------------------------------------
    def _report_telemetry(self, step: int, metrics: dict):
        """Post telemetry to IACS if available."""
        if not self.args.iacs_telemetry:
            return
        try:
            import requests
            requests.post(
                "http://localhost:9100/api/v1/telemetry",
                json={"agent_id": "odyssey_v3", "step": step, "metrics": metrics},
                timeout=2,
            )
        except Exception:
            pass  # Non-critical

    # --- Setup ------------------------------------------------------------
    def setup(self):
        logger.info(f"=== OdysseyV3 Stage: {self.stage_cfg.name} ===")

        # Tokenizer & LLM
        model_id = self.args.model_id
        logger.info(f"Loading tokenizer & LLM from {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm.requires_grad_(False)

        # QLoRA (Stage 2/3)
        self._setup_qlora()

        # Vision encoder
        self._load_vision_encoder()

        # Geometric adapter
        logger.info("Creating geometric adapter...")
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)

        # Load checkpoint
        self._load_checkpoint()

        # Spectral norm + hook
        self._inject_spectral_norm(self.adapter)
        self.adapter.to(DEVICE)
        self._inject_adapter_hook()
        self.adapter.train()

        # Enable gradient checkpointing if VRAM is tight
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
            if free_vram < 6.0:
                logger.info(f"Low VRAM ({free_vram:.1f}GB free) — enabling gradient checkpointing")
                if hasattr(self.llm, "gradient_checkpointing_enable"):
                    self.llm.gradient_checkpointing_enable()

        # Optimizer
        self._setup_optimizer()

        # Datasets
        self._prepare_datasets()

        logger.info("Setup complete")

    # --- Training loop ----------------------------------------------------
    def train(self):
        sc = self.stage_cfg
        logger.info(
            f"Training: stage={sc.name}, steps={self.start_step}->{sc.max_steps}, "
            f"grad_accum={sc.grad_accum}, geo_lambda_max={sc.geo_lambda_max}"
        )

        train_loader = DataLoader(
            self.train_ds,
            batch_size=1,
            collate_fn=multimodal_collate,
        )
        iter_loader = iter(train_loader)
        step = self.start_step

        while step < sc.max_steps:
            # --- Get batch ---
            try:
                batch = next(iter_loader)
            except (StopIteration, Exception) as e:
                if not isinstance(e, StopIteration):
                    logger.error(f"Batch error: {e}")
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # --- Vision features (if present) ---
            self._batch_pixel_values = None
            if "pixel_values" in batch and batch.get("has_image", torch.tensor([False])).any():
                pv = batch["pixel_values"]
                vis_features = self._extract_vision_features(pv)
                if vis_features is not None:
                    self._batch_pixel_values = vis_features

            # --- Forward pass ---
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss_llm = outputs.loss
            loss_geo = torch.tensor(0.0, device=DEVICE)
            geo_losses = {}

            if self._current_geo_state and hasattr(self.adapter, "compute_geometric_losses"):
                geo_losses = self.adapter.compute_geometric_losses(self._current_geo_state)

                # Entropy ramp for resumed checkpoints
                resume_step = self.start_step
                entropy_ramp = (
                    min(1.0, max(0.0, (step - resume_step) / 200.0))
                    if resume_step > 0
                    else 1.0
                )

                for k, v in geo_losses.items():
                    if k in ("fiber_diversity", "fiber_entropy"):
                        loss_geo = loss_geo + v * entropy_ramp
                    else:
                        loss_geo = loss_geo + v

            # Homotopy schedule
            lambda_t = sc.geo_lambda_max * min(1.0, step / max(1, sc.geo_ramp_steps))
            total_loss = loss_llm + lambda_t * loss_geo

            # --- Backward ---
            total_loss.backward()

            if (step + 1) % sc.grad_accum == 0:
                # Surgical NaN zeroing (no external clip — SymplecticSPIDER handles it)
                nan_count = 0
                for name, p in self.adapter.named_parameters():
                    if p.grad is not None:
                        bad = torch.isnan(p.grad) | torch.isinf(p.grad)
                        if bad.any():
                            nan_count += bad.sum().item()
                            p.grad[bad] = 0.0

                self.optimizer.step()
                self.optimizer.zero_grad()

                # VRAM cleanup
                if step % 64 == 0:
                    torch.cuda.empty_cache()

                # --- Logging ---
                if (step + 1) % (sc.grad_accum * 2) == 0 or step < 10:
                    geo_detail = ""
                    S_actual = None
                    if geo_losses:
                        parts = [f"{k}={v.item() if hasattr(v, 'item') else v:.3f}"
                                 for k, v in sorted(geo_losses.items())]
                        geo_detail = f" [{', '.join(parts)}]"

                    if (self._current_geo_state is not None
                            and self._current_geo_state.fiber_sections is not None):
                        with torch.no_grad():
                            p = self._current_geo_state.fiber_sections.clamp(min=1e-8)
                            S_actual = -(p * p.log()).sum(dim=-1).mean().item()
                            geo_detail += f" S={S_actual:.4f}"

                    nan_info = f" NaN={nan_count}" if nan_count > 0 else ""
                    logger.info(
                        f"[{sc.name}] step={step} loss={loss_llm.item():.4f} "
                        f"geo={loss_geo.item():.3f} lam={lambda_t:.4f}{geo_detail}{nan_info}"
                    )

                    # IACS telemetry
                    metrics = {
                        "loss_llm": loss_llm.item(),
                        "loss_geo": loss_geo.item(),
                        "lambda_t": lambda_t,
                        "nan_count": nan_count,
                    }
                    if S_actual is not None:
                        metrics["S"] = S_actual
                    self._report_telemetry(step, metrics)

            # --- Checkpoint ---
            step += 1
            if step % sc.checkpoint_every == 0 or step == sc.max_steps:
                ckpt_dir = os.path.join(
                    self.args.output_dir, f"checkpoint-{sc.name}-{step}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    self.adapter.state_dict(),
                    os.path.join(ckpt_dir, "adapter_weights.pt"),
                )
                # Save QLoRA weights separately if active
                if sc.use_qlora and hasattr(self.llm, "save_pretrained"):
                    qlora_dir = os.path.join(ckpt_dir, "qlora")
                    self.llm.save_pretrained(qlora_dir)

                logger.info(f"Checkpoint saved: {ckpt_dir}")

        logger.info(f"Stage '{sc.name}' complete at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="OdysseyV3: Foundational Multimodal Training")
    parser.add_argument(
        "--stage",
        choices=["alignment", "instruction", "domain"],
        default="alignment",
        help="Training stage (default: alignment)",
    )
    parser.add_argument(
        "--model-id",
        default="h:/LLM-MANIFOLD/igbundle_qwen7b_cp600",
        help="Base model path",
    )
    parser.add_argument(
        "--output-dir",
        default="igbundle_odyssey_v3",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to resume from (scans for latest checkpoint-*)",
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable SigLIP2 vision encoder for multimodal training",
    )
    parser.add_argument(
        "--use-delta-fiber",
        action="store_true",
        help="Use delta-net fiber dynamics (Epic 17b)",
    )
    parser.add_argument(
        "--iacs-telemetry",
        action="store_true",
        help="Report training metrics to IACS telemetry endpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = OdysseyV3Trainer(args)

    try:
        trainer.setup()
        trainer.train()
        logger.info("OdysseyV3 training complete.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        # Emergency checkpoint
        if trainer.adapter is not None:
            try:
                ckpt_dir = os.path.join(
                    args.output_dir,
                    f"checkpoint-emergency-{int(time.time())}",
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    trainer.adapter.state_dict(),
                    os.path.join(ckpt_dir, "adapter_weights.pt"),
                )
                logger.info(f"Emergency checkpoint: {ckpt_dir}")
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
