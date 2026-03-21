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
import zlib
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from collections import deque
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

BASELINE_EOS_RATIO = 0.64
PROMOTION_EOS_THRESHOLD = 0.64
PROMOTION_EVAL_S_THRESHOLD = 0.60
PROMOTION_LEAKAGE_THRESHOLD = 0.0
PROMOTION_DEGEN_THRESHOLD = 0.02
PROMOTION_BENCHMARK_FLOOR = 0.80
ENTROPY_TARGET_BAND = (0.80, 1.40)
ENTROPY_MARGINAL_BAND = (0.60, 0.80)
ENTROPY_COLLAPSE_THRESHOLD = 0.30
TRAINING_S_WINDOW = 50
EVAL_SAMPLE_COUNT = 50
BENCHMARK_PROMPTS = [
    {
        "prompt": "Maria has 3x as many apples as Joao. After Joao buys 12 more and Maria gives away a third of hers, they have the same amount. How many did Maria start with?",
        "check_words": ["36"],
        "anti_words": ["48", "24", "72"],
        "weight": 1.5,
    },
    {
        "prompt": "If all Bloops are Razzles, and all Razzles are Lazzles, but no Lazzles are Wazzles, what can we conclude about Bloops and Wazzles?",
        "check_words": ["no", "bloops", "wazzles"],
        "anti_words": [],
        "weight": 1.0,
    },
    {
        "prompt": "The element with atomic number 79 is gold. It shares a group with the element used in most electrical wiring. What are both elements?",
        "check_words": ["gold", "copper"],
        "anti_words": ["silver"],
        "weight": 1.0,
    },
    {
        "prompt": "What is the difference between a tensor, a vector, and a scalar in differential geometry? Give one concrete example of each.",
        "check_words": ["tensor", "vector", "scalar", "transform"],
        "anti_words": [],
        "weight": 1.0,
    },
    {
        "prompt": "What is the second law of thermodynamics, and how does it relate to the arrow of time?",
        "check_words": ["entropy", "increase", "disorder"],
        "anti_words": [],
        "weight": 1.0,
    },
    {
        "prompt": "A train travels from A to B at 80 km/h and returns at 120 km/h. What is the average speed for the round trip?",
        "check_words": ["96"],
        "anti_words": ["100"],
        "weight": 1.5,
    },
    {
        "prompt": "The philosopher who wrote the Critique of Pure Reason was born in the same city where he died. That city is now in which country?",
        "check_words": ["russia", "kaliningrad", "kant"],
        "anti_words": [],
        "weight": 1.0,
    },
    {
        "prompt": "Write a step-by-step proof that there are infinitely many primes.",
        "check_words": ["assume", "contradiction", "finite", "product"],
        "anti_words": [],
        "weight": 1.2,
    },
]
EVAL_SYSTEM_PROMPT = "Answer directly with step-by-step reasoning. Be concise."
LEAKAGE_FRAGMENTS = [
    "answer directly with step-by-step",
    "<|im_start|>",
    "<|im_end|>",
    "<|im_start|>system",
    "<|im_start|>assistant",
    "user:",
    "assistant:",
]

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
    entropy_loss_scale: float = 1.0  # multiplier for fiber_diversity + fiber_entropy losses
    eos_ce_weight: float = 2.0
    eos_margin_weight: float = 1.0
    eos_stop_kl_weight: float = 0.25
    eos_norm_weight: float = 0.05


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
        entropy_loss_scale=3.0,  # 3x boost to prevent S collapse seen in Stage 1
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
    "alignment-eos": StageConfig(
        name="alignment-eos",
        max_steps=2000,
        geo_lambda_max=0.05,
        geo_ramp_steps=200,
        use_qlora=False,           # adapter-only — keep base frozen
        qlora_rank=0,
        qlora_alpha=0,
        base_lr=5e-5,              # slightly lower: fine-tuning, not cold start
        fiber_lr=3e-3,
        text_weight=0.5,
        multimodal_weight=0.5,
        grad_accum=16,
        max_seq_len=512,
        checkpoint_every=200,
        entropy_loss_scale=5.0,    # strong: S collapsed to 0.006 with 2.0
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
        self.best_eval = {
            "eos_ratio": float("-inf"),
            "eval_S_mean": float("-inf"),
            "degeneration_rate": float("inf"),
            "step": None,
            "checkpoint_dir": None,
        }
        self.checkpoint_eval_history = {}
        self.train_entropy_history = deque(maxlen=TRAINING_S_WINDOW)
        self.promoted_checkpoint = None
        self.promotion_benchmark_floor = PROMOTION_BENCHMARK_FLOOR

    def _apply_runtime_overrides(self):
        overrides = {
            "max_steps": getattr(self.args, "max_steps", None),
            "checkpoint_every": getattr(self.args, "checkpoint_every", None),
            "entropy_loss_scale": getattr(self.args, "entropy_loss_scale", None),
            "base_lr": getattr(self.args, "base_lr", None),
            "fiber_lr": getattr(self.args, "fiber_lr", None),
            "geo_lambda_max": getattr(self.args, "geo_lambda_max", None),
            "geo_ramp_steps": getattr(self.args, "geo_ramp_steps", None),
            "eos_ce_weight": getattr(self.args, "eos_ce_weight", None),
            "eos_margin_weight": getattr(self.args, "eos_margin_weight", None),
            "eos_stop_kl_weight": getattr(self.args, "eos_stop_kl_weight", None),
            "eos_norm_weight": getattr(self.args, "eos_norm_weight", None),
        }
        for field_name, value in overrides.items():
            if value is None:
                continue
            setattr(self.stage_cfg, field_name, value)
            logger.info(f"Runtime override: {field_name}={value}")

    def _resolve_special_token_id(
        self, token_text: str, fallback_id: Optional[int] = None
    ) -> Optional[int]:
        """Resolve chat-template tokens even when tokenizer lookup falls back to UNK."""
        token_id = self.tokenizer.convert_tokens_to_ids(token_text)
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk_token_id:
            return fallback_id
        return token_id

    def _stop_token_ids(self) -> List[int]:
        eos_ids = []
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_ids.append(eos_token_id)
        im_end_id = self._resolve_special_token_id("<|im_end|>", fallback_id=151645)
        if im_end_id is not None and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
        return eos_ids

    def _compute_entropy_from_geo_state(self) -> Optional[float]:
        if (
            self._current_geo_state is None
            or self._current_geo_state.fiber_sections is None
        ):
            return None
        with torch.no_grad():
            p = self._current_geo_state.fiber_sections.clamp(min=1e-8)
            return float((-(p * p.log()).sum(dim=-1).mean()).item())

    def _apply_eval_chat_template(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return (
            f"<|im_start|>system\n{EVAL_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _detect_leakage(self, text: str) -> bool:
        lower = text.lower()
        return any(fragment in lower for fragment in LEAKAGE_FRAGMENTS)

    def _detect_degeneration(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        if len(lowered) > 50:
            compressed = len(zlib.compress(lowered.encode("utf-8")))
            ratio = compressed / max(1, len(lowered.encode("utf-8")))
            if ratio > 0.95:
                return True

        tokens = lowered.split()
        for ngram_size in (2, 3, 4):
            if len(tokens) < ngram_size * 4:
                continue
            last_ngram = tokens[-ngram_size:]
            reps = 1
            idx = len(tokens) - ngram_size * 2
            while idx >= 0 and tokens[idx:idx + ngram_size] == last_ngram:
                reps += 1
                idx -= ngram_size
            if reps >= 4:
                return True

        sentences = [s.strip() for s in lowered.split(".") if s.strip()]
        if len(sentences) >= 4:
            tail = sentences[-1]
            if tail and sum(1 for s in sentences if s == tail) >= 4:
                return True

        return False

    def _score_benchmark_response(self, response: str, prompt_info: dict) -> float:
        lower = response.lower()
        score = 0.0

        for word in prompt_info.get("check_words", []):
            if word.lower() in lower:
                score += 1.0 / max(1, len(prompt_info["check_words"]))
        for word in prompt_info.get("anti_words", []):
            if word.lower() in lower:
                score -= 0.3

        word_count = len(response.split())
        if word_count < 15:
            score -= 0.3
        elif word_count > 400:
            score -= 0.1

        if self._detect_degeneration(response):
            score -= 0.5
        if self._detect_leakage(response):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _write_json(self, path: str, payload: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_existing_promoted_checkpoint(self) -> Optional[dict]:
        path = os.path.join(self.args.output_dir, "promoted_checkpoint.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning(f"Could not read promoted checkpoint metadata: {path}")
            return None

    def _initialize_promotion_state(self):
        os.makedirs(self.args.output_dir, exist_ok=True)

        existing = self._load_existing_promoted_checkpoint()
        if existing is not None:
            self.promoted_checkpoint = existing
            baseline_score = existing.get("benchmark_score")
            if isinstance(baseline_score, (int, float)):
                self.promotion_benchmark_floor = max(
                    PROMOTION_BENCHMARK_FLOOR, 0.8 * float(baseline_score)
                )
            return

        promoted = {
            "checkpoint_name": "checkpoint-alignment-eos-1400",
            "checkpoint_dir": os.path.join(
                self.args.checkpoint_dir or "", "checkpoint-alignment-eos-1400"
            ),
            "step": 1400,
            "eos_ratio": BASELINE_EOS_RATIO,
            "promotion_status": True,
            "source": "baseline_reference",
            "baseline_benchmark_floor": PROMOTION_BENCHMARK_FLOOR,
        }
        self.promoted_checkpoint = promoted
        self._write_json(
            os.path.join(self.args.output_dir, "promoted_checkpoint.json"),
            promoted,
        )

    def _is_better_best_checkpoint(self, metrics: dict) -> bool:
        if metrics["eos_ratio"] > self.best_eval["eos_ratio"]:
            return True
        if metrics["eos_ratio"] < self.best_eval["eos_ratio"]:
            return False
        if metrics["eval_S_mean"] > self.best_eval["eval_S_mean"]:
            return True
        if metrics["eval_S_mean"] < self.best_eval["eval_S_mean"]:
            return False
        return metrics["degeneration_rate"] < self.best_eval["degeneration_rate"]

    def _checkpoint_is_promotable(self, metrics: dict) -> bool:
        return (
            metrics["eos_ratio"] > PROMOTION_EOS_THRESHOLD
            and metrics["eval_S_mean"] >= PROMOTION_EVAL_S_THRESHOLD
            and metrics["leakage_rate"] <= PROMOTION_LEAKAGE_THRESHOLD
            and metrics["degeneration_rate"] < PROMOTION_DEGEN_THRESHOLD
            and metrics["benchmark_score"] >= self.promotion_benchmark_floor
        )

    def _save_adapter_checkpoint(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            self.adapter.state_dict(),
            os.path.join(ckpt_dir, "adapter_weights.pt"),
        )
        if self.stage_cfg.use_qlora and hasattr(self.llm, "save_pretrained"):
            qlora_dir = os.path.join(ckpt_dir, "qlora")
            self.llm.save_pretrained(qlora_dir)

    def _save_emergency_collapse_checkpoint(self, step: int, rolling_s: float):
        ckpt_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-emergency-collapse-{step}",
        )
        self._save_adapter_checkpoint(ckpt_dir)
        self._write_json(
            os.path.join(ckpt_dir, "collapse_metrics.json"),
            {
                "stage": self.stage_cfg.name,
                "step": step,
                "rolling_train_S_50": rolling_s,
                "collapse_threshold": ENTROPY_COLLAPSE_THRESHOLD,
            },
        )
        logger.error(
            f"Entropy collapse detected at step {step}: rolling_train_S_50={rolling_s:.4f}. "
            f"Emergency checkpoint saved to {ckpt_dir}"
        )

    def _should_abort_run_after_checkpoint(self, metrics: dict) -> bool:
        if metrics["leakage_rate"] > 0.0:
            logger.warning(
                f"Aborting run after checkpoint {metrics['step']}: leakage_rate={metrics['leakage_rate']:.4f}"
            )
            return True
        if metrics["degeneration_rate"] >= PROMOTION_DEGEN_THRESHOLD:
            logger.warning(
                f"Aborting run after checkpoint {metrics['step']}: degeneration_rate={metrics['degeneration_rate']:.4f}"
            )
            return True
        if self.stage_cfg.name != "alignment-eos":
            return False
        step = metrics["step"]
        if step == 50 and metrics["eval_S_mean"] < PROMOTION_EVAL_S_THRESHOLD:
            logger.warning(
                f"Stopping after checkpoint 50: eval_S_mean={metrics['eval_S_mean']:.4f} < {PROMOTION_EVAL_S_THRESHOLD:.2f}"
            )
            return True
        if step == 100 and (
            metrics["eos_ratio"] < 0.50 or metrics["eval_S_mean"] < PROMOTION_EVAL_S_THRESHOLD
        ):
            logger.warning(
                f"Stopping after checkpoint 100: eos_ratio={metrics['eos_ratio']:.4f}, "
                f"eval_S_mean={metrics['eval_S_mean']:.4f}"
            )
            return True
        if step == 150:
            checkpoint_100 = self.checkpoint_eval_history.get(100)
            if checkpoint_100 is not None and metrics["eos_ratio"] < checkpoint_100["eos_ratio"] - 0.05:
                logger.warning(
                    f"Stopping after checkpoint 150: eos_ratio regressed from "
                    f"{checkpoint_100['eos_ratio']:.4f} to {metrics['eos_ratio']:.4f}"
                )
                return True
        return False

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
        # Navigate PEFT wrapper / Qwen2 hierarchy to find transformer layers
        model = self.llm
        layers = None
        # Try all known paths: vanilla Qwen2, PeftModel, double-wrapped
        for path_fn in [
            lambda m: m.model.layers,                       # Qwen2ForCausalLM
            lambda m: m.model.model.layers,                 # PeftModel(Qwen2ForCausalLM)
            lambda m: m.base_model.model.model.layers,      # double-wrapped
            lambda m: m.layers,                              # already at Qwen2Model
        ]:
            try:
                candidate = path_fn(model)
                if candidate is not None and len(candidate) > 0:
                    layers = candidate
                    break
            except (AttributeError, IndexError):
                continue
        if layers is None:
            raise RuntimeError(f"Cannot find transformer layers on {type(model).__name__}")
        target_layer = layers[target_layer_idx]
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

            # Store base and adapted hidden states for EOS loss computation
            self._hook_h_base = h.detach()
            self._hook_h_adapted = adapted_out  # keep grad path

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

        # If a specific checkpoint name is given via --resume-from, use it
        resume_from = getattr(self.args, "resume_from", None)
        if resume_from:
            ckpt_path = os.path.join(ckpt_dir, resume_from, "adapter_weights.pt")
            if os.path.exists(ckpt_path):
                logger.info(f"Resuming from explicit checkpoint: {ckpt_path}")
                state = torch.load(ckpt_path, map_location=DEVICE)
                self.adapter.load_state_dict(state, strict=False)
                # For cross-stage resume, start_step resets to 0
                self.start_step = 0
                return
            else:
                logger.warning(f"Checkpoint {ckpt_path} not found, falling back to latest")

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
        self._initialize_promotion_state()
        self._apply_runtime_overrides()

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
            max_memory={0: "5GiB", "cpu": "30GiB"},
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

    def _generate_eval_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_ids: List[int],
        max_new_tokens: int = 100,
    ) -> Tuple[torch.Tensor, str, Optional[float]]:
        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                use_cache=True,
            )
        input_len = input_ids.shape[-1]
        generated = outputs[0, input_len:]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        eval_s = self._compute_entropy_from_geo_state()
        return outputs, decoded, eval_s

    def _run_benchmark_prompt_eval(self, eos_ids: List[int]) -> float:
        scores = []
        for prompt_info in BENCHMARK_PROMPTS:
            prompt_text = self._apply_eval_chat_template(prompt_info["prompt"])
            enc = self.tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            try:
                _outputs, decoded, _eval_s = self._generate_eval_output(
                    enc.input_ids, enc.attention_mask, eos_ids, max_new_tokens=160
                )
                score = self._score_benchmark_response(decoded, prompt_info)
            except Exception as e:
                logger.warning(f"Benchmark prompt evaluation failed: {e}")
                score = 0.0
            scores.append(score * prompt_info.get("weight", 1.0))
        if not scores:
            return 0.0
        total_weight = sum(p.get("weight", 1.0) for p in BENCHMARK_PROMPTS)
        return float(sum(scores) / max(1e-8, total_weight))

    # --- Evaluation loop --------------------------------------------------
    def evaluate(self, step):
        logger.info(f"Running evaluation at step {step}...")
        self.llm.eval()
        self.adapter.eval()
        
        has_gsp = False
        try:
            from igbundle.steering.gsp import create_gsp_for_qwen7b
            gsp, _bridge = create_gsp_for_qwen7b(telemetry_dict={
                "curvature": -1.0, "entropy": 0.0,
                "active_fiber": "Bundle-6", "constraint_score": 1.0,
            })
            gsp.attach(self.llm, range(8, 21))
            has_gsp = True
            logger.info("GSP attached for evaluation (layers 8-20).")
        except Exception as e:
            logger.warning(f"Could not attach GSP for evaluation: {e}")

        natural_eos_count = 0
        leakage_count = 0
        degeneration_count = 0
        eval_s_values = []
        sample_count = 0
        total_eval_samples = EVAL_SAMPLE_COUNT

        eval_loader = iter(DataLoader(self.train_ds, batch_size=1, collate_fn=multimodal_collate))
        eos_ids = self._stop_token_ids()

        for _ in range(total_eval_samples):
            try:
                batch = next(eval_loader)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            sample_count += 1

            outputs, decoded, eval_s = self._generate_eval_output(
                input_ids, attention_mask, eos_ids, max_new_tokens=100
            )
            last_token = outputs[0, -1].item()
            if last_token in eos_ids:
                natural_eos_count += 1
            if eval_s is not None:
                eval_s_values.append(eval_s)
            if self._detect_leakage(decoded):
                leakage_count += 1
            if self._detect_degeneration(decoded):
                degeneration_count += 1

        benchmark_score = self._run_benchmark_prompt_eval(eos_ids)

        if has_gsp:
            gsp.detach()

        # Return back to train mode
        if self.stage_cfg.use_qlora:
            self.llm.train()
        self.adapter.train()

        sample_count = max(1, sample_count)
        eos_ratio = natural_eos_count / sample_count
        eval_s_mean = float(sum(eval_s_values) / len(eval_s_values)) if eval_s_values else 0.0
        leakage_rate = leakage_count / sample_count
        degeneration_rate = degeneration_count / sample_count
        metrics = {
            "step": step,
            "eos_ratio": eos_ratio,
            "eval_S_mean": eval_s_mean,
            "entropy_mean": eval_s_mean,
            "leakage_rate": leakage_rate,
            "degeneration_rate": degeneration_rate,
            "benchmark_score": benchmark_score,
        }
        metrics["promotion_status"] = self._checkpoint_is_promotable(metrics)

        logger.info(
            f"Evaluation step {step}: eos_ratio={eos_ratio:.2f} ({natural_eos_count}/{sample_count}) "
            f"S_mean={eval_s_mean:.2f} leak={leakage_rate:.2%} "
            f"deg={degeneration_rate:.2%} benchmark={benchmark_score:.2f} "
            f"promote={metrics['promotion_status']}"
        )

        self._report_telemetry(
            step,
            {
                "eos_validation_ratio": eos_ratio,
                "eval_S_mean": eval_s_mean,
                "leakage_rate": leakage_rate,
                "degeneration_rate": degeneration_rate,
                "benchmark_score": benchmark_score,
                "promotion_status": float(metrics["promotion_status"]),
            },
        )
        return metrics

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
            geo_weighted_terms = {}
            geo_effective_terms = {}

            if self._current_geo_state and hasattr(self.adapter, "compute_geometric_losses"):
                geo_losses = self.adapter.compute_geometric_losses(self._current_geo_state)

                # Entropy ramp for resumed checkpoints
                resume_step = self.start_step
                entropy_ramp = (
                    min(1.0, max(0.0, (step - resume_step) / 200.0))
                    if resume_step > 0
                    else 1.0
                )

                ent_scale = sc.entropy_loss_scale

                for k, v in geo_losses.items():
                    term = v * entropy_ramp * ent_scale if k in ("fiber_diversity", "fiber_entropy") else v
                    geo_weighted_terms[k] = term.item() if hasattr(term, "item") else float(term)
                    if k in ("fiber_diversity", "fiber_entropy"):
                        loss_geo = loss_geo + term
                    else:
                        loss_geo = loss_geo + term

            # Homotopy schedule
            lambda_t = sc.geo_lambda_max * min(1.0, step / max(1, sc.geo_ramp_steps))
            for k, weighted_value in geo_weighted_terms.items():
                geo_effective_terms[k] = weighted_value * lambda_t
            total_loss = loss_llm + lambda_t * loss_geo

            # --- EOS Preservation Auxiliary Losses ---
            # Keep stop behavior only where the dataset explicitly supervises EOS.
            loss_eos_total = torch.tensor(0.0, device=DEVICE)
            eos_detail = {}
            h_base = getattr(self, "_hook_h_base", None)
            h_adapted = getattr(self, "_hook_h_adapted", None)

            if h_base is not None and h_adapted is not None:
                eos_ramp = min(1.0, max(0.0, (step - self.start_step) / 200.0))

                try:
                    lm_head = self.llm.lm_head if hasattr(self.llm, "lm_head") else None
                    if lm_head is None and hasattr(self.llm, "base_model"):
                        lm_head = self.llm.base_model.model.lm_head

                    if lm_head is not None:
                        lm_dtype = next(lm_head.parameters()).dtype
                        h_b_shift = h_base[:, :-1, :].to(lm_dtype)
                        h_a_shift = h_adapted[:, :-1, :].to(lm_dtype)
                        shift_labels = labels[:, 1:]

                        delta = (h_a_shift - h_b_shift).float()
                        norm_ratio = (delta.norm() ** 2) / (h_b_shift.float().norm() ** 2).clamp(min=1e-8)
                        loss_eos_norm = sc.eos_norm_weight * norm_ratio * eos_ramp
                        loss_eos_total = loss_eos_total + loss_eos_norm
                        eos_detail["eos_norm"] = norm_ratio.item()

                        eos_ids = self._stop_token_ids()
                        eos_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                        for eos_id in eos_ids:
                            eos_mask = eos_mask | (shift_labels == eos_id)
                        eos_mask = eos_mask & (shift_labels != -100)
                        eos_supervised_count = int(eos_mask.sum().item())
                        eos_detail["eos_supervised_count"] = float(eos_supervised_count)

                        if eos_supervised_count > 0:
                            h_b_sel = h_b_shift[eos_mask]
                            h_a_sel = h_a_shift[eos_mask]
                            target_labels = shift_labels[eos_mask]

                            with torch.no_grad():
                                base_logits = lm_head(h_b_sel).float()
                            adapted_logits = lm_head(h_a_sel).float()

                            eos_ce = F.cross_entropy(adapted_logits, target_labels)
                            loss_eos_total = loss_eos_total + sc.eos_ce_weight * eos_ce * eos_ramp
                            eos_detail["eos_ce"] = eos_ce.item()

                            base_target_logits = base_logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
                            adapted_target_logits = adapted_logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
                            eos_margin = F.relu((base_target_logits - 0.5) - adapted_target_logits).mean()
                            loss_eos_total = loss_eos_total + sc.eos_margin_weight * eos_margin * eos_ramp
                            eos_detail["eos_margin"] = eos_margin.item()

                            valid_stop_ids = [
                                eos_id
                                for eos_id in eos_ids
                                if eos_id is not None and 0 <= eos_id < adapted_logits.shape[-1]
                            ]
                            if len(valid_stop_ids) >= 2:
                                base_stop_logits = base_logits[:, valid_stop_ids]
                                adapted_stop_logits = adapted_logits[:, valid_stop_ids]
                                base_stop_probs = F.softmax(base_stop_logits, dim=-1)
                                adapted_stop_log_probs = F.log_softmax(adapted_stop_logits, dim=-1)
                                eos_stop_kl = F.kl_div(
                                    adapted_stop_log_probs,
                                    base_stop_probs,
                                    reduction="batchmean",
                                )
                                loss_eos_total = loss_eos_total + sc.eos_stop_kl_weight * eos_stop_kl * eos_ramp
                                eos_detail["eos_stop_kl"] = eos_stop_kl.item()

                            with torch.no_grad():
                                base_target_probs = F.softmax(base_logits, dim=-1).gather(
                                    1, target_labels.unsqueeze(1)
                                ).squeeze(1)
                                adapted_target_probs = F.softmax(adapted_logits, dim=-1).gather(
                                    1, target_labels.unsqueeze(1)
                                ).squeeze(1)
                            eos_detail["P_eos_base_supervised"] = base_target_probs.mean().item()
                            eos_detail["P_eos_adapted_supervised"] = adapted_target_probs.mean().item()

                except Exception as e:
                    if step < 5 or step % 100 == 0:
                        logger.warning(f"EOS loss computation failed: {e}")

                # Clean up hook state
                self._hook_h_base = None
                self._hook_h_adapted = None

            # Always clean up hook state (even when EOS loss skipped)
            if getattr(self, "_hook_h_base", None) is not None:
                self._hook_h_base = None
            if getattr(self, "_hook_h_adapted", None) is not None:
                self._hook_h_adapted = None

            total_loss = total_loss + loss_eos_total

            S_actual = self._compute_entropy_from_geo_state()
            rolling_s = None
            if S_actual is not None:
                self.train_entropy_history.append(S_actual)
                rolling_s = float(sum(self.train_entropy_history) / len(self.train_entropy_history))
                if (
                    len(self.train_entropy_history) == TRAINING_S_WINDOW
                    and rolling_s < ENTROPY_COLLAPSE_THRESHOLD
                ):
                    self._report_telemetry(
                        step,
                        {
                            "S": S_actual,
                            "rolling_train_S_50": rolling_s,
                            "collapse_guard_triggered": 1.0,
                        },
                    )
                    self._save_emergency_collapse_checkpoint(step, rolling_s)
                    raise RuntimeError(
                        f"Entropy collapse guard tripped: rolling_train_S_50={rolling_s:.4f} "
                        f"< {ENTROPY_COLLAPSE_THRESHOLD:.2f}"
                    )

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

                metrics = {
                    "loss_llm": loss_llm.item(),
                    "loss_geo": loss_geo.item(),
                    "loss_geo_effective": (lambda_t * loss_geo).item(),
                    "lambda_t": lambda_t,
                    "nan_count": nan_count,
                }
                if S_actual is not None:
                    metrics["S"] = S_actual
                if rolling_s is not None:
                    metrics["rolling_train_S_50"] = rolling_s
                if geo_losses:
                    if "curvature" in geo_losses:
                        v = geo_losses["curvature"]
                        metrics["K"] = v.item() if hasattr(v, "item") else float(v)
                    if "fiber_diversity" in geo_losses:
                        v = geo_losses["fiber_diversity"]
                        metrics["fiber_diversity"] = v.item() if hasattr(v, "item") else float(v)
                    for k, v in geo_losses.items():
                        metrics[f"geo_raw_{k}"] = v.item() if hasattr(v, "item") else float(v)
                    for k, v in geo_weighted_terms.items():
                        metrics[f"geo_weighted_{k}"] = v
                    for k, v in geo_effective_terms.items():
                        metrics[f"geo_effective_{k}"] = v
                if eos_detail:
                    metrics["loss_eos"] = loss_eos_total.item()
                    metrics.update({f"eos_{k}": v for k, v in eos_detail.items()})
                self._report_telemetry(step, metrics)

                # --- Logging ---
                if (step + 1) % (sc.grad_accum * 2) == 0 or step < 10:
                    geo_detail = ""
                    if geo_losses:
                        parts = [f"{k}={v.item() if hasattr(v, 'item') else v:.3f}"
                                 for k, v in sorted(geo_losses.items())]
                        geo_detail = f" [{', '.join(parts)}]"
                    if geo_effective_terms:
                        top_geo = sorted(
                            geo_effective_terms.items(),
                            key=lambda item: abs(item[1]),
                            reverse=True,
                        )[:4]
                        geo_top = ", ".join(f"{k}={v:.3f}" for k, v in top_geo)
                        geo_detail += f" GeoEff[{geo_top}]"

                    if S_actual is not None:
                        geo_detail += f" S={S_actual:.4f}"
                    if rolling_s is not None:
                        geo_detail += f" S50={rolling_s:.4f}"

                    nan_info = f" NaN={nan_count}" if nan_count > 0 else ""
                    eos_info = ""
                    if eos_detail:
                        eos_parts = [f"{k}={v:.4f}" for k, v in sorted(eos_detail.items())]
                        eos_info = f" EOS[{', '.join(eos_parts)}]"
                    logger.info(
                        f"[{sc.name}] step={step} loss={loss_llm.item():.4f} "
                        f"geo={loss_geo.item():.3f} eos={loss_eos_total.item():.4f} "
                        f"lam={lambda_t:.4f}{geo_detail}{eos_info}{nan_info}"
                    )

            # --- Checkpoint ---
            step += 1
            if step % sc.checkpoint_every == 0 or step == sc.max_steps:
                eval_metrics = self.evaluate(step)

                ckpt_dir = os.path.join(
                    self.args.output_dir, f"checkpoint-{sc.name}-{step}"
                )
                self._save_adapter_checkpoint(ckpt_dir)

                checkpoint_record = {
                    "stage": sc.name,
                    "checkpoint_dir": ckpt_dir,
                    **eval_metrics,
                }
                self.checkpoint_eval_history[step] = checkpoint_record

                self._write_json(
                    os.path.join(ckpt_dir, "metrics.json"),
                    {
                        "stage": sc.name,
                        "step": step,
                        "eos_validation_ratio": eval_metrics["eos_ratio"],
                        "eval_S_mean": eval_metrics["eval_S_mean"],
                    },
                )
                self._write_json(
                    os.path.join(ckpt_dir, "eval_metrics.json"),
                    checkpoint_record,
                )

                if self._is_better_best_checkpoint(checkpoint_record):
                    self.best_eval = {
                        "eos_ratio": checkpoint_record["eos_ratio"],
                        "eval_S_mean": checkpoint_record["eval_S_mean"],
                        "degeneration_rate": checkpoint_record["degeneration_rate"],
                        "step": step,
                        "checkpoint_dir": ckpt_dir,
                        "leakage_rate": checkpoint_record["leakage_rate"],
                        "benchmark_score": checkpoint_record["benchmark_score"],
                        "promotion_status": checkpoint_record["promotion_status"],
                    }
                    best_path = os.path.join(self.args.output_dir, "best_eos_checkpoint.json")
                    self._write_json(best_path, self.best_eval)
                    logger.info(
                        f"New best EOS checkpoint: step={step} ratio={checkpoint_record['eos_ratio']:.2f} "
                        f"S={checkpoint_record['eval_S_mean']:.2f} path={ckpt_dir}"
                    )

                if checkpoint_record["promotion_status"]:
                    self.promoted_checkpoint = checkpoint_record
                    self._write_json(
                        os.path.join(self.args.output_dir, "promoted_checkpoint.json"),
                        checkpoint_record,
                    )
                    logger.info(
                        f"New promoted checkpoint: step={step} eos_ratio={checkpoint_record['eos_ratio']:.2f} "
                        f"S={checkpoint_record['eval_S_mean']:.2f}"
                    )

                logger.info(f"Checkpoint saved: {ckpt_dir}")

                if self._should_abort_run_after_checkpoint(checkpoint_record):
                    logger.warning(f"Stopping run after checkpoint {step} due to recovery guardrails.")
                    break

        logger.info(f"Stage '{sc.name}' complete at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="OdysseyV3: Foundational Multimodal Training")
    parser.add_argument(
        "--stage",
        choices=["alignment", "alignment-eos", "instruction", "domain"],
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
        "--resume-from",
        default=None,
        help="Specific checkpoint folder name within checkpoint-dir (e.g. checkpoint-alignment-1200)",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override stage max_steps for short iterative runs",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Override checkpoint/evaluation cadence",
    )
    parser.add_argument(
        "--entropy-loss-scale",
        type=float,
        default=None,
        help="Override entropy loss multiplier for the selected stage",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=None,
        help="Override adapter base parameter learning rate",
    )
    parser.add_argument(
        "--fiber-lr",
        type=float,
        default=None,
        help="Override adapter fiber parameter learning rate",
    )
    parser.add_argument(
        "--geo-lambda-max",
        type=float,
        default=None,
        help="Override max homotopy weight for geometric loss",
    )
    parser.add_argument(
        "--geo-ramp-steps",
        type=int,
        default=None,
        help="Override geometric loss ramp length",
    )
    parser.add_argument(
        "--eos-ce-weight",
        type=float,
        default=None,
        help="Override supervised EOS cross-entropy weight",
    )
    parser.add_argument(
        "--eos-margin-weight",
        type=float,
        default=None,
        help="Override supervised EOS margin loss weight",
    )
    parser.add_argument(
        "--eos-stop-kl-weight",
        type=float,
        default=None,
        help="Override supervised EOS stop-distribution KL weight",
    )
    parser.add_argument(
        "--eos-norm-weight",
        type=float,
        default=None,
        help="Override hidden-state EOS norm regularizer weight",
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
