#!/usr/bin/env python3
"""
IGBundle Geometric Training Script v2.0
Enhanced training script with geometric IGBundle adapter integration.

This script provides comprehensive geometric training with:
- GeometricIGBundleAdapter with true Riemannian geometry
- RiemannianOptimizer with natural gradients
- Curvature-aware learning rate scheduling
- Comprehensive geometric metrics collection
- Memory-efficient 8GB VRAM operation
- Backward compatibility with standard training

Author: LLMOS SystemAgent
License: MIT
"""

# Same compatibility fixes as train.py
import sys
import os as _os
_src_path = _os.path.join(_os.path.dirname(__file__), "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Disable problematic imports
_os.environ['DISABLE_TORCHAO'] = '1'
sys.modules['torchao'] = None
_os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from igbundle.utils import triton_fix

import os
import argparse
import yaml
import torch
import time
import json
import gc
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig, TrainerCallback, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

# Import geometric modules
from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter, GeometricState
from igbundle.training.geometric_training import GeometricTrainer, GeometricTrainingConfig, RiemannianOptimizer
from igbundle.modules.adapter import IGBundleAdapter  # Fallback standard adapter
from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector
from igbundle.modules.losses import SheafLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingMode:
    """Training mode configuration."""
    GEOMETRIC = "geometric"
    STANDARD = "standard"
    AUTO = "auto"  # Choose based on config

class GeometricMetricsCallback(TrainerCallback):
    """Callback to collect and save comprehensive geometric training metrics."""

    def __init__(self, output_dir: str, config: Dict[str, Any]):
        self.output_dir = output_dir
        self.config = config
        self.geometric_metrics = []
        self.training_history = []
        self.step_times = []
        self.memory_usage = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

        # Track memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            self.memory_usage.append({
                "step": state.global_step,
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved
            })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step_time = time.time() - getattr(self, 'step_start_time', time.time())
            self.step_times.append(step_time)

            # Extract all losses from logs
            geometric_info = {
                "step": state.global_step,
                "epoch": state.epoch,
                "learning_rate": logs.get("learning_rate", 0.0),
                "loss": logs.get("loss", 0.0),
                "train_loss": logs.get("train_loss", 0.0),

                # Geometric losses (if available)
                "curvature_loss": logs.get("curvature_loss", 0.0),
                "sheaf_loss": logs.get("sheaf_loss", 0.0),
                "bundle_loss": logs.get("bundle_loss", 0.0),
                "lambda_loss": logs.get("lambda_loss", 0.0),
                "geometric_total_loss": (
                    logs.get("curvature_loss", 0.0) +
                    logs.get("sheaf_loss", 0.0) +
                    logs.get("bundle_loss", 0.0) +
                    logs.get("lambda_loss", 0.0)
                ),

                # Training efficiency metrics
                "step_time": step_time,
                "grad_norm": logs.get("grad_norm", 0.0),

                # Model metrics (if available)
                "model_perplexity": np.exp(logs.get("loss", 0.0)) if logs.get("loss", 0.0) > 0 else float('inf')
            }

            self.geometric_metrics.append(geometric_info)
            self.training_history.append(logs.copy())

            # Log progress
            if state.global_step % 10 == 0:
                logger.info(f"Step {state.global_step}: Loss={logs.get('loss', 0.0):.4f}, "
                          f"Geometric={geometric_info['geometric_total_loss']:.4f}, "
                          f"LR={logs.get('learning_rate', 0.0):.2e}")

    def on_train_end(self, args, state, control, **kwargs):
        # Save comprehensive metrics
        self._save_training_metrics()

        # Compute and save summary
        summary = self._compute_training_summary()
        self._save_training_summary(summary)

        logger.info(f"Training completed. Metrics saved to {self.output_dir}")

    def _save_training_metrics(self):
        """Save detailed training metrics."""
        os.makedirs(self.output_dir, exist_ok=True)

        metrics_data = {
            "config": self.config,
            "geometric_metrics": self.geometric_metrics,
            "training_history": self.training_history,
            "memory_usage": self.memory_usage,
            "step_times": self.step_times
        }

        metrics_path = os.path.join(self.output_dir, "geometric_training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

    def _compute_training_summary(self) -> Dict[str, float]:
        """Compute comprehensive training summary metrics."""
        if not self.geometric_metrics:
            return {"status": "no_data"}

        # Extract key metrics
        losses = [m["loss"] for m in self.geometric_metrics if m["loss"] > 0]
        curvature_losses = [m["curvature_loss"] for m in self.geometric_metrics]
        step_times = [t for t in self.step_times if t > 0]

        summary = {
            # Basic training metrics
            "total_steps": len(self.geometric_metrics),
            "final_loss": losses[-1] if losses else float('inf'),
            "initial_loss": losses[0] if losses else float('inf'),
            "best_loss": min(losses) if losses else float('inf'),
            "final_perplexity": np.exp(losses[-1]) if losses else float('inf'),

            # Convergence analysis
            "loss_reduction": (losses[0] - losses[-1]) if len(losses) > 1 else 0.0,
            "convergence_rate": self._compute_convergence_rate(losses),

            # Training stability
            "loss_stability": self._compute_stability(losses),
            "training_efficiency": 1.0 / np.mean(step_times) if step_times else 0.0,
            "average_step_time": np.mean(step_times) if step_times else 0.0,

            # Geometric quality metrics
            "curvature_alignment": self._compute_curvature_alignment(curvature_losses),
            "geometric_consistency": self._compute_geometric_consistency(),
            "final_curvature_loss": curvature_losses[-1] if curvature_losses else 0.0,

            # Resource utilization
            "peak_memory_gb": max([m["allocated_gb"] for m in self.memory_usage]) if self.memory_usage else 0.0,
            "average_memory_gb": np.mean([m["allocated_gb"] for m in self.memory_usage]) if self.memory_usage else 0.0
        }

        return summary

    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """Compute convergence rate (higher is better)."""
        if len(losses) < 10:
            return 0.0

        # Fit exponential decay to loss curve
        x = np.arange(len(losses))
        try:
            log_losses = np.log(np.maximum(losses, 1e-10))
            slope = np.polyfit(x, log_losses, 1)[0]
            return max(0.0, -slope)  # Negative slope = positive convergence
        except:
            return 0.0

    def _compute_stability(self, losses: List[float]) -> float:
        """Compute training stability (higher is better)."""
        if len(losses) < 10:
            return 0.0

        # Use coefficient of variation in final 25% of training
        final_quarter = losses[len(losses)//4*3:]
        if len(final_quarter) < 5:
            return 0.0

        mean_loss = np.mean(final_quarter)
        std_loss = np.std(final_quarter)

        if mean_loss > 0:
            cv = std_loss / mean_loss
            return 1.0 / (1.0 + cv)  # Higher stability = lower coefficient of variation
        return 0.0

    def _compute_curvature_alignment(self, curvature_losses: List[float]) -> float:
        """Compute how well model achieves target curvature."""
        if not curvature_losses:
            return 0.0

        final_curvature_loss = curvature_losses[-1]
        # Convert loss to alignment score (lower loss = better alignment)
        return max(0.0, 1.0 - final_curvature_loss)

    def _compute_geometric_consistency(self) -> float:
        """Compute overall geometric consistency."""
        if not self.geometric_metrics:
            return 0.0

        # Average of normalized geometric loss components
        final_metrics = self.geometric_metrics[-1]
        sheaf_quality = max(0.0, 1.0 - final_metrics["sheaf_loss"])
        bundle_quality = max(0.0, 1.0 - final_metrics["bundle_loss"])
        lambda_quality = max(0.0, 1.0 - final_metrics["lambda_loss"])

        return (sheaf_quality + bundle_quality + lambda_quality) / 3.0

    def _save_training_summary(self, summary: Dict[str, float]):
        """Save training summary for optimization."""
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

class SlowStepCallback(TrainerCallback):
    """Thermal management callback with configurable delay."""

    def __init__(self, delay_seconds: float = 5.0):
        self.delay_seconds = delay_seconds

    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(self.delay_seconds)

class SaveAdapterCallback(TrainerCallback):
    """Save only adapter weights at checkpoints."""

    def __init__(self, adapter_class: str = "geometric"):
        self.adapter_class = adapter_class

    def on_save(self, args, state, control, **kwargs):
        # Extract adapter weights and save separately
        model = kwargs.get("model")
        if model is None:
            return

        # Get adapter state dict
        adapter_state = {}
        for name, param in model.named_parameters():
            if "adapter" in name.lower() or "igbundle" in name.lower():
                adapter_state[name] = param.data.clone()

        # Save adapter weights
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        adapter_path = os.path.join(checkpoint_dir, f"{self.adapter_class}_adapter_weights.pt")
        torch.save(adapter_state, adapter_path)

        logger.info(f"Saved {self.adapter_class} adapter weights: {adapter_path}")

class Config:
    """Configuration wrapper for nested dictionary access."""
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def create_geometric_config(yaml_config: dict) -> GeometricTrainingConfig:
    """Create GeometricTrainingConfig from YAML configuration."""
    geo_cfg = yaml_config.get("geometric_training", {})
    training_cfg = yaml_config.get("training", {})

    return GeometricTrainingConfig(
        # Standard training
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        batch_size=training_cfg.get("per_device_train_batch_size", 1),
        max_steps=training_cfg.get("max_steps", 1000),
        warmup_steps=training_cfg.get("warmup_steps", 100),

        # Geometric loss weights
        lambda_curvature=geo_cfg.get("lambda_curvature", 0.01),
        lambda_sheaf=geo_cfg.get("lambda_sheaf", 0.001),
        lambda_bundle=geo_cfg.get("lambda_bundle", 0.001),
        lambda_lambda=geo_cfg.get("lambda_lambda", 0.0001),

        # Riemannian optimization
        use_natural_gradients=geo_cfg.get("use_natural_gradients", True),
        fisher_update_freq=geo_cfg.get("fisher_update_freq", 10),
        fisher_momentum=geo_cfg.get("fisher_momentum", 0.95),

        # Curvature scheduling
        target_curvature_schedule=geo_cfg.get("target_curvature_schedule", "exponential"),
        initial_target_curvature=geo_cfg.get("initial_target_curvature", 0.0),
        final_target_curvature=geo_cfg.get("final_target_curvature", -1.0),

        # Bundle structure
        preserve_bundle_topology=geo_cfg.get("preserve_bundle_topology", True),
        topology_check_freq=geo_cfg.get("topology_check_freq", 50),
    )

def load_model_and_tokenizer(config: Config):
    """Load model and tokenizer with proper quantization and device mapping."""
    logger.info(f"Loading model: {config.base_model_id}")

    # Check hardware and adjust if needed
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Switching to smaller model for CPU.")
        config.base_model_id = "Qwen/Qwen2.5-0.5B"

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Setup quantization
    use_4bit = getattr(config.training, 'bf16', True) and torch.cuda.is_available()

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    logger.info(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'CPU'}")
    return model, tokenizer

def apply_lora(model, config: Config):
    """Apply LoRA configuration to model."""
    logger.info("Applying LoRA configuration...")

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    logger.info(f"LoRA applied: r={config.lora.r}, alpha={config.lora.lora_alpha}")
    return model

def apply_adapter(model, config: Config, mode: str = TrainingMode.GEOMETRIC):
    """Apply IGBundle adapter (geometric or standard)."""
    logger.info(f"Applying {mode} IGBundle adapter...")

    # Update hidden size from model config
    actual_model = model.model if hasattr(model, "model") else model
    if hasattr(actual_model.config, "hidden_size"):
        config.ig_adapter.hidden_size = actual_model.config.hidden_size
        logger.info(f"Updated hidden_size: {config.ig_adapter.hidden_size}")

    # Choose adapter class
    if mode == TrainingMode.GEOMETRIC:
        adapter_class = GeometricIGBundleAdapter
        logger.info("Using GeometricIGBundleAdapter with Riemannian geometry")
    else:
        adapter_class = IGBundleAdapter
        logger.info("Using standard IGBundleAdapter")

    # Apply adapter
    model = wrap_hf_candidate(model, config.ig_adapter, adapter_class=adapter_class)

    # Ensure adapter parameters are trainable
    adapter_params = 0
    for name, param in model.named_parameters():
        if "adapter" in name or "igbundle" in name or "lora" in name:
            param.requires_grad = True
            adapter_params += param.numel()

    logger.info(f"Trainable adapter parameters: {adapter_params:,}")

    # Print trainable parameters summary
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

    return model

def load_dataset_for_training(tokenizer, config: Config, dataset_size: Optional[int] = None):
    """Load and prepare dataset for training."""
    logger.info("Loading dataset...")

    if dataset_size and dataset_size <= 1000:
        # Create small synthetic dataset for testing/optimization
        logger.info(f"Creating synthetic dataset with {dataset_size} examples")
        examples = []
        for i in range(dataset_size):
            text = f"This is example number {i+1}. " * 10  # Repeat for sufficient length
            examples.append({"text": text + tokenizer.eos_token})
        data = Dataset.from_list(examples)
    else:
        # Load full Alpaca dataset
        logger.info("Loading Alpaca dataset...")
        data = load_dataset("yahma/alpaca-cleaned", split="train")

        if dataset_size:
            data = data.select(range(min(dataset_size, len(data))))
            logger.info(f"Using {len(data)} examples from Alpaca dataset")

        # Format Alpaca examples
        def format_alpaca(example):
            if example.get('instruction') and example.get('output'):
                prompt = f"Below is an instruction. Write a response.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n"
                text = prompt + example['output'] + tokenizer.eos_token
            else:
                # Fallback for malformed examples
                text = f"Example: {example.get('instruction', 'No instruction')}" + tokenizer.eos_token
            return {"text": text}

        data = data.map(format_alpaca, remove_columns=data.column_names)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    tokenized_data = data.map(
        tokenize_function,
        batched=True,
        remove_columns=data.column_names,
        desc="Tokenizing dataset"
    )

    logger.info(f"Dataset prepared: {len(tokenized_data)} examples")
    return tokenized_data

def setup_geometric_trainer(model, config: Config, geometric_config: GeometricTrainingConfig,
                          output_dir: str) -> Tuple[GeometricTrainer, List[TrainerCallback]]:
    """Setup geometric trainer with enhanced metrics collection."""
    logger.info("Setting up geometric trainer...")

    # Create geometric trainer
    trainer = GeometricTrainer(model, geometric_config)

    # Setup callbacks
    callbacks = [
        GeometricMetricsCallback(output_dir, config.__dict__ if hasattr(config, '__dict__') else {}),
        SlowStepCallback(delay_seconds=5.0),  # Thermal management
        SaveAdapterCallback(adapter_class="geometric")
    ]

    logger.info("Geometric trainer configured with RiemannianOptimizer")
    return trainer, callbacks

def setup_standard_trainer(model, config: Config, tokenized_data, output_dir: str) -> Tuple[Trainer, List[TrainerCallback]]:
    """Setup standard HuggingFace trainer as fallback."""
    logger.info("Setting up standard trainer...")

    # Setup state collector for standard IGBundle
    collector = StateCollector()
    collector.attach(model)

    # Setup sheaf loss
    sheaf_loss = SheafLoss(
        num_patches=getattr(config.loss, 'num_patches', 8),
        latent_dim=config.ig_adapter.latent_dim,
        tau=getattr(config.loss, 'tau', 1.0)
    )

    if torch.cuda.is_available():
        sheaf_loss = sheaf_loss.to(torch.device("cuda"))

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=getattr(config.training, 'gradient_accumulation_steps', 8),
        learning_rate=float(config.training.learning_rate),
        max_steps=getattr(config.training, 'max_steps', 1000),
        warmup_steps=getattr(config.training, 'warmup_steps', 100),
        logging_steps=1,
        save_steps=getattr(config.training, 'save_steps', 250),
        bf16=getattr(config.training, 'bf16', True) and torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        max_grad_norm=getattr(config.training, 'max_grad_norm', 0.3),
        weight_decay=getattr(config.training, 'weight_decay', 0.01),
        remove_unused_columns=False,  # Keep all columns for debugging
    )

    # Import standard trainer
    from train import IGBundleTrainer

    trainer = IGBundleTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        state_collector=collector,
        sheaf_loss_fn=sheaf_loss,
        lambda_glue=float(getattr(config.loss, 'lambda_glue', 0.01)),
    )

    # Setup callbacks
    callbacks = [
        GeometricMetricsCallback(output_dir, config.__dict__ if hasattr(config, '__dict__') else {}),
        SlowStepCallback(delay_seconds=5.0),
        SaveAdapterCallback(adapter_class="standard")
    ]

    logger.info("Standard trainer configured")
    return trainer, callbacks

def run_geometric_training(trainer: GeometricTrainer, callbacks: List[TrainerCallback],
                         tokenized_data, config: Config, geometric_config: GeometricTrainingConfig):
    """Run geometric training with custom training loop."""
    logger.info("Starting geometric training loop...")

    os.makedirs(config.training.output_dir, exist_ok=True)

    # Initialize callbacks
    for callback in callbacks:
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin(None, None, None)

    # Simple training loop
    for step in range(geometric_config.max_steps):
        # Trigger step begin callbacks
        for callback in callbacks:
            if hasattr(callback, 'on_step_begin'):
                callback.on_step_begin(None, type('State', (), {'global_step': step}), None)

        try:
            # Sample batch
            batch_size = min(geometric_config.batch_size, len(tokenized_data))
            indices = torch.randint(0, len(tokenized_data), (batch_size,))
            batch_texts = [tokenized_data[i]["input_ids"] for i in indices]

            # Pad batch
            max_len = min(512, max(len(text) for text in batch_texts))
            batch = torch.zeros(batch_size, max_len, dtype=torch.long)

            for i, text in enumerate(batch_texts):
                length = min(len(text), max_len)
                batch[i, :length] = torch.tensor(text[:length])

            if torch.cuda.is_available():
                batch = batch.cuda()

            # Training step
            losses = trainer.train_step(batch)

            # Create logs for callbacks
            logs = {
                "loss": losses.get("total_loss", 0.0),
                "curvature_loss": losses.get("curvature_loss", 0.0),
                "sheaf_loss": losses.get("sheaf_loss", 0.0),
                "bundle_loss": losses.get("bundle_loss", 0.0),
                "lambda_loss": losses.get("lambda_loss", 0.0),
                "learning_rate": trainer.get_current_lr(),
            }

            # Trigger log callbacks
            state = type('State', (), {'global_step': step, 'epoch': step / len(tokenized_data)})
            for callback in callbacks:
                if hasattr(callback, 'on_log'):
                    callback.on_log(None, state, None, logs=logs)

            # Trigger step end callbacks
            for callback in callbacks:
                if hasattr(callback, 'on_step_end'):
                    callback.on_step_end(None, state, None)

            # Save checkpoints periodically
            if step > 0 and step % 250 == 0:
                checkpoint_dir = os.path.join(config.training.output_dir, f"checkpoint-{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Trigger save callbacks
                for callback in callbacks:
                    if hasattr(callback, 'on_save'):
                        callback.on_save(None, state, None, model=trainer.model)

                logger.info(f"Checkpoint saved: {checkpoint_dir}")

        except Exception as e:
            logger.error(f"Training step {step} failed: {e}")
            # Continue training on single step failure
            continue

    # Training complete
    logger.info("Geometric training loop completed")

    # Trigger training end callbacks
    final_state = type('State', (), {'global_step': geometric_config.max_steps})
    for callback in callbacks:
        if hasattr(callback, 'on_train_end'):
            callback.on_train_end(None, final_state, None)

def memory_cleanup():
    """Aggressive memory cleanup for trial isolation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Memory cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="IGBundle Geometric Training v2.0")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--mode", type=str, choices=[TrainingMode.GEOMETRIC, TrainingMode.STANDARD, TrainingMode.AUTO],
                       default=TrainingMode.AUTO, help="Training mode")
    parser.add_argument("--dataset_size", type=int, default=None, help="Limit dataset size for testing")
    parser.add_argument("--optuna_trial", type=int, default=None, help="Optuna trial number (for optimization)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"🧮 IGBundle Geometric Training v2.0")
    logger.info(f"   📄 Config: {args.config}")
    logger.info(f"   🔧 Mode: {args.mode}")
    if args.optuna_trial is not None:
        logger.info(f"   🔬 Optuna Trial: {args.optuna_trial}")

    try:
        # Load configuration
        with open(args.config, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        config = Config(cfg_dict)

        # Override output directory if specified
        if args.output_dir:
            config.training.output_dir = args.output_dir

        # Determine training mode
        training_mode = args.mode
        if training_mode == TrainingMode.AUTO:
            # Auto-detect based on config
            has_geometric_config = "geometric_training" in cfg_dict
            training_mode = TrainingMode.GEOMETRIC if has_geometric_config else TrainingMode.STANDARD

        logger.info(f"   🎯 Training Mode: {training_mode}")

        # Setup output directory
        os.makedirs(config.training.output_dir, exist_ok=True)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)

        # Apply LoRA
        model = apply_lora(model, config)

        # Apply adapter
        model = apply_adapter(model, config, mode=training_mode)

        # Load dataset
        tokenized_data = load_dataset_for_training(tokenizer, config, args.dataset_size)

        # Setup trainer based on mode
        if training_mode == TrainingMode.GEOMETRIC:
            # Geometric training with RiemannianOptimizer
            geometric_config = create_geometric_config(cfg_dict)
            trainer, callbacks = setup_geometric_trainer(model, config, geometric_config, config.training.output_dir)

            logger.info("🚀 Starting geometric training...")
            run_geometric_training(trainer, callbacks, tokenized_data, config, geometric_config)

        else:
            # Standard HuggingFace training
            trainer, callbacks = setup_standard_trainer(model, config, tokenized_data, config.training.output_dir)

            # Add callbacks to trainer
            for callback in callbacks:
                trainer.add_callback(callback)

            logger.info("🚀 Starting standard training...")
            trainer.train()

        logger.info("✅ Training completed successfully")

        # Final memory cleanup
        memory_cleanup()

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")

        # Emergency memory cleanup
        try:
            memory_cleanup()
        except:
            pass

        raise e

if __name__ == "__main__":
    main()