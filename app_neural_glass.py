import os
import sys

# Fix Windows charmap encoding crash — force UTF-8 on stdout/stderr
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_CONTENT.*")
import torch
import gradio as gr
import time
import threading
import queue
import random
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    SiglipVisionModel,
    SiglipImageProcessor,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
import numpy as np
import plotly.graph_objects as go

# --- OPTIMIZATION FLAGS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# --- PATHS & IMPORTS ---
sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.modules.regularization import LipschitzPenalty
from igbundle.geometry.hyperbolic import PoincareBall
from igbundle.fibers.constraint import ConstraintExtractor, ConstraintScorer
from imd_memory import IMDMemory
from model_state import ModelState

# --- GENERATION STOP EVENT ---
# Thread-safe mechanism to halt model.generate() when corruption is detected.
# The main thread sets the event; StoppingCriteria checks it each token step.
class _StopOnEvent(StoppingCriteria):
    """StoppingCriteria that halts generation when a threading.Event is set."""
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

class _StopOnNewTurn(StoppingCriteria):
    """Stop generation if model emits <|im_start|> or continues after <|im_end|>.

    The adapter sometimes suppresses <|im_end|> probability, so the model skips its EOS
    and continues directly into a new turn (e.g. "Please answer the following question...").

    Three detection layers:
    1. Check last N tokens for <|im_start|> (catches delayed detection)
    2. Track <|im_end|> — if generation continues 2+ tokens after it, force stop
    3. Both work at token level BEFORE text decoding (skip_special_tokens=True hides these)
    """
    def __init__(self, im_start_id: int, im_end_id: int, prompt_len: int, min_gen: int = 10):
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.prompt_len = prompt_len
        self.min_gen = min_gen
        self._saw_im_end_at = -1  # gen_len when <|im_end|> was first seen

    def __call__(self, input_ids, scores, **kwargs):
        gen_len = input_ids.shape[-1] - self.prompt_len
        if gen_len < self.min_gen:
            return False
        # Window scan: check last 4 generated tokens for <|im_start|>
        window = min(4, gen_len)
        recent = input_ids[0, -window:].tolist()
        if self.im_start_id in recent:
            return True
        # Track <|im_end|>: if model generated it but didn't stop (EOS suppressed),
        # allow 1 token grace (possible trailing whitespace) then force stop
        last_tok = input_ids[0, -1].item()
        if last_tok == self.im_end_id:
            self._saw_im_end_at = gen_len
        if self._saw_im_end_at > 0 and gen_len > self._saw_im_end_at + 1:
            return True
        return False

# --- CONSTANTS ---
BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
ADAPTER_PATH = "" # Autodetect
CHECKPOINT_DIR = "h:/LLM-MANIFOLD/igbundle-llm/igbundle_phase9_odyssey" # Phase 9 Odyssey checkpoints


# --- GLOBAL STATE ---
# We use a global state to allow the UI to poll telemetry while generating
TELEMETRY_STATE = {
    "curvature": -1.0,
    "entropy": 0.0,
    "active_fiber": "None",
    "thought_trace": [],
    "history_k": [], # History for plotting
    "history_s": [],
    "active_constraints": [],
    "constraint_score": 1.0,
    "lipschitz_ratio": 1.0,
    "last_geo_pos": None, # For computing d_M(t, t-1)
    "last_euc_pos": None,
    "manifold_trace": [], # List of (x,y) coordinates for plotting
    "gibbs_beta": 4.6,  # Effective inverse temperature
    "damping": 0.01     # Damping parameter for Gibbs calculation
}

# --- GIBBS TEMPERATURE HELPER ---
def compute_gibbs_temperature(damping: float) -> float:
    import math
    if damping > 0 and damping < 1:
        return -math.log(damping / (1 - damping))
    return float('inf')

MODELS = {
    "llm": None,
    "tokenizer": None,
    "vision_model": None,
    "processor": None,
    "adapter": None
}

# --- STATEFUL CONFIG ---
MODEL_STATE = ModelState()

# --- CONTINUOUS RUN MODE ---
# Prompts organized by Poincaré zone — the manifold position determines
# whether we push outward (explore), stay balanced, or pull inward (consolidate).
_CONT_PROMPTS_ANCHOR = [
    # r < 0.3 — stable region. Push outward for exploration.
    "What broader connections does this relate to? Explore an adjacent domain.",
    "Consider a counterexample or edge case. What breaks?",
    "What would someone who disagrees say? Steelman the opposing view.",
]
_CONT_PROMPTS_BALANCED = [
    # 0.3 ≤ r < 0.6 — productive zone. Maintain depth.
    "Continue your analysis. What follows from what you just said?",
    "Elaborate further. What are the deeper implications?",
    "Extend this reasoning. What new connections emerge?",
]
_CONT_PROMPTS_EXPLORATORY = [
    # 0.6 ≤ r < 0.85 — approaching boundary. Start pulling back.
    "Synthesize your key findings so far into a coherent framework.",
    "What is the most fundamental insight from this analysis?",
    "Ground this reasoning in a concrete, specific example.",
]
_CONT_PROMPTS_BOUNDARY = [
    # r ≥ 0.85 — boundary/unstable. Strong pull toward center.
    "Stop and summarize: what do you know for certain?",
    "Return to first principles. What is the simplest true statement here?",
    "Consolidate. State your conclusion in one precise sentence.",
]

def _select_continuation_prompt() -> str:
    """
    Geometry-aware continuation prompt selection.

    Uses the current Poincaré ball position to balance the L→R generation
    momentum against the R→L memory/consolidation pull:
      - Near origin: encourage exploration (centrifugal)
      - Mid-range: maintain productive depth
      - Near boundary: pull back toward coherence (centripetal)

    This creates a natural oscillation on the manifold rather than
    random walk or boundary-hugging.
    """
    trace = TELEMETRY_STATE.get("manifold_trace", [])
    if not trace:
        return random.choice(_CONT_PROMPTS_BALANCED)

    current = trace[-1]
    r = float(np.sqrt(current[0]**2 + current[1]**2))

    # Track momentum direction for thought trace
    if len(trace) >= 2:
        prev = trace[-2]
        dr = r - float(np.sqrt(prev[0]**2 + prev[1]**2))
        direction = "outward" if dr > 0.01 else "inward" if dr < -0.01 else "stable"
        TELEMETRY_STATE["thought_trace"].append(
            f"[Continuous] r={r:.2f} ({direction}) -> zone selection"
        )

    if r < 0.3:
        return random.choice(_CONT_PROMPTS_ANCHOR)
    elif r < 0.6:
        return random.choice(_CONT_PROMPTS_BALANCED)
    elif r < 0.85:
        return random.choice(_CONT_PROMPTS_EXPLORATORY)
    else:
        return random.choice(_CONT_PROMPTS_BOUNDARY)


CONTINUOUS_RUN = {
    "enabled": False,
    "turn_count": 0,
    "max_turns": 50,
    "consolidate_every": 5,
}

# --- MODEL LOADING ---
def find_latest_adapter():
    """Dynamic checkpoint resolution via ModelState."""
    state = MODEL_STATE.load()
    ckpt_path = state["checkpoint"]["path"]
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[ModelState] {MODEL_STATE.summary()}")
        return ckpt_path, state["checkpoint"]["adapter_type"]
    # Fallback: scan for latest
    path, step = ModelState.find_latest_checkpoint()
    if path:
        MODEL_STATE.update_checkpoint(path, step)
        return path, "geometric"
    return "DEBUG_MODE_NO_WEIGHTS", "none"

def load_models():
    if MODELS["llm"] is not None: return "Models Already Loaded"

    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    # Epic: CPU Swapping - Restrict GPU memory to force layer offloading
    max_memory = {0: "5GiB", "cpu": "30GiB"}
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", max_memory=max_memory, trust_remote_code=True
    )

    print("Loading SigLIP (CPU — moved to GPU on demand)...")
    processor = SiglipImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip2-so400m-patch14-384", torch_dtype=torch.float16
    )  # Stays on CPU — saves ~500MB VRAM for text-only queries

    print("Loading Geometric Adapter...")
    config = IGBundleConfig(
        hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
        use_dynamics=True, use_geodesic_attn=True, supported_modalities=["vision", "text"],
        enable_meta_cognition=True
    )
    adapter = create_geometric_adapter(config).to("cuda")

    weight_path, weight_type = find_latest_adapter()
    if weight_path != "DEBUG_MODE_NO_WEIGHTS":
        print(f"Found Weights: {weight_path} ({weight_type})")

        if weight_type == "lora":
            print(f"Loading LoRA Adapter from {weight_path}...")
            llm = PeftModel.from_pretrained(llm, weight_path)

        elif weight_type == "geometric":
            print(f"Loading Geometric Weights from {weight_path}...")
            adapter.load_state_dict(torch.load(weight_path), strict=False)

    MODELS.update({
        "llm": llm, "tokenizer": tokenizer,
        "vision_model": vision_model, "processor": processor, "adapter": adapter,
        "constraint_extractor": ConstraintExtractor(),
        "constraint_scorer": ConstraintScorer(),
        "memory": IMDMemory(telemetry_state=TELEMETRY_STATE)  # IMD: 3-tier geometric memory
    })

    # Inject Hook
    inject_adapter_hook(llm, adapter)

    # Restore telemetry from previous session
    saved_telem = MODEL_STATE.restore_telemetry()
    if saved_telem:
        for k, v in saved_telem.items():
            if k in TELEMETRY_STATE:
                TELEMETRY_STATE[k] = v
        print(f"[ModelState] Restored telemetry: {list(saved_telem.keys())}")

    # Report VRAM usage at startup
    try:
        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        vram_free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        vram_used = vram_total - vram_free
        print(f"[VRAM] Total: {vram_total:.1f}GB | Used: {vram_used:.1f}GB | Free: {vram_free:.1f}GB")
    except Exception:
        pass

    return f"System Online. Weights: {weight_path} | {MODEL_STATE.summary()}"

# --- ADAPTER HOOK (THE "GHOST" MECHANISM) ---
# This is a shared container for vision features that changes per request
VISION_CONTEXT = {"feats": None}

def inject_adapter_hook(llm, adapter):
    model_root = llm
    if hasattr(model_root, "model") and hasattr(model_root.model, "layers"):
         layers = model_root.model.layers
    elif hasattr(model_root, "model") and hasattr(model_root.model, "model") and hasattr(model_root.model.model, "layers"):
         layers = model_root.model.model.layers
    elif hasattr(model_root, "base_model") and hasattr(model_root.base_model, "model") and hasattr(model_root.base_model.model, "layers"):
         layers = model_root.base_model.model.layers
    else:
         print("ERROR: Could not locate transformer layers in model structure. Hook injection failed.")
         return

    target_layer = layers[12]
    llm._geo_target_layer = target_layer
    original_forward = target_layer.forward

    def adapter_hook(hidden_states, *args, **kwargs):
        # 1. Base Forward
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype

        # 2. Adapter Logic
        h_in = h.to("cuda").to(torch.float32)
        vis_feats = None
        if VISION_CONTEXT['feats'] is not None:
             vis_feats = VISION_CONTEXT['feats'].to("cuda").to(torch.float32)

        with torch.no_grad():
             adapted_out, geo_state = adapter(h_in, pixel_values=vis_feats)

        # 3. Telemetry Extraction
        try:
            curv = 0.0
            if hasattr(adapter, 'riemannian_geometry'):
                 pass

            probs = geo_state.fiber_sections.clamp(min=1e-8)
            entropy_tensor = -torch.sum(probs * torch.log(probs), dim=-1).mean()
            entropy = float(entropy_tensor.item()) if entropy_tensor.numel() == 1 else float(entropy_tensor.mean().item())

            if hasattr(geo_state, 'active_indices') and geo_state.active_indices is not None:
                 flat_indices = geo_state.active_indices.reshape(-1)
                 idx = flat_indices[0].item() if flat_indices.numel() > 0 else 0
                 active_fiber = f"Bundle-{int(idx)}"
            else:
                 active_fiber = "Standby"

            if hasattr(adapter, 'riemannian_geometry') and hasattr(geo_state, 'base_coordinates'):
                 try:
                     positions = geo_state.base_coordinates
                     real_curvature = adapter.riemannian_geometry.estimate_sectional_curvature_stochastic(
                         positions, num_samples=3
                     )
                     curv_val = real_curvature.mean().item()
                     TELEMETRY_STATE["curvature"] = round(float(curv_val), 2)
                 except Exception as e:
                     print(f"Curvature Calc Failed: {e}")
                     TELEMETRY_STATE["curvature"] = round(-1.0 + random.uniform(-0.1, 0.1), 2)
            else:
                 TELEMETRY_STATE["curvature"] = round(-1.0 + random.uniform(-0.05, 0.05), 2)
            TELEMETRY_STATE["entropy"] = round(float(entropy), 4)

            # OOD Teleportation Detection
            current_euc = h_in[:, -1, :]

            if hasattr(geo_state, 'base_coordinates'):
                 current_geo = geo_state.base_coordinates[:, -1, 0, :]

                 if TELEMETRY_STATE["last_geo_pos"] is not None:
                     prev_geo = TELEMETRY_STATE["last_geo_pos"]
                     prev_euc = TELEMETRY_STATE["last_euc_pos"]
                     # Move prev tensors to same device for comparison
                     if hasattr(prev_geo, 'to'):
                         prev_geo = prev_geo.to(current_geo.device)
                     if hasattr(prev_euc, 'to'):
                         prev_euc = prev_euc.to(current_euc.device)

                     d_M = PoincareBall.dist(current_geo, prev_geo, c=1.0).item()
                     d_E = torch.norm(current_euc - prev_euc).item() + 1e-9

                     ratio = d_M / d_E
                     TELEMETRY_STATE["lipschitz_ratio"] = ratio

                     if ratio > 5.0:
                         TELEMETRY_STATE["thought_trace"].append(f"TELEPORT DETECTED (L={ratio:.1f})")

                 # Store on CPU to free VRAM
                 TELEMETRY_STATE["last_geo_pos"] = current_geo.detach().cpu()
                 TELEMETRY_STATE["last_euc_pos"] = current_euc.detach().cpu()

                 try:
                     pt = current_geo[0, 0:2].float().cpu().numpy()
                     TELEMETRY_STATE["manifold_trace"].append(pt)
                     if len(TELEMETRY_STATE["manifold_trace"]) > 100:
                         TELEMETRY_STATE["manifold_trace"].pop(0)
                 except: pass

            TELEMETRY_STATE["active_fiber"] = str(active_fiber)

            # System 2 Refinement Detection
            if hasattr(geo_state, 'meta_info') and geo_state.meta_info:
                 meta = geo_state.meta_info
                 if meta.get("refined", False):
                      initial_e = meta.get("initial_energy", 0)
                      final_e = meta.get("final_energy", 0)
                      msg = f"SYSTEM 2 REFINEMENT: E {initial_e:.2f} -> {final_e:.2f}"
                      if not TELEMETRY_STATE["thought_trace"] or TELEMETRY_STATE["thought_trace"][-1] != msg:
                           TELEMETRY_STATE["thought_trace"].append(msg)

            TELEMETRY_STATE["history_k"].append(TELEMETRY_STATE["curvature"])
            TELEMETRY_STATE["history_s"].append(entropy)

            if len(TELEMETRY_STATE["history_k"]) > 50:
                TELEMETRY_STATE["history_k"].pop(0)
                TELEMETRY_STATE["history_s"].pop(0)

            # Cap thought trace to prevent unbounded memory growth
            if len(TELEMETRY_STATE["thought_trace"]) > 50:
                TELEMETRY_STATE["thought_trace"] = TELEMETRY_STATE["thought_trace"][-30:]

            if random.random() < 0.1:
                 TELEMETRY_STATE["thought_trace"].append(f"Activating {active_fiber} | S={entropy:.2f}")

        except Exception as e:
            print(f"Telemetry Error: {e}")

        adapted = adapted_out.to(orig_dtype)

        # --- FUNDAMENTAL FIX: Clamp adapter perturbation magnitude ---
        # The adapter does x + scale * output_proj(combined).
        # With scale=1.0 and unconstrained output_proj, the adapter can
        # produce perturbations that dwarf the base hidden states, destroying
        # the language model's distribution and causing degeneration.
        # Fix: clamp adapter perturbation to a safe fraction of base hidden state norm.
        # Trained at ~32%. Too high → degeneration. Too low → model goes mute.
        # 25% is close to trained ratio (~32%) but prevents worst degeneration modes.
        # Too low (10-20%) → model goes mute (adapter can't steer generation).
        # Too high (32%+) → degeneration loops ("-Shirt", fake Q&A).
        h_base = h.to(orig_dtype)
        delta = adapted - h_base  # Adapter's contribution
        base_norm = h_base.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        max_ratio = 0.25  # Adapter can perturb at most 25% of base signal
        scale_factor = torch.where(
            delta_norm > max_ratio * base_norm,
            max_ratio * base_norm / delta_norm,
            torch.ones_like(delta_norm)
        )
        adapted = h_base + delta * scale_factor

        # Periodic diagnostic: show actual perturbation ratio
        if not hasattr(adapter_hook, '_call_count'):
            adapter_hook._call_count = 0
        adapter_hook._call_count += 1
        if adapter_hook._call_count <= 1 or adapter_hook._call_count % 100 == 0:
            mean_ratio = (delta_norm / base_norm).mean().item()
            clamped_pct = (delta_norm > max_ratio * base_norm).float().mean().item() * 100
            print(f"[Adapter #{adapter_hook._call_count}] Perturbation: {mean_ratio:.1%} of base "
                  f"({clamped_pct:.0f}% clamped to {max_ratio:.0%})")

        if isinstance(out, tuple): return (adapted,) + out[1:]
        return adapted

    target_layer.forward = adapter_hook
    print("Hook Injected.")


# --- ROLLING CONTEXT MANAGEMENT ---
MAX_ROLLING_MESSAGES = 30   # System + last 30 non-system msgs (15 user/assistant turns)
MAX_GENERATION_TOKENS = 4096  # 8GiB GPU, ~3GiB free after NF4 model = ~54K tokens capacity

def _make_rolling_context(history: list, max_messages: int = MAX_ROLLING_MESSAGES) -> list:
    """
    Apply rolling window to conversation history.
    Keeps system messages + last `max_messages` non-system messages.
    Creates a compact summary of dropped messages to preserve continuity.
    """
    if not history:
        return history

    system_msgs = [m for m in history if m.get("role") == "system"]
    non_system = [m for m in history if m.get("role") != "system"]

    if len(non_system) <= max_messages:
        return history

    dropped = non_system[:-max_messages]
    kept = non_system[-max_messages:]

    # Build compact summary of dropped messages (extractive — no LLM call)
    summary_parts = []
    for msg in dropped:
        content = msg.get("content", "")
        first_line = content.split("\n")[0][:80]
        if first_line:
            tag = "User" if msg.get("role") == "user" else "Asst"
            summary_parts.append(f"- {tag}: {first_line}")

    if summary_parts:
        summary = "\n".join(summary_parts)
        if len(summary) > 500:
            summary = summary[:500] + "..."
        context_note = {
            "role": "system",
            "content": f"[Earlier conversation ({len(dropped)} msgs):\n{summary}]"
        }
        return system_msgs + [context_note] + kept

    return system_msgs + kept


def _safe_empty_cache():
    """torch.cuda.empty_cache() that won't crash if CUDA context is corrupted."""
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass  # CUDA context may be irrecoverable — don't cascade

def _cleanup_vram():
    """Aggressive VRAM cleanup between turns."""
    VISION_CONTEXT['feats'] = None
    # Move telemetry tensors to CPU
    for key in ("last_geo_pos", "last_euc_pos"):
        t = TELEMETRY_STATE.get(key)
        if t is not None and hasattr(t, 'cpu'):
            try:
                TELEMETRY_STATE[key] = t.cpu()
            except Exception:
                TELEMETRY_STATE[key] = None
    # Clear KV cache from previous generation (can hold 100s of MB)
    llm = MODELS.get("llm")
    if llm is not None and hasattr(llm, '_past_key_values'):
        try:
            del llm._past_key_values
        except Exception:
            pass
    # Cap thought trace
    trace = TELEMETRY_STATE.get("thought_trace", [])
    if len(trace) > 50:
        TELEMETRY_STATE["thought_trace"] = trace[-30:]
    import gc
    gc.collect()
    _safe_empty_cache()
    gc.collect()  # Double collect — releases cyclic references missed on first pass


# --- GREETING PREFIX STRIPPING (module-level for use in both generate_stream and bot_turn) ---
_GREETING_PREFIXES = [
    "Hello! I'm Neural Glass",
    "Hello! I am Neural Glass",
    "Hi! I'm Neural Glass",
    "Hi! I am Neural Glass",
    "Hello! I'm your",
    "Hello! I am your",
    "Hello! I'll help",
    "Hi there! I'm",
    "Greetings! I'm",
    "I'm Neural Glass, your",
    "I am Neural Glass, your",
    "I'm Neural Glass.",
    "I am Neural Glass.",
    "As Neural Glass,",
    "How may I assist you",
    "How can I assist you",
    "How can I help you",
    "How may I help you",
    "What can I help you with",
    "What would you like",
    "Neural Glass is a",
    "Neural Glass is",
    "I am Nezha",
    "I'm Nezha",
    "Nezha here",
    "My name is",
    "Sure thing!",
    "Sure! Let me",
    "Alrighty then",
    "Here goes nothing",
    "Let me break down your request",
]

def _strip_greeting_prefix(text: str) -> str:
    """Remove known greeting boilerplate from start of output."""
    if not text:
        return text
    stripped = text.lstrip()
    for prefix in _GREETING_PREFIXES:
        if stripped.startswith(prefix):
            # Skip past the greeting paragraph (up to next double-newline or end of line)
            double_nl = stripped.find('\n\n', len(prefix))
            if double_nl >= 0:
                result = stripped[double_nl + 2:].lstrip('\n')
                if result:
                    return result
            # Try single newline
            single_nl = stripped.find('\n', len(prefix))
            if single_nl >= 0:
                result = stripped[single_nl + 1:].lstrip('\n')
                if result:
                    return result
            # Entire output is just a greeting
            return ""
    return text


# --- INFERENCE GEN ---
def generate_stream(text, image_path, max_new_tokens):
    load_models()

    # Process Vision (SigLIP lives on CPU — move to GPU only when needed)
    VISION_CONTEXT['feats'] = None
    if image_path:
        if isinstance(image_path, dict):
             if 'path' in image_path and image_path['path']:
                 image_path = image_path['path']
             elif 'name' in image_path:
                 image_path = image_path['name']

        try:
            img = Image.open(image_path).convert("RGB")
            inputs = MODELS["processor"](images=img, return_tensors="pt")
            # Move SigLIP to GPU temporarily for inference
            vis_model = MODELS["vision_model"]
            vis_model.to("cuda")
            with torch.no_grad():
                vis_out = vis_model(inputs.pixel_values.to("cuda").to(torch.float16))
                VISION_CONTEXT['feats'] = vis_out.last_hidden_state.cpu()  # Keep features on CPU
            # Move SigLIP back to CPU immediately to free ~500MB VRAM
            vis_model.to("cpu")
            del vis_out
            _safe_empty_cache()
        except Exception as e:
            # Ensure SigLIP is back on CPU even if processing fails
            try:
                MODELS["vision_model"].to("cpu")
                _safe_empty_cache()
            except Exception:
                pass
            yield f"Error loading image: {e}"
            return

    # Truncate individual message contents to prevent massive payloads
    if isinstance(text, list):
        for msg in text:
            if isinstance(msg, dict) and isinstance(msg.get('content'), str):
                if len(msg['content']) > 6000:
                    msg['content'] = msg['content'][:3000] + "\n...[truncated]...\n" + msg['content'][-3000:]
    print(f"DEBUG: text type={type(text)}, len={len(text) if isinstance(text, (str, list)) else '?'}")

    prompt_text = ""
    prompt_ids = None
    mem_ctx = ""

    if isinstance(text, list):
        sanitized_history = []
        for msg in text:
             content = msg['content']
             if isinstance(content, list):
                  text_parts = [p['text'] for p in content if 'text' in p]
                  content = "\n".join(text_parts)
             elif not isinstance(content, str):
                  content = str(content)
             sanitized_history.append({"role": msg['role'], "content": content})

        tokenizer = MODELS["tokenizer"]

        # --- TOKEN-AWARE CONTEXT MANAGEMENT ---
        INPUT_TOKEN_BUDGET = 6144  # Allows ~4K words of context; leaves ~26K for generation within 32K KV

        # NOTE: rolling window already applied by bot_turn before calling generate_stream

        # 2. Extract last user message for mem0 query
        last_msg = ""
        for _m in reversed(sanitized_history):
            if _m['role'] == 'user':
                last_msg = _m['content']
                break

        # 3. Retrieve memory context BEFORE template application
        # Apply Lipschitz damping: high ratio = manifold teleportation = memory is
        # pulling trajectory away from its natural geodesic. Reduce memory influence.
        # Skip memory retrieval entirely during self-heal (prevents echo re-injection)
        _skip_memory = TELEMETRY_STATE.pop("_skip_memory_retrieval", False)
        _memory = MODELS.get("memory")
        if _memory is not None and last_msg and not _skip_memory:
            try:
                mem_ctx = _memory.get_context_string(last_msg)
                # Deduplication: if memory context overlaps >40% with user message,
                # the memory is just echoing the input — discard it
                if mem_ctx and len(last_msg) > 100:
                    overlap_chunk = last_msg[:150]
                    if overlap_chunk in mem_ctx:
                        print(f"DEBUG: Memory discarded — echo of user input ({len(mem_ctx)} chars)")
                        mem_ctx = ""
                if mem_ctx:
                    lip_ratio = TELEMETRY_STATE.get("lipschitz_ratio", 1.0)
                    if lip_ratio > 5.0:
                        # High Lipschitz = memory causing teleportation → heavy damping
                        max_mem_chars = max(100, int(len(mem_ctx) / lip_ratio))
                        mem_ctx = mem_ctx[:max_mem_chars]
                        print(f"DEBUG: Memory damped: L={lip_ratio:.1f} -> {max_mem_chars} chars")
                        TELEMETRY_STATE["thought_trace"].append(
                            f"MEMORY DAMPED: L={lip_ratio:.1f}, {max_mem_chars} chars"
                        )
                    else:
                        print(f"DEBUG: Memory retrieved {len(mem_ctx)} chars")
                        TELEMETRY_STATE["thought_trace"].append(f"MEMORY RECALL: {mem_ctx[:100]}...")
            except Exception as e:
                print(f"DEBUG: Memory retrieval error: {e}")

        # 4. Build ONE unified system message (preset + memory in single block)
        sys_content = MODEL_STATE.state.get("system_prompt",
            "Answer directly with step-by-step reasoning. Be concise."
        )
        if mem_ctx:
            # Strip tier tags [W]/[E]/[S] and header — model treats them as text to continue
            _clean_mem = mem_ctx.replace("[Memory Context]", "").strip()
            _clean_mem = "\n".join(
                line.lstrip("- ").replace("[W] ", "").replace("[E] ", "").replace("[S] ", "")
                for line in _clean_mem.splitlines() if line.strip()
            )
            if _clean_mem:
                sys_content += f"\n\nRelevant context:\n{_clean_mem}"

        sanitized_history = [m for m in sanitized_history if m['role'] != 'system']
        sanitized_history.insert(0, {"role": "system", "content": sys_content})

        # 5. Char pre-filter to avoid tokenizing huge contexts
        while sum(len(m['content']) for m in sanitized_history) > 24000 and len(sanitized_history) > 2:
            sanitized_history.pop(1)

        # 6. Check for "continue" prompt to seamlessly resume generations
        is_continue = False
        if len(sanitized_history) >= 2:
            last_msg = sanitized_history[-1]
            if last_msg['role'] == 'user' and last_msg['content'].strip().lower() in ["continue", "continue.", "keep going", "keep going."]:
                prev_msg = sanitized_history[-2]
                if prev_msg['role'] == 'assistant' and prev_msg['content'].strip():
                    is_continue = True
                    sanitized_history.pop(-1) # remove the "continue" user prompt
                    print("DEBUG: Intercepted 'continue' prompt -> applying continue_final_message=True")

        def _apply_template(history):
            if is_continue:
                return tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False, continue_final_message=True)
            else:
                return tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

        # 7. Precise token-based trimming
        prompt_text = _apply_template(sanitized_history)
        input_tokens = len(tokenizer.encode(prompt_text))

        while input_tokens > INPUT_TOKEN_BUDGET and len(sanitized_history) > 2:
            sanitized_history.pop(1)
            prompt_text = _apply_template(sanitized_history)
            input_tokens = len(tokenizer.encode(prompt_text))
            print(f"DEBUG: Trimmed to {len(sanitized_history)} msgs ({input_tokens} tokens)")

        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        print(f"DEBUG: Context: {inputs.input_ids.shape[-1]} tokens, {len(sanitized_history)} messages")
    else:
        if not isinstance(text, str): text = str(text)
        if len(text) > 4000: text = text[-4000:]
        prompt_text = text
        inputs = MODELS["tokenizer"](text, return_tensors="pt").to("cuda")

    # --- CONSTRAINT EXTRACTION ---
    constraints = []
    last_msg = ""
    if "constraint_extractor" in MODELS:
        if 'sanitized_history' in locals() and sanitized_history:
             last_msg = sanitized_history[-1]['content']
        else:
             if isinstance(text, list) and len(text) > 0:
                 last_msg = text[-1]['content']
             else:
                 last_msg = str(text) if not isinstance(text, list) else ""

        if not isinstance(last_msg, str): last_msg = str(last_msg)
        extract_input = last_msg[-1000:]

        try:
            constraints = MODELS["constraint_extractor"].extract(extract_input)

            # Memory constraints (NO re-injection — already in unified system prompt)
            if mem_ctx:
                 try:
                     mem_constraints = MODELS["constraint_extractor"].extract(mem_ctx)
                     if mem_constraints:
                          constraints.extend(mem_constraints)
                          print(f"DEBUG: Memory Attractors: {mem_constraints}")
                 except Exception as e:
                     print(f"Memory constraint extraction error: {e}")

            TELEMETRY_STATE["active_constraints"] = list(set(constraints))
            TELEMETRY_STATE["constraint_score"] = 0.0 if constraints else 1.0
            if constraints:
                print(f"Active Attractors: {TELEMETRY_STATE['active_constraints']}")
                TELEMETRY_STATE["thought_trace"].append(f"ATTRACTORS: {TELEMETRY_STATE['active_constraints']}")
        except Exception as e:
            print(f"Constraint/Memory Extraction Failed: {e}")
    else:
        if isinstance(text, list) and len(text) > 0:
            last_msg = text[-1]['content']
        else:
            last_msg = str(text) if not isinstance(text, list) else ""
        if not isinstance(last_msg, str): last_msg = str(last_msg)

    model = MODELS["llm"]

    print(f"DEBUG: Starting Generation. Input Length={len(prompt_text)}")

    # Aggressive pre-generation VRAM cleanup
    import gc
    VISION_CONTEXT['feats'] = None  # Release vision tensors before generation
    gc.collect()
    _safe_empty_cache()
    try:
        # PyTorch allocator metrics — more accurate than OS-level mem_get_info
        _reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        _allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        _pool_free = _reserved - _allocated  # Available within PyTorch's reserved pool
        _os_free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
        _total_available = _pool_free + _os_free
        print(f"DEBUG: VRAM before gen: pool={_pool_free:.0f}MB + os={_os_free:.0f}MB = {_total_available:.0f}MB available")
    except Exception:
        _total_available = 500  # Conservative fallback
        pass

    # --- THERMODYNAMIC SAMPLING ---
    # High |K| = unstable manifold region = COOL DOWN (reduce randomness)
    # Low  |K| = stable region = can afford slight exploration
    _gen_params = MODEL_STATE.generation_params
    base_temp = _gen_params.get("temperature", 0.3)
    curv = TELEMETRY_STATE.get("curvature", 0.0)

    # Self-healing override: if bot_turn injected a cooler temperature, use it
    _heal_override = TELEMETRY_STATE.pop("_heal_temp_override", None)
    if _heal_override is not None:
        dynamic_temp = _heal_override
        print(f"DEBUG: SELF-HEAL temperature override -> T={dynamic_temp:.2f}")
    else:
        curv_factor = 1.0 - 0.1 * min(abs(curv), 3.0)
        dynamic_temp = base_temp * curv_factor
        dynamic_temp = max(0.15, min(dynamic_temp, 0.45))
        print(f"DEBUG: Thermodynamic Sampling | K={curv} -> factor={curv_factor:.2f} -> T={dynamic_temp:.2f}")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)

    # Adaptive generation budget — fit within remaining KV cache headroom
    _input_len = inputs.input_ids.shape[-1] if hasattr(inputs, 'input_ids') else 1000
    _remaining = 32768 - _input_len - 64
    safe_max_tokens = max(512, min(max_new_tokens or MAX_GENERATION_TOKENS, _remaining, MAX_GENERATION_TOKENS))

    # VRAM-aware dynamic reduction — use PyTorch allocator metrics (not OS-level)
    try:
        # _total_available computed above: pool_free + os_free
        # KV cache per token: 28 layers × 2(K/V) × 4 KV heads × 128 dim × 2 bytes = 56 KB
        # With SDPA/flash attention, activations are O(1) not O(N²)
        vram_safe_tokens = int((_total_available * 1024) / 56)  # MB -> KB -> tokens
        if vram_safe_tokens < safe_max_tokens:
            safe_max_tokens = max(1024, vram_safe_tokens)  # Floor raised 512→1024 for reasoning
            print(f"DEBUG: VRAM constraint: {_total_available:.0f}MB avail -> capped to {safe_max_tokens} tokens")
    except Exception:
        pass

    print(f"DEBUG: Generation budget: {safe_max_tokens} tokens (input={_input_len}, headroom={_remaining})")

    # Stop event: main thread signals generation to halt on corruption
    _stop_generation = threading.Event()

    # Qwen chat uses <|im_end|> as turn terminator AND <|endoftext|> as EOS.
    # Both must be stop tokens — otherwise model either stops too early or never stops.
    _eos_ids = [tokenizer.eos_token_id]
    _im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if _im_end_id is not None and _im_end_id != tokenizer.eos_token_id:
        _eos_ids.append(_im_end_id)

    # New-turn hallucination guard: stop if model emits <|im_start|> or continues after <|im_end|>
    _im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    _prompt_len = inputs.input_ids.shape[-1]
    _stop_criteria = [_StopOnEvent(_stop_generation)]
    if _im_start_id is not None and _im_end_id is not None:
        _stop_criteria.append(_StopOnNewTurn(_im_start_id, _im_end_id, _prompt_len, min_gen=10))
        print(f"DEBUG: NewTurn guard active (im_start={_im_start_id}, im_end={_im_end_id}, prompt_len={_prompt_len})")
    else:
        print(f"WARNING: NewTurn guard DISABLED — im_start={_im_start_id}, im_end={_im_end_id}")

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=safe_max_tokens,
        do_sample=True,
        temperature=dynamic_temp,
        top_p=_gen_params.get("top_p", 0.95),
        top_k=_gen_params.get("top_k", 50),
        repetition_penalty=1.10, # Balanced: 1.05 too low (subtle reps), 1.2 broke EOS
        min_new_tokens=8,  # Reduced from 32: high values force hallucination at natural stops
        eos_token_id=_eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList(_stop_criteria)
    )

    def generate_thread():
        try:
             model.generate(**generation_kwargs)
        except torch.cuda.OutOfMemoryError as e:
             print(f"CUDA OOM in Generate Thread: {e}")
             gc.collect()
             _safe_empty_cache()
             # Retry with halved token budget
             retry_tokens = max(256, generation_kwargs.get("max_new_tokens", 1536) // 2)
             print(f"DEBUG: OOM retry with {retry_tokens} tokens")
             try:
                 retry_kwargs = dict(generation_kwargs)
                 retry_kwargs["max_new_tokens"] = retry_tokens
                 model.generate(**retry_kwargs)
             except torch.cuda.OutOfMemoryError:
                 print(f"CUDA OOM on retry too — giving up")
                 streamer.text_queue.put("[CUDA Out of Memory -- try a shorter prompt or restart]")
                 _safe_empty_cache()
             except Exception as e2:
                 print(f"ERROR in OOM retry: {e2}")
                 streamer.text_queue.put(f"[Generation Error: {e2}]")
        except Exception as e:
             print(f"ERROR in Generate Thread: {e}")
             streamer.text_queue.put(f"[Generation Error: {e}]")
        finally:
             streamer.end()
             gc.collect()
             _safe_empty_cache()

    thread = threading.Thread(target=generate_thread)
    thread.start()

    # --- MEMORY STORAGE (user msg + bot response later) ---
    # Only store SHORT user messages (< 500 chars). Bulk inputs like salvos
    # would pollute memory and get re-injected, doubling context → OOM.
    def save_memory_thread():
         memory = MODELS.get("memory")
         if memory is not None and last_msg and len(last_msg) < 500:
             try:
                 memory.add(last_msg)
                 print(f"DEBUG: Memory recorded user message ({len(last_msg)} chars).")
             except Exception as e:
                 print(f"Memory Save Error: {e}")
             finally:
                 gc.collect()
                 _safe_empty_cache()
         elif memory is not None and last_msg:
             print(f"DEBUG: Skipped memory storage — message too long ({len(last_msg)} chars)")

    mem_thread = threading.Thread(target=save_memory_thread)
    mem_thread.start()


    partial_text = ""
    # --- REPETITION DETECTION (v2: substring-based) ---
    # --- AI VOICE CATALOGUE ---
    # All detected persona leakage patterns from salvo runs.
    # Category A: Explicit persona injection (model claims to be someone else)
    # Category B: Tokenization artifacts (role separator bleed-through)
    # Category C: System prompt corruption (own prompt leaking into output)
    # Category D: Generic AI boilerplate / verbose filler

    # Dynamic: extract distinctive fragment from current system prompt for leak detection
    _sys_prompt_raw = MODEL_STATE.state.get("system_prompt", "")
    _SYS_PROMPT_FRAGMENT = None
    if _sys_prompt_raw and len(_sys_prompt_raw) >= 20:
        # Use first 40 chars as a unique fingerprint — enough to detect verbatim leakage
        _SYS_PROMPT_FRAGMENT = _sys_prompt_raw[:40]

    _LEAKAGE_FRAGMENTS = [
        # --- Cat A: Foreign persona injection ---
        "You are Elizabeth",               # 77 occurrences in cp3001 salvos
        "izabeth, a highly intelligent",   # boundary-corrupted Elizabeth variant
        "I am Nezha",                      # Qwen identity drift
        "I'm Nezha",
        "Nezha, a virtual",
        # --- Cat B: Tokenization / role separator artifacts ---
        "inquiringuser",                   # 85 occurrences — role tag bleed
        "inquiringassistant",              # 84 occurrences
        "yttuser",                         # 33 occurrences — separator corruption
        "yttassistant",                    # 33 occurrences
        "atismuser",                       # 12 occurrences — another separator variant
        "\u00e4\u00dfuser",               # Qwen <|im_start|>user decoded as garbled UTF-8
        "\u00dfassistant",                # Qwen <|im_start|>assistant decoded as garbled UTF-8
        "\u00e4\u00df",                   # Generic Qwen separator artifact prefix
        "atism",                           # Generic Qwen separator artifact prefix variant
        "\nA:\n",                          # Single-letter dialogue role separator
        "\nQ:\n",                          # Question/Answer dialogue role separator
        "\nUser:\n",                       # Explicit User: role marker with newlines
        "ownerId:",                        # Hallucinated credentials/PII
        "password:",                       # Hallucinated credentials/PII
        "<|im_start|>",                    # Raw chat template token leaked into output
        "<|im_end|>",                      # Raw chat template end token
        "<|im_sep|>",                      # Raw chat template separator
        # --- Cat C: Own system prompt leaking into output ---
        "You are Neural Glass",            # 474 occurrences — should never appear in output
        "Neural Glass is a",              # Any product description hallucination (e.g. "Neural Glass is a...")
        "is a powerful tool that can help", # Generic tool description hallucination
        "Think step-by-step and provide clear",
        "Help as much as you can",
        "Compute answers directly",
        "Answer directly with step-by-step",  # Default system prompt leaking verbatim
        "show your reasoning step-by-step with",
        "follows instruction extremely well",  # 224 occurrences — generic AI instruction
        # --- Cat D: Generic AI boilerplate / identity drift ---
        # NOTE: Greeting patterns ("Hello! I am", "Hello! I'm") are NOT here.
        # They are handled by _strip_greeting_prefix (silent stripping, no self-heal).
        "You are an AI assistant",
        "As an AI language model",
        "As a large language model",
        "I am a virtual assistant",
        "I'm a virtual assistant",
        "powered by artificial intelligence",
        "natural language processing (NLP)",
        "ad infinitum perpetually",
        "I cannot provide",                # refusal pattern (model isn't supposed to refuse)
        "I apologize, but as an AI",
        "please provide me with",          # evasive "tell me more" instead of answering
        "using advanced algorithms",       # generic AI marketing boilerplate
        "machine learning techniques",     # generic AI marketing boilerplate
        "bear patience",                   # patronizing filler
        "tailored specifically towards",   # generic AI marketing
        "What can I help you with",        # evasive redirection
        # --- Cat E: Code generation (RULE 2 violation) ---
        "```python",                       # Code block start
        "```javascript",
        "```java",
        "```cpp",
        "\ndef ",                           # Function definition at line start
        "\nimport ",                        # Import statement at line start
        "\nclass ",                         # Class definition at line start
        "print(",                          # Print function call (code gen violation)
        ".append(",                        # List append — hallucinated code pattern
        " = [",                            # List assignment — hallucinated code
        " = {",                            # Dict assignment — hallucinated code
        # --- Cat F: Hallucinated Q&A dataset pattern ---
        "\nquestion\n",                    # Fake Q&A format (no colon)
        "\nanswer\n",                      # Fake Q&A answer block
        "\nQuestion:",                     # Fake Q&A with colon
        "\nAnswer:",                       # Fake Q&A answer with colon
        # --- Cat G: Hallucinated new-prompt continuation ---
        # Model finishes answer then generates a new instruction/question without <|im_end|>
        "Please answer the following",     # Most common hallucinated prompt prefix
        "Given the sentence '",            # NLI-style hallucinated prompt
        "Given the following",             # Generic hallucinated prompt
        "Based on the passage",            # RC-style hallucinated prompt
        "Read the following passage",      # RC-style hallucinated prompt
        "Consider the following",          # Generic hallucinated prompt
        "\nInstruction:",                  # Hallucinated instruction block
        "\nPrompt:",                       # Hallucinated prompt block
    ]
    # Dynamic system prompt fingerprint — catches verbatim leakage of ANY configured prompt
    if _SYS_PROMPT_FRAGMENT:
        _LEAKAGE_FRAGMENTS.append(_SYS_PROMPT_FRAGMENT)
    _repeat_halt = False
    _last_clean_len = 0

    # Store user's last message for echo detection
    _user_input_for_echo = last_msg[:500] if isinstance(last_msg, str) and len(last_msg) > 100 else ""

    def _check_repetition_v2(full_text: str, min_block: int = 80) -> bool:
        """
        Detect repeated blocks in accumulated output.
        Multi-layer detection: short-pattern degeneration, substring repetition,
        leakage fragments, and hallucinated conversation patterns.
        """
        n = len(full_text)

        # --- LAYER 0: Short-pattern degeneration (fires early, catches "-Shirt" × N) ---
        # If the last 60 chars consist of a repeating pattern of ≤30 chars, it's degenerate.
        # This catches single-token loops MUCH earlier than the 160-char min_block threshold.
        if n >= 60:
            recent_60 = full_text[-60:]
            for plen in range(2, 31):
                pat = recent_60[:plen]
                reps = 60 // plen
                if reps >= 3 and pat * reps == recent_60[:plen * reps]:
                    print(f"DEBUG: Short-pattern degeneration: '{pat[:20]}' × {reps}")
                    return True

        if n < min_block * 2:
            return False

        # Echo detection: model copying the user's input back as output
        if _user_input_for_echo and n >= 100:
            # Check if a 100-char chunk from the input appears in the output
            input_chunk = _user_input_for_echo[:100]
            if input_chunk in full_text:
                return True

        # Check if the last block appears earlier (substring repetition)
        tail = full_text[-min_block:]
        earlier = full_text[:-min_block]
        if tail in earlier:
            return True

        # Longer block check (catches multi-line code loops)
        if n >= 300:
            long_tail = full_text[-200:]
            long_earlier = full_text[:-200]
            if long_tail in long_earlier:
                return True

        # Degeneration detector: incrementing counter patterns
        # Catches: "userType(1)=18 userType(2)=6 userType(3)=18..."
        # Method: strip digits, then check if the last 60 chars appear earlier
        import re
        stripped = re.sub(r'\d+', '#', full_text)
        if len(stripped) >= 120:
            stripped_tail = stripped[-60:]
            stripped_earlier = stripped[:-60]
            if stripped_tail in stripped_earlier:
                return True

        # System prompt leakage in last 200 chars
        recent = full_text[-200:]
        for frag in _LEAKAGE_FRAGMENTS:
            if frag in recent:
                return True

        # Hallucinated conversation detector: model generating fake multi-turn dialogue
        # REGEX-based: catches ANY garbled role separator (present + future variants)
        # Matches: standalone line that is a single fused token ending with "user" or "assistant"
        # This catches: atismuser, inquiringuser, yttuser, äßuser, etc.
        # Won't match "The user asked" — only fused junk tokens without internal spaces
        import re
        _role_sep_re = re.compile(
            r'(?:^|\n)\s*(?:[^\s\n]{0,20}(?:user|assistant)\s*:?|[AQU]\s*:)\s*(?:\n|$)',
            re.IGNORECASE
        )
        if _role_sep_re.search(full_text):
            return True

        # Also check explicit markers for single-occurrence detection
        _convo_markers = ["\u00e4\u00dfuser", "\u00dfassistant", "inquiringuser",
                          "inquiringassistant", "yttuser", "yttassistant",
                          "atismuser", "atism",
                          "<|im_start|>", "<|im_end|>"]
        if any(m in full_text for m in _convo_markers):
            return True

        # Degeneration detector: very long run without sentence-ending punctuation
        # indicates verbose gibberish (the "Nezha" failure mode)
        # Relaxed from 300→600 chars — 300 caused false positives on multi-hop reasoning
        last_800 = full_text[-800:] if n >= 800 else full_text
        sentences = last_800.replace('!', '.').replace('?', '.').replace(':', '.').replace(';', '.').split('.')
        if sentences:
            longest_run = max(len(s) for s in sentences)
            if longest_run > 600:
                return True

        return False

    # Conversation hallucination markers — truncate aggressively at FIRST occurrence
    _CONVO_HALT_MARKERS = [
        "\u00e4\u00dfuser", "\u00dfassistant", "\u00e4\u00df",
        "atismuser", "atism",
        "<|im_start|>", "<|im_end|>", "<|im_sep|>",
        "\nA:\n", "\nQ:\n", "\nUser:\n", "\nAssistant:\n",
        "ownerId:", "password:", "username:",  # Hallucinated PII
        "Neural Glass is a",                   # Product hallucination (any variant)
        "is a powerful tool that can help",    # Generic tool hallucination
        "\nquestion\n", "\nanswer\n",          # Fake Q&A dataset pattern
        "\nQuestion:", "\nAnswer:",            # Fake Q&A with colon
        "Please answer the following",         # Hallucinated new-prompt continuation
        "Given the sentence '",                # NLI hallucinated prompt
        "\nInstruction:", "\nPrompt:",         # Hallucinated instruction block
    ]

    # Regex: garbled role separator on its own line (catches ALL future variants)
    # Must be a STANDALONE line of <30 chars that's just garbage+user/assistant
    # Won't match "The user asked" (has space before "user") — only fused junk like "atismuser"
    import re as _re_mod
    _ROLE_SEP_STRIP_RE = _re_mod.compile(
        r'(?:^|\n)\s*(?:[^\s\n]{0,20}(?:user|assistant)\s*:?|[AQU]\s*:)\s*(?:\n|$)',
        _re_mod.IGNORECASE
    )

    def _strip_prompt_leakage(text_out: str) -> str:
        # 0. Strip greeting prefix (handles "Hello! I'm Neural Glass..." at position 0)
        text_out = _strip_greeting_prefix(text_out)
        if not text_out:
            return ""

        # 1. Regex: truncate at first garbled role separator line
        role_match = _ROLE_SEP_STRIP_RE.search(text_out)
        if role_match:
            cut_pos = role_match.start()
            if cut_pos > 0:
                text_out = text_out[:cut_pos].rstrip()
            else:
                # Role separator at very start — try to skip past it
                end_pos = role_match.end()
                text_out = text_out[end_pos:].lstrip('\n')
            if not text_out:
                return ""

        # 2. Explicit marker truncation (backup for regex edge cases)
        earliest_convo = len(text_out)
        for marker in _CONVO_HALT_MARKERS:
            idx = text_out.find(marker)
            if idx >= 0 and idx < earliest_convo:
                earliest_convo = idx
        if earliest_convo < len(text_out):
            text_out = text_out[:earliest_convo].rstrip()
            if not text_out:
                return ""

        # 3. Standard leakage stripping — truncate at first leakage fragment
        for frag in _LEAKAGE_FRAGMENTS:
            idx = text_out.find(frag)
            if idx >= 0:
                if idx < 10:
                    # Leakage at/near start: skip past the line containing it
                    newline = text_out.find('\n', idx + len(frag))
                    if newline >= 0:
                        remainder = text_out[newline + 1:].lstrip('\n')
                        if remainder:
                            text_out = remainder
                            continue  # Re-check for more leakage
                    return ""  # Entire output is leakage
                else:
                    return text_out[:idx].rstrip()
        return text_out

    try:
        first_token = True
        token_count = 0
        _corruption_type = None  # Track what kind of corruption was detected
        _last_token_time = time.time()
        _TOKEN_TIMEOUT = 60  # seconds — if no token arrives in 60s, generation is stuck
        _last_unique_text = ""  # Track last unique token for consecutive-repeat detection
        _consecutive_repeat_count = 0

        for new_text in streamer:
            token_count += 1
            if first_token:
                print("DEBUG: Stream started receiving tokens.")
                first_token = False

            _last_token_time = time.time()
            if not new_text: continue

            # --- CONSECUTIVE TOKEN REPEAT DETECTOR (fires EVERY token, no interval) ---
            # Catches single-token degeneration like "-Shirt" × N immediately
            _stripped_tok = new_text.strip()
            if _stripped_tok and _stripped_tok == _last_unique_text:
                _consecutive_repeat_count += 1
                if _consecutive_repeat_count >= 8:
                    print(f"DEBUG: Consecutive token repeat '{_stripped_tok[:20]}' × {_consecutive_repeat_count}")
                    _repeat_halt = True
                    _last_clean_len = max(0, len(partial_text) - len(new_text) * _consecutive_repeat_count)
                    _corruption_type = "repetition"
                    TELEMETRY_STATE["thought_trace"].append(
                        f"SELF-HEAL: token-repeat '{_stripped_tok[:15]}' × {_consecutive_repeat_count} at token {token_count}"
                    )
                    _stop_generation.set()
                    break
            else:
                if _stripped_tok:
                    _last_unique_text = _stripped_tok
                    _consecutive_repeat_count = 0

            partial_text += new_text

            # Full corruption check — only after 200 chars to avoid false positives on greetings.
            # The consecutive-token detector (layer 0 above) still fires every token for true degeneration.
            if token_count % 4 == 0 and len(partial_text) >= 200 and _check_repetition_v2(partial_text):
                if not _repeat_halt:
                    _repeat_halt = True
                    _last_clean_len = len(partial_text)
                    tail = partial_text[-80:]
                    first_occurrence = partial_text.find(tail)
                    if first_occurrence >= 0 and first_occurrence < _last_clean_len - 80:
                        _last_clean_len = first_occurrence + 80

                    # Classify the corruption type for healing
                    import re as _re
                    recent = partial_text[-200:]
                    full = partial_text

                    # Hallucinated conversation: regex-based role separator detection
                    _role_sep_re = _re.compile(
                        r'(?:^|\n)\s*(?:[^\s\n]{0,20}(?:user|assistant)\s*:?|[AQU]\s*:)\s*(?:\n|$)',
                        _re.IGNORECASE
                    )
                    _convo_markers = ["\u00e4\u00dfuser", "\u00dfassistant", "inquiringuser",
                                      "inquiringassistant", "yttuser", "yttassistant",
                                      "atismuser", "atism",
                                      "<|im_start|>", "<|im_end|>"]
                    has_role_sep = bool(_role_sep_re.search(full))
                    has_explicit_marker = any(m in full for m in _convo_markers)

                    # Echo detection: model copying input back as output
                    is_echo = False
                    if _user_input_for_echo and len(full) >= 100:
                        input_chunk = _user_input_for_echo[:100]
                        is_echo = input_chunk in full

                    if is_echo:
                        _corruption_type = "echo"
                    elif has_role_sep or has_explicit_marker:
                        _corruption_type = "hallucinated_conversation"
                    elif any(f in recent for f in ["I am Nezha", "I'm Nezha", "Nezha, a virtual",
                                                      "You are Elizabeth", "izabeth, a highly",
                                                      "I am a virtual assistant"]):
                        _corruption_type = "identity_drift"
                    elif any(f in recent for f in ["You are Neural Glass", "Neural Glass is a powerful",
                                                    "is a powerful tool that can help",
                                                    "Compute answers directly",
                                                    "follows instruction extremely well"]):
                        _corruption_type = "prompt_leakage"
                    elif any(f in recent for f in ["\nquestion\n", "\nanswer\n",
                                                    "\nQuestion:", "\nAnswer:",
                                                    "print(", ".append(",
                                                    "\nfor ", "\ndef "]):
                        _corruption_type = "hallucinated_conversation"
                    else:
                        _corruption_type = "repetition"

                    print(f"DEBUG: Corruption [{_corruption_type}] at token {token_count}. Clean len={_last_clean_len}")
                    TELEMETRY_STATE["thought_trace"].append(
                        f"SELF-HEAL: {_corruption_type} at token {token_count}"
                    )
                    # Signal generation thread to STOP — prevents phantom forward passes
                    _stop_generation.set()
                    break

            # Constraint pressure loop
            if constraints and token_count % 100 == 0:
                scorer = MODELS["constraint_scorer"]
                scores = scorer.score(partial_text, constraints)
                avg_score = sum(scores.values()) / len(scores) if scores else 1.0
                TELEMETRY_STATE["constraint_score"] = avg_score

                if avg_score < 0.1 and token_count > 500 and random.random() < 0.3:
                     lip_violation = TELEMETRY_STATE.get("lipschitz_ratio", 1.0) > 8.0
                     if lip_violation:
                         msg = f"OOD TELEPORTATION (L={TELEMETRY_STATE['lipschitz_ratio']:.1f}) -> JUMP!"
                     else:
                         msg = f"CRITICAL VIOLATION ({avg_score:.2f}) -> [JUMPS DISABLED]"
                     if not TELEMETRY_STATE["thought_trace"] or TELEMETRY_STATE["thought_trace"][-1] != msg:
                        TELEMETRY_STATE["thought_trace"].append(msg)

            # Chunk UI updates to prevent Gradio WebSocket flooding
            # Only yield when stripped text has content (prevents greeting flash-then-vanish)
            if token_count < 200:
                if token_count % 3 == 0 or token_count == 1:
                    _cleaned = _strip_prompt_leakage(partial_text)
                    if _cleaned.strip():
                        yield _cleaned
            elif token_count < 1000:
                if token_count % 15 == 0: yield _strip_prompt_leakage(partial_text)
            else:
                if token_count % 50 == 0: yield _strip_prompt_leakage(partial_text)

            if token_count > safe_max_tokens:
                 print("DEBUG: Generation hit safe token limit. Forcing stop.")
                 _stop_generation.set()
                 break

        # Final yield — truncate if corrupted
        if partial_text:
            clean = _strip_prompt_leakage(partial_text)
            if _repeat_halt and _last_clean_len > 0:
                clean = clean[:_last_clean_len].rstrip()
            if clean:
                yield clean
            if len(clean) < len(partial_text):
                print(f"DEBUG: Stripped {len(partial_text) - len(clean)} chars of corruption")

        # Signal corruption type to bot_turn for self-healing
        TELEMETRY_STATE["_generation_corrupted"] = _corruption_type

    except queue.Empty:
        # TextIteratorStreamer raised timeout — generation thread stalled
        print(f"ERROR: Streamer timeout after {_TOKEN_TIMEOUT}s — generation stalled at token {token_count}")
        TELEMETRY_STATE["thought_trace"].append(f"STALL: timeout at token {token_count}")
        # Signal stall to bot_turn — token_count=0 means total VRAM exhaustion
        TELEMETRY_STATE["_generation_stalled"] = token_count
        if partial_text:
            yield _strip_prompt_leakage(partial_text) + "\n\n**[Generation stalled — VRAM pressure. Try a shorter prompt or restart.]**"
        else:
            yield "**[Generation timed out. The model may be under VRAM pressure. Try restarting.]**"
    except Exception as e:
         print(f"ERROR in Streamer Loop: {e}")
         yield f"[System Error: {e}]"
    finally:
        # Wait for generation thread to finish (with timeout) to prevent overlap with next turn
        if thread.is_alive():
            print("DEBUG: Waiting for generation thread to finish...")
            thread.join(timeout=10)
            if thread.is_alive():
                print("WARNING: Generation thread still alive after join timeout — proceeding anyway")
        # Snapshot telemetry to disk
        try:
            MODEL_STATE.snapshot_telemetry(TELEMETRY_STATE)
        except Exception as e:
            print(f"DEBUG: Telemetry snapshot error: {e}")
        # VRAM cleanup after generation
        _cleanup_vram()

# --- TELEMETRY POLLING ---
def get_zone_color(d):
    if d < 0.3:
        return 'rgba(0, 255, 100, 0.9)'
    elif d < 0.6:
        return 'rgba(255, 255, 0, 0.9)'
    elif d < 0.85:
        return 'rgba(255, 150, 0, 0.9)'
    else:
        return 'rgba(255, 50, 50, 0.9)'

def get_zone_name(d):
    if d < 0.3:
        return "ANCHOR"
    elif d < 0.6:
        return "BALANCED"
    elif d < 0.85:
        return "EXPLORATORY"
    else:
        return "BOUNDARY"

def poll_telemetry():
    try:
        damping = TELEMETRY_STATE.get("damping", 0.01)
        beta = compute_gibbs_temperature(damping)
        TELEMETRY_STATE["gibbs_beta"] = beta

        # Metrics History Plot
        history_fig = go.Figure()
        k_hist = TELEMETRY_STATE.get("history_k", [])
        s_hist = TELEMETRY_STATE.get("history_s", [])
        x_hist = list(range(len(k_hist)))

        history_fig.add_trace(go.Scatter(x=x_hist, y=k_hist, name="Curvature (K)", line=dict(color="#00d4ff", width=2)))
        history_fig.add_trace(go.Scatter(x=x_hist, y=s_hist, name="Entropy (S)", line=dict(color="#a855f7", width=2)))

        history_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=150,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor="#222", zeroline=False, tickfont=dict(size=8))
        )

        # Poincare projection
        fig = go.Figure()
        theta = np.linspace(0, 2*np.pi, 100)

        # Cognitive Zones
        r_anchor = 0.3
        fig.add_trace(go.Scatter(
            x=r_anchor * np.cos(theta), y=r_anchor * np.sin(theta),
            fill='toself', fillcolor='rgba(0, 255, 100, 0.15)',
            line=dict(color='rgba(0, 255, 100, 0.4)', width=1, dash='dot'),
            name='Anchor', hoverinfo='text',
            hovertext='ANCHOR ZONE | High confidence | System 1'
        ))

        r_balanced = 0.6
        fig.add_trace(go.Scatter(
            x=r_balanced * np.cos(theta), y=r_balanced * np.sin(theta),
            fill='tonext', fillcolor='rgba(255, 255, 0, 0.08)',
            line=dict(color='rgba(255, 255, 0, 0.3)', width=1, dash='dot'),
            name='Balanced', hoverinfo='text',
            hovertext='BALANCED ZONE | Weighing options'
        ))

        r_explore = 0.85
        fig.add_trace(go.Scatter(
            x=r_explore * np.cos(theta), y=r_explore * np.sin(theta),
            fill='tonext', fillcolor='rgba(255, 150, 0, 0.08)',
            line=dict(color='rgba(255, 150, 0, 0.3)', width=1, dash='dot'),
            name='Exploratory', hoverinfo='text',
            hovertext='EXPLORATORY ZONE | System 2 thinking'
        ))

        # Boundary
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines', line=dict(color='cyan', width=2),
            name='Boundary', hoverinfo='text',
            hovertext='STABILITY BOUNDARY'
        ))

        # Origin
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(size=10, color='lime', symbol='diamond', line=dict(color='white', width=1)),
            name='Anchor', hoverinfo='text', hovertext='SEMANTIC ANCHOR'
        ))

        # Thought Trajectory
        if TELEMETRY_STATE["manifold_trace"]:
            arr = np.array(TELEMETRY_STATE["manifold_trace"])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                n_points = len(arr)
                distances = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2)

                for i in range(len(arr) - 1):
                    opacity = 0.3 + 0.7 * i / max(n_points - 1, 1)
                    fig.add_trace(go.Scatter(
                        x=[arr[i, 0], arr[i+1, 0]], y=[arr[i, 1], arr[i+1, 1]],
                        mode='lines', line=dict(color=f'rgba(255, 0, 255, {opacity})', width=2),
                        showlegend=False, hoverinfo='skip'
                    ))

                colors = [get_zone_color(d) for d in distances]
                sizes = [4 + 8 * i/max(n_points, 1) for i in range(n_points)]
                hover_texts = [f'Step {i+1} | Zone: {get_zone_name(d)} | r={d:.2f}' for i, d in enumerate(distances)]

                fig.add_trace(go.Scatter(
                    x=arr[:, 0], y=arr[:, 1], mode='markers',
                    marker=dict(size=sizes, color=colors, line=dict(color='white', width=0.5)),
                    name='Steps', hoverinfo='text', hovertext=hover_texts
                ))

                if n_points > 0:
                    current_d = distances[-1]
                    if current_d < 0.3: star_color, status = 'lime', 'ANCHORED'
                    elif current_d < 0.6: star_color, status = 'yellow', 'BALANCED'
                    elif current_d < 0.85: star_color, status = 'orange', 'EXPLORING'
                    else: star_color, status = 'red', 'UNSTABLE'

                    fig.add_trace(go.Scatter(
                        x=[arr[-1, 0]], y=[arr[-1, 1]], mode='markers',
                        marker=dict(size=14, color=star_color, symbol='star', line=dict(color='white', width=2)),
                        name='Current', hoverinfo='text',
                        hovertext=f'CURRENT: {status} | Bundle: {TELEMETRY_STATE["active_fiber"]} | r={current_d:.3f}'
                    ))

        rw_status = "ok" if beta > 1.87 else "low"
        rw_color = 'lime' if beta > 1.87 else 'orange'

        fig.update_layout(
            title=dict(text="Poincare Manifold Projection", font=dict(size=12, color='cyan'), x=0.5),
            template="plotly_dark",
            xaxis=dict(range=[-1.25, 1.25], showgrid=False, zeroline=False, showticklabels=False, title=None, fixedrange=True),
            yaxis=dict(range=[-1.25, 1.25], showgrid=False, zeroline=False, showticklabels=False, title=None, scaleanchor='x', fixedrange=True),
            width=320, height=320,
            margin=dict(l=5, r=5, t=35, b=45),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            annotations=[
                dict(x=0, y=1.12, text='<b>ABSTRACT</b>', showarrow=False, font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=0, y=-1.12, text='<b>CONCRETE</b>', showarrow=False, font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=1.12, y=0, text='<b>CREATIVE</b>', showarrow=False, font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=-1.12, y=0, text='<b>ANALYTICAL</b>', showarrow=False, font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=0, y=0.15, text='ANCHOR', showarrow=False, font=dict(size=6, color='rgba(0, 255, 100, 0.4)')),
                dict(x=0, y=0.45, text='BALANCED', showarrow=False, font=dict(size=6, color='rgba(255, 255, 0, 0.4)')),
                dict(x=0, y=0.72, text='EXPLORE', showarrow=False, font=dict(size=6, color='rgba(255, 150, 0, 0.4)')),
                dict(x=0.5, y=-0.12, xref='paper', yref='paper',
                     text=f'<b>B={beta:.1f}</b> {rw_status} | K={TELEMETRY_STATE["curvature"]:.1f} | S={TELEMETRY_STATE["entropy"]:.2f}',
                     showarrow=False, font=dict(size=9, color=rw_color))
            ]
        )

        return (
            f"{TELEMETRY_STATE['curvature']}",
            f"{TELEMETRY_STATE['entropy']}",
            f"{TELEMETRY_STATE['active_fiber']}",
            f"{TELEMETRY_STATE['constraint_score']:.2f}",
            "\n".join(TELEMETRY_STATE["thought_trace"][-8:]),
            fig,
            history_fig
        )
    except Exception as e:
        print(f"Telemetry Poll Error: {e}")
        return ("0.0", "0.0", "Error", "0.0", f"Error: {e}", go.Figure(), go.Figure())

# --- UI BUILD ---
css_path = os.path.join(os.path.dirname(__file__), "theme_neural_glass.css")
if not os.path.exists(css_path):
    css_path = "igbundle-llm/theme_neural_glass.css"

css = open(css_path).read()

with gr.Blocks(theme=gr.themes.Base(), css=css, title="NEURAL GLASS") as app:

    start_btn = gr.Button("Initialize System", visible=False)
    start_btn.click(load_models, None, None)

    with gr.Row(elem_classes="glass-panel"):
        gr.Markdown("# <span class='neon-header'>IGBUNDLE // NEURAL GLASS</span>")

    with gr.Row():
        # SIDEBAR
        with gr.Column(scale=1, elem_classes="glass-panel"):
            gr.Markdown("### SYSTEM STATUS")

            with gr.Group(elem_classes="info-card"):
                gr.Markdown("**MODEL IDENTITY**\nUnified Manifold v2.1\nQwen 7B (NF4) + SigLIP")

            with gr.Group(elem_classes="info-card"):
                k_label = gr.Label(label="SECTIONAL CURVATURE (K)", value="0.0")
                s_label = gr.Label(label="MANIFOLD ENTROPY (S)", value="0.0")
                constraint_label = gr.Label(label="CONSTRAINT SCORE", value="1.0")
                fiber_label = gr.Label(label="ACTIVE BUNDLE", value="Standby")

                manifold_plot = gr.Plot(label="Poincare Manifold")
                metrics_plot = gr.Plot(label="Metrics History (K, S)")

            with gr.Group(elem_classes="info-card"):
                gr.Markdown("**CONTINUOUS MODE**")
                continuous_toggle = gr.Checkbox(
                    label="Continuous Run", value=False,
                    info="Auto-continue with memory consolidation"
                )
                continuous_turns = gr.Slider(
                    label="Max Turns", minimum=2, maximum=100,
                    value=50, step=1, visible=True
                )


        # MAIN STAGE
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_classes=["glass-panel"],
                height=400,
                show_label=False
            )

            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False,
                    placeholder="Enter query or drag image...",
                    container=False,
                    scale=4
                )
                img_input = gr.Image(type="filepath", container=False, scale=1, height=50)
                send_btn = gr.Button("TRANSMIT", variant="primary", scale=1, elem_classes="neon-btn")

            gr.Markdown("### THOUGHT MANIFOLD TRACE")
            thought_log = gr.TextArea(
                label="Internal Monologue",
                lines=5, max_lines=5,
                elem_classes="thought-stream",
                interactive=False
            )

    # --- CONTINUOUS MODE HANDLERS ---
    def toggle_continuous(enabled, max_t):
        CONTINUOUS_RUN["enabled"] = enabled
        CONTINUOUS_RUN["max_turns"] = int(max_t)
        CONTINUOUS_RUN["turn_count"] = 0
        print(f"[Continuous] Mode {'ACTIVE' if enabled else 'OFF'} (max_turns={max_t})")
        return enabled

    continuous_toggle.change(
        toggle_continuous,
        [continuous_toggle, continuous_turns],
        [continuous_toggle]
    )

    # EVENT LOOP
    def user_turn(user_message, history, image):
        print(f"DEBUG: user_turn input type={type(user_message)} val={repr(user_message)}")

        val = user_message
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
             if 'text' in val[0]:
                 val = val[0]['text']
        if isinstance(val, dict) and 'text' in val:
             val = val['text']

        if not val or not str(val).strip():
            if history is None: history = []
            return "", history, image

        if history is None: history = []
        history.append({"role": "user", "content": str(val)})
        return "", history, image

    def bot_turn(history, image):
        print("DEBUG: Entered bot_turn")

        if not history:
            yield history
            return

        if not history[-1] or not isinstance(history[-1], dict) or "content" not in history[-1]:
             yield history
             return

        user_message = history[-1]["content"] if history else ""
        if not user_message:
             yield history
             return

        # --- SALVO INPUT PREPROCESSOR ---
        # Multi-question inputs: let the model see all questions but add structure.
        # Truncate only if truly massive (>4000 chars) to prevent OOM.
        if len(user_message) > 4000:
            user_message = user_message[:4000]
            history[-1]["content"] = user_message
            print(f"DEBUG: SALVO PREPROCESSOR — truncated to 4000 chars")
            TELEMETRY_STATE["thought_trace"].append("SALVO: truncated to 4000 chars")

        # Apply rolling window BEFORE generation to cap context
        history = _make_rolling_context(history, MAX_ROLLING_MESSAGES)

        history.append({"role": "assistant", "content": ""})

        bot_message = ""
        conversation_context = history[:-1]

        MAX_HEAL_RETRIES = 2
        _heal_attempt = 0
        TELEMETRY_STATE["_generation_corrupted"] = None  # Reset
        TELEMETRY_STATE.pop("_generation_stalled", None)  # Reset stall signal

        print(f"DEBUG: Calling generate_stream with {len(conversation_context)} msgs")
        try:
            found_yield = False
            for partial in generate_stream(conversation_context, image, MAX_GENERATION_TOKENS):
                found_yield = True
                bot_message = partial
                history[-1]["content"] = bot_message
                yield history

            if not found_yield:
                 history[-1]["content"] = "**Error: Generation yielded nothing.**"
                 yield history

        except Exception as e:
            print(f"ERROR in bot_turn: {e}")
            history[-1]["content"] = f"**System Error**: {e}"
            yield history

        # --- SELF-HEALING LOOP ---
        # If generate_stream detected corruption (identity drift, repetition, prompt leakage),
        # retry with corrective injection and cooled temperature.
        corruption = TELEMETRY_STATE.get("_generation_corrupted")
        while corruption and _heal_attempt < MAX_HEAL_RETRIES:
            _heal_attempt += 1
            clean_text = bot_message  # Already truncated to clean prefix by generate_stream

            # Build healing directive based on corruption type — keep SHORT for 7B
            if corruption == "echo":
                heal_directive = "Answer the question. Do not repeat it."
            elif corruption == "hallucinated_conversation":
                heal_directive = "Answer the question only. No dialogue."
            elif corruption == "identity_drift":
                heal_directive = "Answer the question directly."
            elif corruption == "token_artifact":
                heal_directive = "Answer clearly."
            elif corruption == "prompt_leakage":
                heal_directive = "Answer the question."
            else:  # repetition
                heal_directive = "Continue without repeating."

            print(f"DEBUG: SELF-HEAL attempt {_heal_attempt}/{MAX_HEAL_RETRIES} [{corruption}]")
            # Cool temperature for retry (deterministic recovery)
            # Echo + hallucinated conversations need even colder temperature
            heal_temp = 0.10 if corruption in ("hallucinated_conversation", "echo") else 0.15

            TELEMETRY_STATE["thought_trace"].append(
                f"SELF-HEAL attempt {_heal_attempt}: {corruption} -> retry T={heal_temp}"
            )
            TELEMETRY_STATE["_heal_temp_override"] = heal_temp
            TELEMETRY_STATE["_generation_corrupted"] = None

            # Build healing context based on corruption type
            if corruption == "echo": # Disabled hallucinated_conversation nuclear option so valid text isn't dropped
                # NUCLEAR OPTION: Strip ALL corrupted output, minimal context.
                # Only system prompt + healing directive + original user question.
                # No partial output — the entire output was fake conversation.
                # Extract original user question
                original_question = ""
                for msg in reversed(conversation_context):
                    if msg.get("role") == "user":
                        original_question = msg["content"]
                        break

                # For echo/hallucinated: if question is very long (multi-salvo),
                # extract just the FIRST question to give the model a chance
                if len(original_question) > 300:
                    # Find first question mark or take first 300 chars
                    first_q = original_question.find("?")
                    if first_q > 0:
                        original_question = original_question[:first_q + 1]
                    else:
                        original_question = original_question[:300]
                    print(f"DEBUG: Heal — trimmed question to {len(original_question)} chars")

                # Extract system prompt — strip injected memory to prevent re-echo
                sys_msg = ""
                for msg in conversation_context:
                    if msg.get("role") == "system":
                        sys_msg = msg["content"]
                        break
                # Strip [User Memory] block that was injected into system prompt
                mem_marker = "\n\n[User Memory]"
                if mem_marker in sys_msg:
                    sys_msg = sys_msg[:sys_msg.index(mem_marker)]

                heal_context = [
                    {"role": "system", "content": (
                        (sys_msg + "\n\n" if sys_msg else "") +
                        f"[CORRECTION: {heal_directive}]"
                    )},
                    {"role": "user", "content": original_question},
                ]
                # Discard ALL previous output
                clean_text = ""
            else:
                # Standard healing: strip greeting from clean_text first
                clean_text = _strip_greeting_prefix(clean_text) if clean_text else ""

                # Extract original user question for re-asking
                original_question = ""
                for msg in reversed(conversation_context):
                    if msg.get("role") == "user":
                        original_question = msg["content"]
                        break

                # Check if clean_text has real substance (not just greeting boilerplate)
                has_substance = clean_text and len(clean_text.strip()) > 50

                if has_substance:
                    # Minimal context: system + directive + user question + clean prefix
                    # (keeping full conversation_context caused VRAM exhaustion on heal retry)
                    sys_msg = ""
                    for msg in conversation_context:
                        if msg.get("role") == "system":
                            sys_msg = msg["content"]
                            break
                    # Trim question for VRAM safety
                    heal_question = original_question
                    if len(heal_question) > 500:
                        first_q = heal_question.find("?")
                        if first_q > 0 and first_q < 500:
                            heal_question = heal_question[:first_q + 1]
                        else:
                            heal_question = heal_question[:500]
                    heal_context = [
                        {"role": "system", "content": (
                            (sys_msg + "\n\n" if sys_msg else "") +
                            f"[CORRECTION: {heal_directive}]"
                        )},
                        {"role": "user", "content": heal_question},
                    ]
                    # Include clean prefix only if short enough to not eat VRAM
                    if len(clean_text) < 500:
                        heal_context.append({
                            "role": "assistant",
                            "content": clean_text
                        })
                else:
                    # Clean text is too short / just greeting — fresh re-ask with NUCLEAR context
                    clean_text = ""
                    sys_msg = ""
                    for msg in conversation_context:
                        if msg.get("role") == "system":
                            sys_msg = msg["content"]
                            break
                    # Strip memory from system prompt to save tokens
                    mem_marker = "\n\n[User Memory]"
                    if mem_marker in sys_msg:
                        sys_msg = sys_msg[:sys_msg.index(mem_marker)]
                    # Trim question for VRAM safety (same as has_substance path)
                    heal_question = original_question
                    if len(heal_question) > 500:
                        first_q = heal_question.find("?")
                        if first_q > 0 and first_q < 500:
                            heal_question = heal_question[:first_q + 1]
                        else:
                            heal_question = heal_question[:500]
                        print(f"DEBUG: Heal (no-substance) — trimmed question to {len(heal_question)} chars")
                    heal_context = [
                        {"role": "system", "content": (
                            (sys_msg + "\n\n" if sys_msg else "") +
                            f"[CORRECTION: {heal_directive}]"
                        )},
                        {"role": "user", "content": heal_question},
                    ]

            # VRAM cleanup before retry + skip memory retrieval (prevents echo re-injection)
            _cleanup_vram()
            TELEMETRY_STATE["_skip_memory_retrieval"] = True

            # Retry generation
            bot_message = clean_text  # Start from clean prefix (may be empty)
            history[-1]["content"] = bot_message if bot_message else "..."
            yield history

            try:
                for partial in generate_stream(heal_context, image, MAX_GENERATION_TOKENS):
                    if clean_text and len(clean_text.strip()) > 50:
                        # Append healed output to substantive clean prefix
                        bot_message = clean_text.rstrip() + "\n" + partial
                    else:
                        bot_message = partial
                    history[-1]["content"] = bot_message
                    yield history
            except Exception as e:
                print(f"SELF-HEAL attempt {_heal_attempt} error: {e}")
                break

            # Check if healing worked
            corruption = TELEMETRY_STATE.get("_generation_corrupted")
            if corruption:
                print(f"DEBUG: SELF-HEAL attempt {_heal_attempt} still corrupted [{corruption}]")
            else:
                print(f"DEBUG: SELF-HEAL attempt {_heal_attempt} succeeded")
                TELEMETRY_STATE["thought_trace"].append(
                    f"SELF-HEAL attempt {_heal_attempt}: RECOVERED"
                )

        # --- STALL RECOVERY ---
        # If generate_stream stalled (timeout with 0 tokens), the context was too large
        # for available VRAM. Retry with aggressively reduced context (system + user only).
        _stall_token = TELEMETRY_STATE.pop("_generation_stalled", None)
        if _stall_token is not None and _stall_token == 0:
            print("DEBUG: STALL RECOVERY — token-0 stall detected. Retrying with minimal context.")
            TELEMETRY_STATE["thought_trace"].append("STALL RECOVERY: trimming context to system+user only")
            _cleanup_vram()

            # Extract just the user question (first question if multi-question)
            original_question = user_message
            if len(original_question) > 500:
                first_q = original_question.find("?")
                if first_q > 0 and first_q < 500:
                    original_question = original_question[:first_q + 1]
                else:
                    original_question = original_question[:500]
                print(f"DEBUG: STALL RECOVERY — trimmed question to {len(original_question)} chars")

            # Minimal context: system + single question only
            sys_content = MODEL_STATE.state.get("system_prompt",
                "Answer directly with step-by-step reasoning. Be concise."
            )
            minimal_context = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": original_question},
            ]

            TELEMETRY_STATE["_skip_memory_retrieval"] = True
            TELEMETRY_STATE["_heal_temp_override"] = 0.15

            history[-1]["content"] = "..."
            yield history

            try:
                for partial in generate_stream(minimal_context, None, min(MAX_GENERATION_TOKENS, 768)):
                    bot_message = partial
                    history[-1]["content"] = bot_message
                    yield history
                print("DEBUG: STALL RECOVERY succeeded")
                TELEMETRY_STATE["thought_trace"].append("STALL RECOVERY: succeeded")
            except Exception as e:
                print(f"DEBUG: STALL RECOVERY failed: {e}")
                history[-1]["content"] = "**[VRAM exhausted — please restart Neural Glass or try a shorter prompt.]**"
                yield history

        # Store bot response in memory (significant for retrieval quality)
        if bot_message and not bot_message.startswith("**["):
            _mem = MODELS.get("memory")
            if _mem:
                try:
                    _mem.store(bot_message[:1000], tier="auto")
                except Exception:
                    pass

        # VRAM cleanup after every turn
        _cleanup_vram()

        # --- CONTINUOUS RUN MODE ---
        if CONTINUOUS_RUN["enabled"] and CONTINUOUS_RUN["turn_count"] < CONTINUOUS_RUN["max_turns"]:
            CONTINUOUS_RUN["turn_count"] += 1
            turn = CONTINUOUS_RUN["turn_count"]
            print(f"[Continuous] Turn {turn}/{CONTINUOUS_RUN['max_turns']}")

            # Memory consolidation every N turns — geometric-tagged
            if turn % CONTINUOUS_RUN["consolidate_every"] == 0:
                _mem = MODELS.get("memory")
                if _mem and bot_message:
                    try:
                        # Tag consolidated memory with geometric state
                        # This preserves the manifold position info for later retrieval weighting
                        k = TELEMETRY_STATE.get("curvature", 0.0)
                        s = TELEMETRY_STATE.get("entropy", 0.0)
                        fiber = TELEMETRY_STATE.get("active_fiber", "?")
                        trace = TELEMETRY_STATE.get("manifold_trace", [])
                        r = 0.0
                        if trace:
                            pt = trace[-1]
                            r = float(np.sqrt(pt[0]**2 + pt[1]**2))

                        geo_tag = f"[K={k:.1f} S={s:.2f} r={r:.2f} {fiber}] "
                        # Store with geometric prefix — memories created near boundary
                        # will have higher |K| and r, signaling instability
                        _mem.store(geo_tag + bot_message[:1500], tier="episodic")
                        print(f"[Continuous] Memory consolidated at turn {turn} (r={r:.2f}, K={k:.1f})")
                        TELEMETRY_STATE["thought_trace"].append(
                            f"[Continuous] Consolidated (turn {turn}, r={r:.2f})"
                        )
                    except Exception as e:
                        print(f"[Continuous] Memory consolidation error: {e}")

                try:
                    MODEL_STATE.snapshot_telemetry(TELEMETRY_STATE)
                except Exception:
                    pass

            # Trim history for long-horizon: rolling window
            history = _make_rolling_context(history, MAX_ROLLING_MESSAGES)

            # Auto-inject geometry-aware continuation prompt
            cont_prompt = _select_continuation_prompt()
            history.append({"role": "user", "content": cont_prompt})
            yield history

            # Generate continuation
            history.append({"role": "assistant", "content": ""})
            conversation_context = history[:-1]
            bot_message = ""
            try:
                for partial in generate_stream(conversation_context, image, MAX_GENERATION_TOKENS):
                    bot_message = partial
                    history[-1]["content"] = bot_message
                    yield history
            except Exception as e:
                print(f"[Continuous] Error at turn {turn}: {e}")
                history[-1]["content"] = f"**Continuous mode error**: {e}"
                yield history
                CONTINUOUS_RUN["turn_count"] = CONTINUOUS_RUN["max_turns"]

            # Short response = model signaled completion
            if bot_message and len(bot_message.strip()) < 50:
                print(f"[Continuous] Short response detected, stopping.")
                CONTINUOUS_RUN["turn_count"] = CONTINUOUS_RUN["max_turns"]

            # VRAM cleanup after continuous turn
            _cleanup_vram()

        elif CONTINUOUS_RUN["enabled"] and CONTINUOUS_RUN["turn_count"] >= CONTINUOUS_RUN["max_turns"]:
            CONTINUOUS_RUN["turn_count"] = 0
            CONTINUOUS_RUN["enabled"] = False
            history.append({"role": "assistant", "content": f"*[Continuous mode complete -- {CONTINUOUS_RUN['max_turns']} turns]*"})
            yield history

    txt_input.submit(user_turn, [txt_input, chatbot, img_input], [txt_input, chatbot, img_input]).then(
        bot_turn, [chatbot, img_input], [chatbot]
    )
    send_btn.click(user_turn, [txt_input, chatbot, img_input], [txt_input, chatbot, img_input]).then(
        bot_turn, [chatbot, img_input], [chatbot]
    )

    # Telemetry Timer
    timer = gr.Timer(0.1)
    timer.tick(poll_telemetry, None, [k_label, s_label, fiber_label, constraint_label, thought_log, manifold_plot, metrics_plot])

if __name__ == "__main__":
    print("Initializing Models...")
    load_models()

    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        show_error=True
    )
