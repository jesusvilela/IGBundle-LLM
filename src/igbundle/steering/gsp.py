"""
Geometric Steering Probe (GSP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A GENERIC-compliant, homeostatic, backpropagable inference-time
mechanism to restore entropic diversity when Bundle lock is detected.

Mathematical basis:
  - GENERIC framework: dz/dt = L·∂E/∂z + M·∂S/∂z
  - Bundle-6 lock ≡ M(z) → 0 (dissipation suppressed)
  - GSP restores M via: M_deice = λ(S) · P_⊥ (energy-orthogonal)
  - P_⊥ direction: Log_x(mem(0)) (geodesic toward Poincaré origin)
  - Homeostatic: λ(S) → 0 as S → S_target (self-limiting)
  - PID control in hyperbolic space for over/undershoot prevention

No base model weights are modified. Only ~20 scalar parameters (θ_GSP)
are learned/evolved, operating as forward hooks on the residual stream.

Genetically improvable via CMA-ES on θ_GSP using benchmark fitness.
Backpropagable via the closed-form Jacobian of the Poincaré log map.

Author: IGBundle Research
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Iterable, Optional, List, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────
# POINCARÉ BALL OPERATIONS (hyperbolic geometry primitives)
# ─────────────────────────────────────────────────────────────────────

def poincare_log_map(x: torch.Tensor, y: Optional[torch.Tensor], c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map Log_x(y) in the Poincaré ball with curvature -c.
    Returns the tangent vector at x pointing toward y.
    
    Special case y=None or y=0 (mem(0) = origin):
      Log_x(0) = -(2/√c) · arctanh(√c · ||x||) · x/||x||
               = DEICING DIRECTION: geodesic from x toward origin
    """
    sqrt_c = c ** 0.5
    
    if y is None or (y.norm() < 1e-7):
        # Toward Poincaré origin
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        arg = (sqrt_c * x_norm).clamp(max=1.0 - 1e-7)
        scale = -(2.0 / sqrt_c) * torch.arctanh(arg)
        return scale * x / x_norm
    
    # General case via Möbius addition -x ⊕ y
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy   = (x * y).sum(dim=-1, keepdim=True)
    num  = (1 + 2*c*xy + c*y_sq) * x + (1 - c*x_sq) * y
    den  = (1 + 2*c*xy + c**2 * x_sq * y_sq).clamp(min=1e-8)
    v    = -num / den
    
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    arg = (sqrt_c * v_norm).clamp(max=1.0 - 1e-7)
    scale = (2.0 / sqrt_c) * torch.arctanh(arg)
    return scale * v / v_norm


def poincare_project(h: torch.Tensor, max_radius: float = 0.9) -> torch.Tensor:
    """Project Euclidean hidden state h into Poincaré ball via norm-based scaling.

    Maps ||h|| → [0, max_radius) using adaptive normalization.
    Qwen layer 12 hidden states have norm ~100-200, so fixed tanh(x*0.3)
    saturates to 1.0 for everything. Instead: r = max_radius * ||h|| / (||h|| + 1).
    This maps 0→0, ∞→max_radius, with meaningful interior at typical norms.
    """
    h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r = max_radius * h_norm / (h_norm + 1.0)
    return r * h / h_norm


# ─────────────────────────────────────────────────────────────────────
# THE GSP GENOME — EVOLVABLE PARAMETER VECTOR (~20 scalars)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GSPGenome:
    """
    The evolvable genome for the GSP.
    No model weights — only control law parameters.
    Optimizable by CMA-ES (black-box) or backprop (gradient).
    """
    # PID gains (homeostatic control in entropy space)
    K_P: float = 0.05        # Proportional to current S deficit
    K_I: float = 0.01        # Integral: breaks chronic Bundle lock
    K_D: float = 0.005       # Derivative: prevents entropy oscillation

    S_target: float = 1.20   # Target entropy (above ~0.75 observed lock band)
    tau_lock:  float = 0.60  # Constraint threshold: GSP activates above this
    tau_bundle: int  = 6     # Bundle index that triggers deicing

    # Per-layer weights α_l for layers 8–20 (13 values)
    layer_weights: List[float] = field(default_factory=lambda: [
        0.00,  # layer  8
        0.10,  # layer  9
        0.20,  # layer 10
        0.40,  # layer 11
        0.80,  # layer 12  ← trained adapter here, strongest effect
        0.60,  # layer 13
        0.50,  # layer 14
        0.40,  # layer 15
        0.30,  # layer 16
        0.20,  # layer 17
        0.15,  # layer 18
        0.10,  # layer 19
        0.05,  # layer 20
    ])

    curvature: float = 1.0             # Poincaré ball curvature |K|
    mem0_semantic_alpha: float = 0.30  # 0=pure origin, 1=pure semantic attractor
    lambda_max: float = 0.10           # Safety cap on deicing strength

    def to_vector(self) -> np.ndarray:
        v = [self.K_P, self.K_I, self.K_D, self.S_target,
             self.tau_lock, self.mem0_semantic_alpha, self.lambda_max]
        return np.array(v + self.layer_weights, dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'GSPGenome':
        g = cls()
        g.K_P, g.K_I, g.K_D = float(v[0]), float(v[1]), float(v[2])
        g.S_target           = float(v[3])
        g.tau_lock           = float(v[4])
        g.mem0_semantic_alpha = float(v[5])
        g.lambda_max         = float(v[6])
        g.layer_weights      = [float(x) for x in v[7:20]]
        return g


# ─────────────────────────────────────────────────────────────────────
# THE GSP CONTROLLER
# ─────────────────────────────────────────────────────────────────────

class GeometricSteeringProbe(nn.Module):
    """
    Inference-time homeostatic controller.
    
    Attaches as forward hooks to transformer layers 8–20.
    Reads K, S, Constraint, Bundle from Neural Glass state.
    Applies GENERIC-compliant, energy-preserving perturbations δh.
    
    Non-destructive: base weights untouched.
    Self-limiting: δh → 0 when S → S_target.
    Backpropagable: via D_Log(x) · J_proj(h).
    Genetically evolvable: only θ_GSP is optimized.
    """

    def __init__(self, genome: Optional[GSPGenome] = None,
                 mem0_attractor: Optional[torch.Tensor] = None,
                 hidden_dim: int = 3584):
        super().__init__()
        self.genome = genome or GSPGenome()
        self.hidden_dim = hidden_dim

        self.register_buffer('_mem0_base',     torch.zeros(hidden_dim))
        self.register_buffer('_mem0_semantic', torch.zeros(hidden_dim))
        if mem0_attractor is not None:
            n = mem0_attractor / mem0_attractor.norm().clamp(min=1e-8)
            self._mem0_semantic = (0.4 * n).to(torch.float32)

        # PID state
        self._integral_error: float = 0.0
        self._prev_error:     float = 0.0
        self._cached_lambda:  float = 0.0   # computed once per step, used by all layers
        self._step_id:        int   = 0     # incremented by step_begin()
        self._layer_weight_index: Dict[int, int] = {}
        self.last_attached_layers: List[int] = []

        # Live geometric state (updated by Neural Glass bridge)
        # AUDIT NOTE (2026-07): K_current defaults to the architecture-constant
        # value of the hardcoded conformal factor at the projection boundary
        # (see draft_paper_falsification.md §2.6). It is NOT evidence of learned
        # hyperbolicity. If the bridge does not call update_geometric_state() with
        # a value from the corrected estimator under learnable_conformal=True,
        # this default remains an unfalsifiable constant — do not quote it as a
        # measurement of the model's geometry.
        self.K_current:          float = -5.88
        self.S_current:          float = 0.80
        self.Constraint_current: float = 0.80
        self.Bundle_current:     int   = 6

        self._hooks: list = []
        self._active: bool = False

    @property
    def mem0(self) -> torch.Tensor:
        α = self.genome.mem0_semantic_alpha
        return (1 - α) * self._mem0_base + α * self._mem0_semantic

    # ── state updates ──────────────────────────────────────────────

    def update_geometric_state(self, K: float, S: float,
                                Constraint: float, Bundle: int):
        self.K_current          = K
        self.S_current          = S
        self.Constraint_current = Constraint
        self.Bundle_current     = Bundle

    def update_semantic_attractor(self, embedding: torch.Tensor):
        """Called when memory recall fires (e.g. 'explanatory_depth')."""
        n = embedding / embedding.norm().clamp(min=1e-8)
        self._mem0_semantic = (0.4 * n).to(self._mem0_base.device)

    # ── core logic ─────────────────────────────────────────────────

    def _should_activate(self) -> bool:
        return (
            self.Constraint_current >= self.genome.tau_lock
            and self.Bundle_current == self.genome.tau_bundle
            and self.S_current < self.genome.S_target
        )

    def _compute_pid_gain(self) -> float:
        """Compute PID gain. Called ONCE per generation step by step_begin()."""
        e  = self.genome.S_target - self.S_current
        de = e - self._prev_error
        self._integral_error = float(np.clip(
            self._integral_error + e, -2.0, 2.0))   # anti-windup
        λ = (self.genome.K_P * e
           + self.genome.K_I * self._integral_error
           + self.genome.K_D * de)
        self._prev_error = e
        return float(np.clip(λ, 0.0, self.genome.lambda_max))

    def step_begin(self):
        """Call ONCE before each model.generate(). Computes PID gain for all layers.

        This prevents the PID integral/derivative from accumulating 13x per
        forward pass (once per hooked layer). The cached λ is used by all hooks.
        """
        self._step_id += 1
        if self._should_activate():
            self._cached_lambda = self._compute_pid_gain()
        else:
            self._cached_lambda = 0.0

    def _delta_h(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute GENERIC-compliant deicing perturbation δh.

        δh = λ · α_l · Log_x(mem(0))  [broadcast to full h shape]

        Approximately orthogonal to ∂E/∂h in Riemannian metric → K preserved.
        Self-limiting via PID gain λ → 0 as S → S_target.
        λ is pre-computed once per step by step_begin() — shared across layers.
        """
        λ = self._cached_lambda
        if λ < 1e-8:
            return torch.zeros_like(h)

        offset = self._layer_weight_index.get(layer_idx, layer_idx - 8)
        if offset < 0 or offset >= len(self.genome.layer_weights):
            return torch.zeros_like(h)
        α_l = self.genome.layer_weights[offset]
        if α_l < 1e-6:
            return torch.zeros_like(h)

        # Project mean hidden state to Poincaré ball
        h_in = h.mean(dim=1) if h.dim() == 3 else h   # [B, d]
        x = poincare_project(h_in.float())              # [B, d]

        # mem(0): semantic-biased origin
        m = self.mem0.to(h.device)
        m_ball = poincare_project(m.unsqueeze(0)) if m.norm() > 1e-6 else None

        # Deicing direction: Log_x(mem0)
        direction = poincare_log_map(x, m_ball, self.genome.curvature)  # [B, d]

        # Unit-normalise, scale, broadcast
        d_norm  = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_2d = (λ * α_l) * d_norm                  # [B, d]

        if h.dim() == 3:
            return delta_2d.unsqueeze(1).to(h.dtype)    # [B, 1, d]
        return delta_2d.to(h.dtype)

    # ── hook management ────────────────────────────────────────────

    def make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if not self._active:
                return output
            h = output[0] if isinstance(output, tuple) else output
            delta = self._delta_h(h, layer_idx)
            h_new = h + delta
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
        return hook_fn

    @staticmethod
    def _find_layers(model):
        """Locate transformer layers across Qwen/Gemma/PeftModel variants."""
        for path in [
            lambda m: m.model.language_model.layers,        # Gemma4ForConditionalGeneration
            lambda m: m.model.layers,                       # vanilla Qwen/Gemma
            lambda m: m.model.model.layers,                 # PeftModel wrapping
            lambda m: m.base_model.model.model.layers,      # double-wrapped
        ]:
            try:
                layers = path(model)
                if layers is not None and len(layers) > 0:
                    return layers
            except (AttributeError, IndexError):
                continue
        return None

    def attach(self, model, target_layers: Iterable[int] = range(8, 21)) -> List[int]:
        self.detach()
        layers = self._find_layers(model)
        if layers is None:
            print("[GSP] ERROR: Could not locate transformer layers. Hook injection failed.")
            self.last_attached_layers = []
            return []
        attached = []
        requested = [int(idx) for idx in target_layers]
        for idx in requested:
            try:
                layer = layers[idx]
                handle = layer.register_forward_hook(self.make_hook(idx))
                self._hooks.append(handle)
                attached.append(idx)
            except (AttributeError, IndexError):
                continue
        if attached:
            weight_count = len(self.genome.layer_weights)
            if len(attached) == 1:
                self._layer_weight_index = {attached[0]: min(weight_count // 2, weight_count - 1)}
            else:
                self._layer_weight_index = {
                    layer_idx: int(round(pos * (weight_count - 1) / max(len(attached) - 1, 1)))
                    for pos, layer_idx in enumerate(attached)
                }
        else:
            self._layer_weight_index = {}
        self.last_attached_layers = attached
        self._active = len(self._hooks) > 0
        print(f"[GSP] Attached to {len(attached)} layers {attached}, "
              f"S_target={self.genome.S_target:.2f}, "
              f"tau_lock={self.genome.tau_lock:.2f}, "
              f"lambda_max={self.genome.lambda_max:.3f}")
        if requested and len(attached) < len(requested):
            print(f"[GSP] Skipped {len(requested) - len(attached)} unavailable target layers.")
        return attached

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._layer_weight_index = {}
        self.last_attached_layers = []
        self._active = False
        self.reset_pid()

    def reset_pid(self):
        self._integral_error = 0.0
        self._prev_error     = 0.0


# ─────────────────────────────────────────────────────────────────────
# NEURAL GLASS BRIDGE
# ─────────────────────────────────────────────────────────────────────

class NeuralGlassBridge:
    """
    Reads Neural Glass telemetry and pushes K, S, Constraint, Bundle
    into the GSP controller before each generation step.
    Also handles memory-recall → mem(0) updates.

    Accepts either:
      - telemetry_dict: live reference to TELEMETRY_STATE (no I/O)
      - state_file: JSON path (fallback for external use)
    """

    def __init__(self, gsp: GeometricSteeringProbe,
                 telemetry_dict: Optional[Dict[str, Any]] = None,
                 state_file: Optional[str] = None):
        self.gsp = gsp
        self.telemetry_dict = telemetry_dict
        self.state_file = state_file

    def poll_and_update(self):
        """Read telemetry → update GSP state → call step_begin()."""
        s = self.telemetry_dict
        if s is None and self.state_file:
            try:
                import json
                with open(self.state_file) as f:
                    s = json.load(f)
            except Exception:
                pass
        if s is None:
            return

        # Parse active_fiber string like "Bundle-6" → int 6
        bundle_raw = s.get('active_fiber', 'Bundle-6')
        try:
            bundle_idx = int(str(bundle_raw).replace('Bundle-', '').replace('Standby', '0'))
        except (ValueError, TypeError):
            bundle_idx = 6

        self.gsp.update_geometric_state(
            K          = float(s.get('curvature',        -5.88)),
            S          = float(s.get('entropy',           0.80)),
            Constraint = float(s.get('constraint_score',  0.80)),
            Bundle     = bundle_idx,
        )
        self.gsp.step_begin()

    def on_memory_recall(self, labels: List[str],
                         embeddings: Dict[str, torch.Tensor]):
        """
        Connect ChromaDB/SimpleMemory attractor recall to mem(0).
        Formalises the cold-start observation where 'explanatory_depth'
        improved S3-Q1 DiffGeo answer quality.
        """
        vecs = [embeddings[l] for l in labels if l in embeddings]
        if vecs:
            combined = torch.stack(vecs).mean(0)
            self.gsp.update_semantic_attractor(combined)


# ─────────────────────────────────────────────────────────────────────
# GENETIC EVOLUTION ENGINE (CMA-ES, no model gradients needed)
# ─────────────────────────────────────────────────────────────────────

class GSPEvolver:
    """
    Black-box optimiser for θ_GSP using benchmark fitness.
    Uses CMA-ES if available, falls back to simple (1+λ)-ES.
    Base model weights are NEVER touched.
    """

    def __init__(self, gsp: GeometricSteeringProbe,
                 benchmark_fn,           # callable: GSPGenome → float
                 sigma0: float = 0.1,
                 popsize: int  = 10):
        self.gsp          = gsp
        self.benchmark_fn = benchmark_fn
        self.sigma0       = sigma0
        self.popsize      = popsize

    def run(self, n_generations: int = 50) -> GSPGenome:
        try:
            import cma
            x0 = self.gsp.genome.to_vector()
            es = cma.CMAEvolutionStrategy(
                x0, self.sigma0,
                {'maxiter': n_generations, 'popsize': self.popsize})
            while not es.stop():
                sols = es.ask()
                fits = [-self.benchmark_fn(GSPGenome.from_vector(np.array(v)))
                        for v in sols]
                es.tell(sols, fits)
                es.disp()
            return GSPGenome.from_vector(np.array(es.result.xbest))
        except ImportError:
            return self._simple_es(n_generations)

    def _simple_es(self, n_generations: int) -> GSPGenome:
        best = self.gsp.genome
        best_f = self.benchmark_fn(best)
        σ = self.sigma0

        for gen in range(n_generations):
            x0 = best.to_vector()
            cands = [GSPGenome.from_vector(
                         np.clip(x0 + np.random.randn(len(x0)) * σ, 0, 1))
                     for _ in range(self.popsize)]
            for g in cands:
                f = self.benchmark_fn(g)
                if f > best_f:
                    best_f, best = f, g
            # 1/5 success rule
            n_ok = sum(1 for g in cands if self.benchmark_fn(g) > best_f * 0.99)
            σ = np.clip(σ * (1.2 if n_ok/self.popsize > 0.2 else 0.8), 1e-5, 0.5)
            print(f"[GSP-ES] gen {gen+1}/{n_generations}  F={best_f:.4f}  σ={σ:.4f}")

        return best


# ─────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────

def create_gsp_for_qwen7b(
    telemetry_dict: Optional[Dict[str, Any]] = None,
    state_file: Optional[str] = None,
    genome: Optional[GSPGenome] = None
) -> Tuple[GeometricSteeringProbe, NeuralGlassBridge]:
    """Legacy factory for Qwen2.5-7B. Use create_gsp() instead."""
    return create_gsp(hidden_dim=3584, telemetry_dict=telemetry_dict,
                       state_file=state_file, genome=genome)


def create_gsp(
    hidden_dim: int = 2560,
    telemetry_dict: Optional[Dict[str, Any]] = None,
    state_file: Optional[str] = None,
    genome: Optional[GSPGenome] = None
) -> Tuple[GeometricSteeringProbe, NeuralGlassBridge]:
    """
    Model-agnostic GSP factory.

    Args:
        hidden_dim: Base model hidden size (2560 for Gemma4-E4B, 3584 for Qwen7B).

    Returns (gsp, bridge) ready to attach.

    Typical usage:
        gsp, bridge = create_gsp(hidden_dim=2560, telemetry_dict=TELEMETRY_STATE)
        gsp.attach(model, range(8, 21))
        bridge.poll_and_update()
        output = model.generate(...)
    """
    g = genome or GSPGenome()
    gsp    = GeometricSteeringProbe(genome=g, hidden_dim=hidden_dim)
    bridge = NeuralGlassBridge(gsp, telemetry_dict=telemetry_dict, state_file=state_file)
    return gsp, bridge
