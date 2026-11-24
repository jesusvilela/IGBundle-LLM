"""
NMEM — Neuromorphic Memory Algorithm
=====================================
Biologically-inspired memory system for Neural Glass.

Architecture:
  Layer 0: Multi-anchor encoding at turn boundaries
  Layer 1: FitzHugh-Nagumo decay + FFT spectral forgetting + Gabor interference
  Layer 2: Threshold-band retrieval (full / reconstruct / ghost)
  Layer 3: HSAM reconstruction loop
  Layer 4: Topological context assembly

Design principles (from human memory outliers):
  - Retrieval pathway redundancy (Method of Loci)
  - Interference-based forgetting, not erasure (retroactive interference)
  - Frequency-selective detail loss before gist loss (FFT decay)
  - Reconsolidation on retrieval (HSAM re-rehearsal)
  - Savant-mode verbatim preservation for flagged content
  - Ghost entries: know what was forgotten without confabulating

Drop-in compatible with IMDMemory API.
"""

import time
import math
import json
import hashlib
import threading
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

import numpy as np

logger = logging.getLogger("nmem")


# ─── Configuration ─────────────────────────────────────────────────

@dataclass
class NMEMConfig:
    """All tunable parameters for the NMEM system."""

    # --- FHN dynamics ---
    fhn_epsilon: float = 0.08       # Timescale separation (slow forgetting var)
    fhn_a: float = 0.7              # FHN parameter a
    fhn_b: float = 0.8              # FHN parameter b
    fhn_I_rest: float = 0.0         # Resting input current (no external drive)
    fhn_dt: float = 0.05            # Integration timestep per tick
    fhn_ticks_per_hour: float = 20  # How many FHN ticks per wall-clock hour

    # --- FFT spectral decay ---
    alpha: float = 0.02             # Frequency-selective attenuation slope
                                     # Higher = faster detail loss, gist preserved

    # --- Gabor interference ---
    gabor_sigma: float = 3600.0     # Temporal width (seconds) for interference
    gabor_omega: float = 2.0        # Oscillation frequency for interference pattern
    gabor_radius: float = 0.7       # Cosine similarity threshold for neighbors
    gabor_strength: float = 0.05    # Perturbation magnitude

    # --- Retrieval thresholds ---
    theta_high: float = 0.4         # Above → full payload returned
    theta_low: float = 0.15         # Below → ghost entry
                                     # Between → reconstruction triggered

    # --- Encoding ---
    embedding_dim: int = 128        # Dimension for hash-based embeddings
    savant_types: Tuple[str, ...] = ("code", "number", "name", "url", "path")
    salience_threshold: float = 0.1 # Below → not encoded (attentional gating)

    # --- Reconstruction ---
    reconstruction_perturbation: float = 0.02  # Reconsolidation drift magnitude
    max_reconstructions: int = 5    # Cap per node before it becomes ghost

    # --- Capacity ---
    max_nodes: int = 1024           # Hard cap on total memory nodes
    persist_dir: str = "./memory/nmem"

    # --- Context assembly ---
    token_budget: int = 600         # Default retrieval budget


# ─── Lightweight Embedding ─────────────────────────────────────────

class HashEmbedder:
    """
    Deterministic hash-based embedding. Zero model overhead.

    Maps text to a fixed-dimension vector via feature hashing (hashing trick).
    Not as rich as sentence-transformers but O(n) with no model load.
    Pluggable: replace with SentenceTransformer for production quality.

    The resulting vectors have meaningful frequency structure under FFT:
    low-index bins capture common n-grams (semantic gist),
    high-index bins capture rare n-grams (episodic detail).
    """

    def __init__(self, dim: int = 128, ngram_range: Tuple[int, int] = (1, 3)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _ngrams(self, text: str) -> List[str]:
        tokens = re.findall(r'\w+', text.lower())
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                grams.append(" ".join(tokens[i:i + n]))
        return grams

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for gram in self._ngrams(text):
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1 if (h // self.dim) % 2 == 0 else -1
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb)


# ─── FitzHugh-Nagumo Dynamics ──────────────────────────────────────

@dataclass
class FHNState:
    """
    Two-variable excitable system per memory node.

    v = retrieval activation (fast variable)
    w = forgetting pressure (slow recovery variable)

    dv/dt = v - v^3/3 - w + I
    dw/dt = epsilon * (v + a - b*w)

    Unretrieved memories relax to stable fixed point.
    Retrieved memories spike then reconsolidate — trajectory back
    to rest differs each time (reconsolidation-as-slight-corruption).
    """
    v: float = 0.5   # Initial activation (medium — just encoded)
    w: float = 0.0   # No forgetting pressure yet


def fhn_step(state: FHNState, I_ext: float, cfg: NMEMConfig) -> FHNState:
    """Advance FHN one integration step (Euler)."""
    v, w = state.v, state.w
    dv = v - (v ** 3) / 3.0 - w + I_ext
    dw = cfg.fhn_epsilon * (v + cfg.fhn_a - cfg.fhn_b * w)
    dt = cfg.fhn_dt
    new_v = v + dt * dv
    new_w = w + dt * dw
    # Clamp to prevent runaway
    new_v = max(-3.0, min(3.0, new_v))
    new_w = max(-3.0, min(3.0, new_w))
    return FHNState(v=new_v, w=new_w)


def fhn_advance(state: FHNState, delta_hours: float, cfg: NMEMConfig,
                retrieved: bool = False) -> FHNState:
    """
    Advance FHN forward by delta_hours of wall-clock time.

    If retrieved=True, inject a current pulse (spike + reconsolidate).
    Otherwise, system relaxes toward rest (forgetting).
    """
    n_ticks = max(1, int(delta_hours * cfg.fhn_ticks_per_hour))
    # Cap ticks to prevent long catch-up loops
    n_ticks = min(n_ticks, 500)

    I_ext = cfg.fhn_I_rest
    if retrieved:
        # Retrieval pulse — spike the activation
        I_ext = 1.5

    s = state
    for i in range(n_ticks):
        # Pulse only on first tick
        pulse = I_ext if i == 0 else cfg.fhn_I_rest
        s = fhn_step(s, pulse, cfg)
    return s


def fhn_activation_score(state: FHNState) -> float:
    """
    Convert FHN state to a [0, 1] activation score.

    Maps v from [-2, 2] to [0, 1] via sigmoid-like transform.
    Higher v = more retrievable.
    """
    return 1.0 / (1.0 + math.exp(-2.0 * state.v))


# ─── FFT Spectral Decay ───────────────────────────────────────────

def fft_decay(embedding: np.ndarray, envelope: float,
              alpha: float) -> np.ndarray:
    """
    Frequency-selective decay on embedding vector.

    High-frequency components (episodic detail) decay faster than
    low-frequency components (semantic gist).

    E_decayed[k] = E[k] * envelope * exp(-alpha * |k|)

    Args:
        embedding: The dense vector to decay
        envelope: FHN activation envelope v(t) in [0, 1]
        alpha: Frequency-selective slope (higher = more selective)

    Returns:
        Decayed embedding (same shape)
    """
    if alpha <= 0 or envelope >= 0.99:
        return embedding.copy()

    spectrum = np.fft.rfft(embedding)
    freqs = np.arange(len(spectrum))

    # Attenuation: envelope controls overall scale, alpha controls selectivity
    attenuation = envelope * np.exp(-alpha * freqs / len(freqs))
    # Low frequencies (gist) attenuate less than high frequencies (detail)
    spectrum *= attenuation

    result = np.fft.irfft(spectrum, n=len(embedding))

    # Renormalize to unit sphere
    norm = np.linalg.norm(result)
    if norm > 1e-12:
        result /= norm

    return result


# ─── Gabor Interference ───────────────────────────────────────────

def gabor_kernel(delta_t: float, cosine_sim: float,
                 cfg: NMEMConfig) -> float:
    """
    Gabor interference kernel between two memories.

    G(Δt, sim) = exp(-Δt²/2σ²) · cos(ω · sim)

    Memories close in time AND semantically similar interfere most.
    The cosine term produces oscillatory constructive/destructive pattern.
    """
    temporal = math.exp(-(delta_t ** 2) / (2.0 * cfg.gabor_sigma ** 2))
    spectral = math.cos(cfg.gabor_omega * cosine_sim)
    return temporal * spectral


# ─── Memory Node ───────────────────────────────────────────────────

@dataclass
class TurnHeader:
    """Lightweight metadata computed at encoding time."""
    index: int                      # Absolute turn number
    delta_t: float                  # Seconds since last turn
    salience: float                 # Cosine drift from previous turn
    topic_hash: str                 # Coarse semantic fingerprint
    savant_flag: bool               # Contains code/numbers/names/URLs


@dataclass
class MemoryNode:
    """
    A memory unit with multi-anchor encoding.

    Exists in multiple representational spaces simultaneously.
    Decays along multiple independent pathways.
    Lost only when ALL pathways fall below threshold.
    """
    id: str
    payload: str                            # Raw text (full or savant-preserved)
    created_at: float
    last_accessed: float
    last_decay_at: float                    # Last time decay was applied

    # Turn context
    header: TurnHeader

    # Multi-anchor encoding
    semantic_vec: List[float]               # Dense embedding (JSON-serializable)
    temporal_anchor: float                  # Score component from time
    topological_anchor: float               # Position in topic manifold
    salience_anchor: float                  # Salience/surprise score
    episodic_anchors: List[str]             # Hashes of ±3 neighboring turns

    # Decay state
    fhn: FHNState = field(default_factory=FHNState)
    retrieval_count: int = 0
    reconstruction_count: int = 0

    # Compression state
    compressed_payload: str = ""            # Lossy-compressed version
    compression_level: int = 0              # 0=full, 1=summarized, 2=gist, 3=tag

    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def age_hours(self) -> float:
        return self.age_seconds() / 3600.0

    def hours_since_decay(self) -> float:
        return (time.time() - self.last_decay_at) / 3600.0

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "payload": self.payload,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "last_decay_at": self.last_decay_at,
            "header": {
                "index": self.header.index,
                "delta_t": self.header.delta_t,
                "salience": self.header.salience,
                "topic_hash": self.header.topic_hash,
                "savant_flag": self.header.savant_flag,
            },
            "semantic_vec": self.semantic_vec,
            "temporal_anchor": self.temporal_anchor,
            "topological_anchor": self.topological_anchor,
            "salience_anchor": self.salience_anchor,
            "episodic_anchors": self.episodic_anchors,
            "fhn_v": self.fhn.v,
            "fhn_w": self.fhn.w,
            "retrieval_count": self.retrieval_count,
            "reconstruction_count": self.reconstruction_count,
            "compressed_payload": self.compressed_payload,
            "compression_level": self.compression_level,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        h = d["header"]
        node = cls(
            id=d["id"],
            payload=d["payload"],
            created_at=d["created_at"],
            last_accessed=d["last_accessed"],
            last_decay_at=d["last_decay_at"],
            header=TurnHeader(
                index=h["index"],
                delta_t=h["delta_t"],
                salience=h["salience"],
                topic_hash=h["topic_hash"],
                savant_flag=h["savant_flag"],
            ),
            semantic_vec=d["semantic_vec"],
            temporal_anchor=d["temporal_anchor"],
            topological_anchor=d["topological_anchor"],
            salience_anchor=d["salience_anchor"],
            episodic_anchors=d["episodic_anchors"],
            fhn=FHNState(v=d.get("fhn_v", 0.5), w=d.get("fhn_w", 0.0)),
            retrieval_count=d.get("retrieval_count", 0),
            reconstruction_count=d.get("reconstruction_count", 0),
            compressed_payload=d.get("compressed_payload", ""),
            compression_level=d.get("compression_level", 0),
        )
        return node


# ─── Savant Detection ──────────────────────────────────────────────

_CODE_PATTERN = re.compile(
    r'(?:def |class |import |from .+ import|function |const |let |var |'
    r'if\s*\(|for\s*\(|while\s*\(|=>|\.py|\.js|\.ts|\.cpp|\.rs)'
)
_NUMBER_PATTERN = re.compile(
    r'\b\d{4,}\b|0x[0-9a-fA-F]{4,}|\b\d+\.\d{3,}\b'
)
_URL_PATTERN = re.compile(r'https?://\S+|localhost:\d+|\w+\.\w+/\S+')
_PATH_PATTERN = re.compile(r'[A-Za-z]:[/\\]\S+|/(?:usr|home|var|etc|opt)/\S+')


def detect_savant_content(text: str) -> bool:
    """Check if text contains content that should be verbatim-preserved."""
    if _CODE_PATTERN.search(text):
        return True
    if _NUMBER_PATTERN.search(text):
        return True
    if _URL_PATTERN.search(text):
        return True
    if _PATH_PATTERN.search(text):
        return True
    return False


# ─── Salience Computation ──────────────────────────────────────────

def compute_salience(current_vec: np.ndarray,
                     previous_vec: Optional[np.ndarray],
                     embedder: HashEmbedder) -> float:
    """
    Salience = prediction error proxy.

    Cosine drift from previous turn embedding.
    High drift = boundary fired, FHN spikes, segment gets sharp marker.
    Low drift = inside segment, temporal resolution is coarse.
    """
    if previous_vec is None:
        return 1.0  # First turn is always salient
    sim = embedder.cosine_sim(current_vec, previous_vec)
    # Drift = 1 - similarity. Clamp to [0, 1].
    return max(0.0, min(1.0, 1.0 - sim))


def compute_topic_hash(text: str) -> str:
    """Coarse semantic fingerprint — first 8 hex of content hash."""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]


# ─── Anchor Scoring ───────────────────────────────────────────────

def score_semantic_anchor(query_vec: np.ndarray,
                          node_vec: np.ndarray,
                          embedder: HashEmbedder) -> float:
    """Cosine similarity between query and node embedding."""
    return max(0.0, embedder.cosine_sim(query_vec, node_vec))


def score_temporal_anchor(node: MemoryNode) -> float:
    """Recency-weighted temporal score. Half-life = 2 hours."""
    age_h = node.age_hours()
    return math.exp(-0.347 * age_h)  # ln(2)/2 ≈ 0.347


def score_salience_anchor(node: MemoryNode) -> float:
    """Salience at encoding, boosted by retrieval count."""
    base = node.salience_anchor
    boost = math.log1p(node.retrieval_count) * 0.1
    return min(1.0, base + boost)


def score_episodic_anchor(query_hash: str, node: MemoryNode) -> float:
    """Contextual match: does the query's topic appear in node's neighborhood?"""
    if query_hash in node.episodic_anchors:
        return 0.8
    return 0.0


def composite_retrieval_score(
    query_vec: np.ndarray,
    query_hash: str,
    node: MemoryNode,
    embedder: HashEmbedder,
    fhn_score: float,
) -> float:
    """
    Max over independent anchor scores, gated by FHN activation.

    Any surviving pathway is sufficient (Method of Loci property).
    FHN envelope modulates overall accessibility.
    """
    node_vec = np.array(node.semantic_vec)

    anchors = [
        score_semantic_anchor(query_vec, node_vec, embedder),
        score_temporal_anchor(node),
        score_salience_anchor(node),
        score_episodic_anchor(query_hash, node),
    ]

    # Max over anchors — any one pathway suffices
    max_anchor = max(anchors)

    # FHN gates overall accessibility
    return max_anchor * fhn_score


# ─── NMEM Controller ──────────────────────────────────────────────

class NMEM:
    """
    Neuromorphic Memory — biologically-inspired memory system.

    Integrates FHN excitable dynamics, FFT spectral forgetting,
    Gabor retroactive interference, multi-anchor retrieval,
    HSAM reconstruction, and savant verbatim preservation.

    Drop-in compatible with IMDMemory API.
    """

    def __init__(self, config: Optional[NMEMConfig] = None,
                 telemetry_state: Optional[Dict] = None):
        self.cfg = config or NMEMConfig()
        self._telemetry = telemetry_state or {}
        self._embedder = HashEmbedder(dim=self.cfg.embedding_dim)

        self._persist_dir = Path(self.cfg.persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self._persist_dir / "nodes.json"

        # Memory store: id → MemoryNode
        self._nodes: OrderedDict[str, MemoryNode] = OrderedDict()
        self._lock = threading.Lock()

        # Turn tracking
        self._turn_counter = 0
        self._last_turn_time = time.time()
        self._last_turn_vec: Optional[np.ndarray] = None

        # Ghost registry: ids of nodes that decayed below theta_low
        self._ghosts: Dict[str, str] = {}  # id → topic_hash

        # Load persisted state
        self._load()

        logger.info(
            f"NMEM online: {len(self._nodes)} nodes, "
            f"{len(self._ghosts)} ghosts, "
            f"persist={self._persist_dir}"
        )

    # ── Persistence ───────────────────────────────────────────────

    def _load(self):
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                for d in data.get("nodes", []):
                    node = MemoryNode.from_dict(d)
                    self._nodes[node.id] = node
                self._ghosts = data.get("ghosts", {})
                self._turn_counter = data.get("turn_counter", 0)
                self._last_turn_time = data.get("last_turn_time", time.time())
                logger.info(f"NMEM loaded {len(self._nodes)} nodes from disk")
            except Exception as e:
                logger.warning(f"NMEM load failed: {e}")

    def _save(self):
        try:
            data = {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "ghosts": self._ghosts,
                "turn_counter": self._turn_counter,
                "last_turn_time": self._last_turn_time,
            }
            tmp = self._store_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception as e:
            logger.warning(f"NMEM save failed: {e}")

    # ── Layer 0: Encoding ─────────────────────────────────────────

    # Content patterns to reject (system/salvo/PII)
    _REJECT_PATTERNS = frozenset({
        "IGBUNDLE // NEURAL GLASS", "SYSTEM STATUS", "MODEL IDENTITY",
        "SECTIONAL CURVATURE", "MANIFOLD ENTROPY", "CONSTRAINT SCORE",
        "Salvo 1", "Salvo 2", "Salvo 3", "Salvo 4", "Salvo 5",
        "ARC-AGI", "GSM8K", "Hamiltonian coherence",
        "password:", "ownerId:", "credential",
    })

    def store(self, content: str,
              curvature: Optional[float] = None,
              entropy: Optional[float] = None,
              tier: str = "auto",
              context: str = "") -> str:
        """
        Encode a memory at the current turn boundary.

        Computes multi-anchor encoding, applies attentional gating
        (salience threshold), detects savant content.

        Returns node ID or empty string if rejected.
        """
        if not content or not content.strip():
            return ""

        # Reject system/salvo/PII
        for pattern in self._REJECT_PATTERNS:
            if pattern in content:
                logger.debug(f"NMEM store rejected (pattern={pattern!r})")
                return ""

        content = content[:2000]  # Hard cap

        with self._lock:
            now = time.time()

            # Compute embedding
            vec = self._embedder.embed(content)

            # Salience = prediction error (cosine drift from last turn)
            salience = compute_salience(vec, self._last_turn_vec, self._embedder)

            # Attentional gating: low-salience content not encoded
            if salience < self.cfg.salience_threshold:
                logger.debug(f"NMEM gated out (salience={salience:.3f})")
                return ""

            # Turn header
            delta_t = now - self._last_turn_time
            self._turn_counter += 1
            header = TurnHeader(
                index=self._turn_counter,
                delta_t=delta_t,
                salience=salience,
                topic_hash=compute_topic_hash(content),
                savant_flag=detect_savant_content(content),
            )

            # Episodic anchors: topic hashes of recent turns
            recent_hashes = []
            recent_nodes = list(self._nodes.values())[-6:]
            for n in recent_nodes:
                recent_hashes.append(n.header.topic_hash)

            # Create node with fresh FHN state
            node_id = hashlib.md5(
                f"{content[:200]}:{now}".encode()
            ).hexdigest()[:16]

            node = MemoryNode(
                id=node_id,
                payload=content,
                created_at=now,
                last_accessed=now,
                last_decay_at=now,
                header=header,
                semantic_vec=vec.tolist(),
                temporal_anchor=1.0,  # Fresh — max temporal score
                topological_anchor=salience,  # Initial position = salience
                salience_anchor=salience,
                episodic_anchors=recent_hashes,
                fhn=FHNState(v=0.8, w=0.0),  # High initial activation
            )

            # Capacity enforcement
            if len(self._nodes) >= self.cfg.max_nodes:
                self._evict_weakest()

            self._nodes[node_id] = node

            # Update turn tracking
            self._last_turn_time = now
            self._last_turn_vec = vec.copy()

            self._save()
            return node_id

    def _evict_weakest(self):
        """Evict the node with lowest FHN activation to ghost."""
        if not self._nodes:
            return
        weakest_id = None
        weakest_score = float("inf")
        for nid, node in self._nodes.items():
            if node.header.savant_flag:
                continue  # Never evict savant nodes by capacity
            score = fhn_activation_score(node.fhn)
            if score < weakest_score:
                weakest_score = score
                weakest_id = nid
        if weakest_id:
            ghost_node = self._nodes.pop(weakest_id)
            self._ghosts[weakest_id] = ghost_node.header.topic_hash
            logger.debug(f"NMEM evicted {weakest_id} to ghost")

    # ── Layer 1: Decay ────────────────────────────────────────────

    def _run_decay(self):
        """
        Apply decay dynamics to all non-savant nodes.

        Called lazily before retrieval — not on a timer.
        Advances FHN, applies FFT spectral decay, runs Gabor interference.
        """
        now = time.time()
        to_ghost = []

        node_list = list(self._nodes.values())
        node_vecs = {
            n.id: np.array(n.semantic_vec)
            for n in node_list
        }

        for node in node_list:
            delta_h = node.hours_since_decay()

            if delta_h >= 0.01:  # At least 36 seconds elapsed
                # --- FHN advance ---
                node.fhn = fhn_advance(node.fhn, delta_h, self.cfg, retrieved=False)

                # --- FFT spectral decay (skip savant) ---
                if not node.header.savant_flag:
                    envelope = fhn_activation_score(node.fhn)
                    vec = np.array(node.semantic_vec)
                    decayed = fft_decay(vec, envelope, self.cfg.alpha)
                    node.semantic_vec = decayed.tolist()
                    node_vecs[node.id] = decayed

                node.last_decay_at = now

            # Check if node should become ghost (always, regardless of elapsed time)
            activation = fhn_activation_score(node.fhn)
            if activation < self.cfg.theta_low and not node.header.savant_flag:
                to_ghost.append(node.id)

        # --- Gabor interference (pairwise, O(N²) but N is small) ---
        ids = [n.id for n in node_list if not n.header.savant_flag]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ni = self._nodes.get(ids[i])
                nj = self._nodes.get(ids[j])
                if ni is None or nj is None:
                    continue

                vi = node_vecs.get(ids[i])
                vj = node_vecs.get(ids[j])
                if vi is None or vj is None:
                    continue

                sim = self._embedder.cosine_sim(vi, vj)
                if abs(sim) < self.cfg.gabor_radius:
                    continue  # Too dissimilar — no interference

                delta_t = abs(ni.created_at - nj.created_at)
                g = gabor_kernel(delta_t, sim, self.cfg)

                if abs(g) < 0.001:
                    continue

                # Apply perturbation to both embeddings
                perturbation = g * self.cfg.gabor_strength
                vi_new = vi + perturbation * vj
                vj_new = vj + perturbation * vi

                # Renormalize
                ni_norm = np.linalg.norm(vi_new)
                nj_norm = np.linalg.norm(vj_new)
                if ni_norm > 1e-12:
                    vi_new /= ni_norm
                if nj_norm > 1e-12:
                    vj_new /= nj_norm

                ni.semantic_vec = vi_new.tolist()
                nj.semantic_vec = vj_new.tolist()
                node_vecs[ids[i]] = vi_new
                node_vecs[ids[j]] = vj_new

        # Move decayed nodes to ghost
        for nid in to_ghost:
            node = self._nodes.pop(nid, None)
            if node:
                self._ghosts[nid] = node.header.topic_hash
                logger.debug(
                    f"NMEM node {nid} → ghost "
                    f"(activation={fhn_activation_score(node.fhn):.3f})"
                )

    # ── Layer 2: Retrieval ────────────────────────────────────────

    def retrieve(self, query: str, token_budget: int = 0,
                 curvature: Optional[float] = None,
                 user_id: str = "demo_user") -> str:
        """
        Multi-anchor retrieval with threshold bands.

        Returns formatted context string within token budget.
        Triggers reconstruction for middle-band nodes.
        Ghost entries appear as bracketed gaps.
        """
        if not query or not query.strip():
            return ""

        budget = token_budget or self.cfg.token_budget

        with self._lock:
            # Run lazy decay before retrieval
            self._run_decay()

            query_vec = self._embedder.embed(query)
            query_hash = compute_topic_hash(query)

            scored: List[Tuple[float, MemoryNode, str]] = []
            # band: "full", "reconstruct", "ghost"

            for node in self._nodes.values():
                fhn_score = fhn_activation_score(node.fhn)
                score = composite_retrieval_score(
                    query_vec, query_hash, node,
                    self._embedder, fhn_score,
                )

                if score >= self.cfg.theta_high:
                    band = "full"
                elif score >= self.cfg.theta_low:
                    band = "reconstruct"
                else:
                    band = "ghost"

                scored.append((score, node, band))

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Build context
            header = "[Memory Context]"
            lines = []
            char_budget = budget * 4  # ~4 chars/token
            used = len(header) + 2

            for score, node, band in scored:
                if used >= char_budget:
                    break

                if band == "full":
                    # Full payload returned
                    text = node.payload
                    if node.header.savant_flag:
                        tag = "S*"  # Savant-preserved
                    else:
                        tag = f"T{node.header.index}"

                    # Mark as retrieved — FHN spike
                    node.fhn = fhn_advance(
                        node.fhn, 0.001, self.cfg, retrieved=True
                    )
                    node.retrieval_count += 1
                    node.last_accessed = time.time()

                elif band == "reconstruct":
                    # Reconstruction: use compressed payload if available
                    text = self._reconstruct(node)
                    tag = f"T{node.header.index}~"  # ~ = reconstructed

                    node.fhn = fhn_advance(
                        node.fhn, 0.001, self.cfg, retrieved=True
                    )
                    node.retrieval_count += 1
                    node.reconstruction_count += 1
                    node.last_accessed = time.time()

                    # Check if too many reconstructions → ghost
                    if node.reconstruction_count > self.cfg.max_reconstructions:
                        self._ghosts[node.id] = node.header.topic_hash
                        del self._nodes[node.id]
                        continue

                else:
                    # Ghost — existence known, content inaccessible
                    continue

                line = f"- [{tag}] {text}"
                if used + len(line) + 1 > char_budget:
                    # Try truncated
                    remaining = char_budget - used - len(f"- [{tag}] ") - 4
                    if remaining > 50:
                        line = f"- [{tag}] {text[:remaining]}..."
                    else:
                        break
                lines.append(line)
                used += len(line) + 1

            # Append ghost indicators for relevant ghosts
            for ghost_id, topic in self._ghosts.items():
                if query_hash == topic or topic in query.lower():
                    ghost_line = f"- [ghost] [content decayed, topic: {topic}]"
                    if used + len(ghost_line) + 1 <= char_budget:
                        lines.append(ghost_line)
                        used += len(ghost_line) + 1

            self._save()

            if not lines:
                return ""
            return header + "\n" + "\n".join(lines)

    # ── Layer 3: Reconstruction ───────────────────────────────────

    def _reconstruct(self, node: MemoryNode) -> str:
        """
        HSAM reconstruction for middle-band nodes.

        Uses episodic neighborhood + surviving semantic content
        to reconstruct a degraded version of the payload.

        Each reconstruction applies a small perturbation
        (reconsolidation drift).
        """
        # If we have a compressed version, use it
        if node.compressed_payload:
            text = node.compressed_payload
        else:
            # Progressive compression based on reconstruction count
            text = node.payload

        # Apply lossy compression proportional to reconstruction count
        level = min(node.reconstruction_count, 3)

        if level == 0:
            # First reconstruction: trim to first 2/3
            cut = max(50, len(text) * 2 // 3)
            compressed = text[:cut]
        elif level == 1:
            # Second: first half
            cut = max(40, len(text) // 2)
            compressed = text[:cut]
        elif level == 2:
            # Third: first sentence or 80 chars
            first_sent = re.split(r'[.!?]\s', text)
            compressed = first_sent[0][:80] if first_sent else text[:80]
        else:
            # Beyond: just the topic hash and turn index
            compressed = f"[gist: T{node.header.index}, topic={node.header.topic_hash}]"

        node.compressed_payload = compressed
        node.compression_level = level

        # Reconsolidation perturbation on embedding
        vec = np.array(node.semantic_vec)
        noise = np.random.randn(len(vec)) * self.cfg.reconstruction_perturbation
        vec += noise
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        node.semantic_vec = vec.tolist()

        return compressed

    # ── Layer 4: Context Assembly ─────────────────────────────────

    def assemble_context(self, query: str,
                         token_budget: int = 0) -> List[Dict[str, Any]]:
        """
        Full context assembly for inference.

        Returns structured list of memory items ordered by
        topological coordinate (narrative arc), not timestamp.
        Includes ghost gaps.
        """
        budget = token_budget or self.cfg.token_budget

        with self._lock:
            self._run_decay()
            query_vec = self._embedder.embed(query)
            query_hash = compute_topic_hash(query)

            items = []
            for node in self._nodes.values():
                fhn_score = fhn_activation_score(node.fhn)
                score = composite_retrieval_score(
                    query_vec, query_hash, node,
                    self._embedder, fhn_score,
                )

                if score >= self.cfg.theta_high:
                    items.append({
                        "turn": node.header.index,
                        "band": "full",
                        "text": node.payload,
                        "salience": node.salience_anchor,
                        "score": score,
                        "savant": node.header.savant_flag,
                        "topic": node.header.topic_hash,
                    })
                elif score >= self.cfg.theta_low:
                    text = self._reconstruct(node)
                    items.append({
                        "turn": node.header.index,
                        "band": "reconstructed",
                        "text": text,
                        "salience": node.salience_anchor,
                        "score": score,
                        "savant": False,
                        "topic": node.header.topic_hash,
                    })

            # Ghost gaps
            for gid, topic in self._ghosts.items():
                items.append({
                    "turn": -1,
                    "band": "ghost",
                    "text": f"[content decayed, topic: {topic}]",
                    "salience": 0.0,
                    "score": 0.0,
                    "savant": False,
                    "topic": topic,
                })

            # Order by topological coordinate (salience as proxy for arc position)
            items.sort(key=lambda x: (-x["salience"], -x["score"]))

            return items

    # ── Drop-in IMDMemory API ─────────────────────────────────────

    def add(self, content: str, user_id: str = "demo_user",
            metadata: Optional[Dict] = None):
        return self.store(content, tier="auto")

    def search(self, query: str, user_id: str = "demo_user",
               limit: int = 2):
        # Return list of dicts for compatibility
        with self._lock:
            self._run_decay()
            query_vec = self._embedder.embed(query)
            query_hash = compute_topic_hash(query)

            results = []
            for node in self._nodes.values():
                fhn_score = fhn_activation_score(node.fhn)
                score = composite_retrieval_score(
                    query_vec, query_hash, node,
                    self._embedder, fhn_score,
                )
                if score >= self.cfg.theta_low:
                    results.append((score, node))

            results.sort(key=lambda x: x[0], reverse=True)
            return [
                {"memory": n.payload, "score": s}
                for s, n in results[:limit]
            ]

    def get_context_string(self, query: str,
                           user_id: str = "demo_user") -> str:
        return self.retrieve(query, token_budget=self.cfg.token_budget,
                             user_id=user_id)

    def compact_conversation(self, messages: List[Dict[str, str]],
                              max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Compact conversation history, storing dropped messages as memories.

        Unlike IMDMemory's flat extractive summary, NMEM stores each
        dropped turn as a separate memory node with full multi-anchor
        encoding. The turn header captures temporal position and salience.
        """
        if not messages or len(messages) <= max_messages + 1:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= max_messages:
            return messages

        dropped = non_system[:-max_messages]
        kept = non_system[-max_messages:]

        # Store each dropped turn as a memory node
        for msg in dropped:
            content = msg.get("content", "")
            if content and len(content) > 20:
                self.store(content[:1000])

        # Ghost-aware summary: include existence markers for stored content
        ghost_note = f"[{len(dropped)} earlier turns stored in NMEM memory]"
        context_note = {
            "role": "system",
            "content": ghost_note,
        }

        return system_msgs + [context_note] + kept

    # ── Diagnostics ───────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            activations = [
                fhn_activation_score(n.fhn) for n in self._nodes.values()
            ]
            savant_count = sum(
                1 for n in self._nodes.values() if n.header.savant_flag
            )
            return {
                "total_nodes": len(self._nodes),
                "ghost_count": len(self._ghosts),
                "savant_count": savant_count,
                "mean_activation": (
                    sum(activations) / len(activations)
                    if activations else 0.0
                ),
                "min_activation": min(activations) if activations else 0.0,
                "max_activation": max(activations) if activations else 0.0,
                "turn_counter": self._turn_counter,
                "config": {
                    "fhn_epsilon": self.cfg.fhn_epsilon,
                    "alpha": self.cfg.alpha,
                    "theta_high": self.cfg.theta_high,
                    "theta_low": self.cfg.theta_low,
                    "max_nodes": self.cfg.max_nodes,
                },
            }

    def get_node_details(self, node_id: str) -> Optional[Dict]:
        """Inspect a specific memory node."""
        node = self._nodes.get(node_id)
        if not node:
            if node_id in self._ghosts:
                return {"status": "ghost", "topic": self._ghosts[node_id]}
            return None
        return {
            "id": node.id,
            "payload_preview": node.payload[:100],
            "turn": node.header.index,
            "savant": node.header.savant_flag,
            "salience": node.salience_anchor,
            "fhn_v": node.fhn.v,
            "fhn_w": node.fhn.w,
            "activation": fhn_activation_score(node.fhn),
            "retrieval_count": node.retrieval_count,
            "reconstruction_count": node.reconstruction_count,
            "compression_level": node.compression_level,
            "age_hours": node.age_hours(),
        }
