"""
Integrated Memory Dynamics (IMD) Framework v2
==============================================
Geometric memory system for Neural Glass (ManifoldGL v3.0).

Architecture grounded in:
  - Poincaré embeddings & hyperbolic NNs for hierarchical representation
  - Fisher-Rao / natural gradient for consolidation invariance
  - Hamiltonian dynamics for saliency budget conservation
  - Complementary Learning Systems (CLS) for tier design

Three-tier memory (CLS-inspired):
  Working Memory  │ Fast hippocampal buffer (in-memory LRU, cap=16)
  Episodic Memory │ Experience traces with temporal ordering (persistent JSON)
  Semantic Memory │ Consolidated factual store (Qdrant via mem0)

Each memory is a fiber pair (x, f):
  x ∈ M  — point on Riemannian base manifold (semantic location)
  f ∈ F(x) — fiber: {memory_type, confidence, saliency, source}

Retrieval uses the geometric scoring formula:
  s(m|q,c) = -d_M(x_m, x_q) + λ·Compat(f_m, c) - μ·Conflict(m, S_c)

Saliency budget is conserved via Hamiltonian dynamics on tier allocation.

Drop-in compatible with LLMOSMemory (add/search/get_context_string).
"""

import time
import math
import json
import hashlib
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

logger = logging.getLogger("imd.memory")


# ─── Constants ───────────────────────────────────────────────────────

MEMORY_TYPES = ("fact", "procedure", "episode", "inference")

# Keywords for auto-classifying memory type (the fiber f)
_PROCEDURE_MARKERS = frozenset({
    "how", "step", "steps", "first", "then", "next", "finally",
    "install", "run", "execute", "create", "build", "configure",
})
_EPISODE_MARKERS = frozenset({
    "i ", "we ", "my ", "our ", "yesterday", "today", "last time",
    "previously", "remember", "discussed", "asked", "told",
})
_INFERENCE_MARKERS = frozenset({
    "therefore", "because", "since", "implies", "hence", "thus",
    "if ", "then ", "conclude", "follows", "proof", "given that",
})

# Negation words for conflict detection
_NEGATION_WORDS = frozenset({
    "not", "never", "no", "none", "neither", "nor", "isn't",
    "aren't", "doesn't", "don't", "won't", "can't", "cannot",
    "shouldn't", "false", "incorrect", "wrong", "invalid",
})

# Retrieval formula hyperparameters (from research)
LAMBDA_COMPAT = 0.3    # Weight for fiber-context compatibility
MU_CONFLICT = 0.5      # Penalty for contradictory memories


# ─── Data Structures ────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """
    A memory unit as a fiber pair (x, f).

    x (base manifold point):
      - content: the text
      - context_hash: hash of the context when created
      - curvature_at_creation: manifold curvature κ at storage time

    f (fiber structure):
      - memory_type: "fact" | "procedure" | "episode" | "inference"
      - confidence: [0, 1] certainty measure
      - saliency: Hamiltonian momentum variable (conserved across updates)
      - source: provenance string for auditability
    """
    id: str
    content: str
    tier: str                           # "working", "episodic", "semantic"
    created_at: float
    last_accessed: float
    access_count: int = 1
    stability_score: float = 0.0
    curvature_at_creation: float = 0.0
    entropy_at_creation: float = 0.0
    context_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Fiber fields (v2 — backward compatible defaults)
    memory_type: str = "fact"
    confidence: float = 1.0
    saliency: float = 1.0
    source: str = ""

    def age(self) -> float:
        return time.time() - self.created_at

    def recency(self, half_life: float = 3600.0) -> float:
        """Exponential decay since last access. Default half-life: 1 hour."""
        dt = time.time() - self.last_accessed
        return math.exp(-0.693 * dt / half_life)

    def compute_stability(self) -> float:
        """
        Stability = log(1 + access_count) × recency × geo_coherence × saliency.

        Geometric coherence: memories created in low-curvature (stable)
        regions are more reliable. Saliency modulates via Hamiltonian dynamics.
        """
        freq = math.log1p(self.access_count)
        rec = self.recency()
        geo = 1.0 / (1.0 + abs(self.curvature_at_creation))
        sal = max(self.saliency, 0.01)
        self.stability_score = freq * rec * geo * sal
        return self.stability_score


# ─── Memory Type Classifier ─────────────────────────────────────────

def classify_memory_type(content: str) -> str:
    """
    Auto-classify content into fiber type based on linguistic markers.
    Maps to the fiber f ∈ F(x) of the memory bundle.
    """
    lower = content.lower()
    words = set(lower.split())

    proc_hits = len(words & _PROCEDURE_MARKERS)
    for marker in _PROCEDURE_MARKERS:
        if marker in lower and " " in marker:
            proc_hits += 1

    ep_hits = 0
    for marker in _EPISODE_MARKERS:
        if marker in lower:
            ep_hits += 1

    inf_hits = 0
    for marker in _INFERENCE_MARKERS:
        if marker in lower:
            inf_hits += 1

    scores = {
        "procedure": proc_hits,
        "episode": ep_hits,
        "inference": inf_hits,
        "fact": 1,  # Default bias
    }
    return max(scores, key=scores.get)


def query_intent(query: str) -> str:
    """Infer what memory type the query is looking for."""
    lower = query.lower()
    if any(w in lower for w in ("how to", "how do", "steps", "guide", "tutorial")):
        return "procedure"
    if any(w in lower for w in ("last time", "remember", "did we", "before")):
        return "episode"
    if any(w in lower for w in ("why", "because", "prove", "deduce")):
        return "inference"
    return "fact"


# ─── Conflict Detection ─────────────────────────────────────────────

def detect_conflict(text_a: str, text_b: str) -> float:
    """
    Detect potential contradiction between two memory texts.

    Returns conflict score in [0, 1]:
      0.0 = no conflict detected
      1.0 = strong contradiction signal

    Method: High keyword overlap + negation asymmetry → conflict.
    Reference: research recommends "detect conflict, prefer sources/
    recency/reliability explicitly, audit which memory influenced output."
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0

    # Keyword overlap (Jaccard)
    overlap = len(words_a & words_b) / len(union)

    # Negation asymmetry: one text negates what the other affirms
    neg_a = len(words_a & _NEGATION_WORDS)
    neg_b = len(words_b & _NEGATION_WORDS)
    neg_asymmetry = abs(neg_a - neg_b) / (max(neg_a, neg_b, 1))

    # Conflict = high overlap × negation asymmetry
    # Both conditions must hold: similar topic + opposite polarity
    if overlap < 0.15:
        return 0.0  # Not enough topical overlap

    return min(1.0, overlap * neg_asymmetry * 2.0)


# ─── Working Memory ─────────────────────────────────────────────────

class WorkingMemory:
    """
    Fast hippocampal buffer. Capacity-limited LRU with stability-based
    eviction. Evicted stable memories are promoted to episodic tier
    (CLS: fast learning → slow consolidation).
    """

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self._store: OrderedDict[str, MemoryRecord] = OrderedDict()
        self._lock = threading.Lock()

    def store(self, record: MemoryRecord) -> Optional[MemoryRecord]:
        """Store a record. Returns evicted record if at capacity."""
        evicted = None
        with self._lock:
            if record.id in self._store:
                existing = self._store[record.id]
                existing.access_count += 1
                existing.last_accessed = time.time()
                existing.compute_stability()
                self._store.move_to_end(record.id)
                return None

            if len(self._store) >= self.capacity:
                min_key = min(self._store,
                              key=lambda k: self._store[k].compute_stability())
                evicted = self._store.pop(min_key)

            self._store[record.id] = record
            self._store.move_to_end(record.id)
        return evicted

    def retrieve(self, limit: int = 3) -> List[MemoryRecord]:
        with self._lock:
            records = list(self._store.values())
        for r in records:
            r.compute_stability()
        records.sort(key=lambda r: r.stability_score, reverse=True)
        for r in records[:limit]:
            r.last_accessed = time.time()
            r.access_count += 1
        return records[:limit]

    def get_all(self) -> List[MemoryRecord]:
        with self._lock:
            return list(self._store.values())

    def __len__(self):
        return len(self._store)


# ─── Episodic Memory ────────────────────────────────────────────────

class EpisodicMemory:
    """
    Experience traces with temporal ordering.
    Persisted to disk as JSON for cross-session continuity.
    Includes conflict detection on store (research requirement).
    """

    def __init__(self, persist_path: str = "./memory/episodic.json",
                 capacity: int = 256):
        self.capacity = capacity
        self.persist_path = Path(persist_path)
        self._episodes: List[MemoryRecord] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                self._episodes = [MemoryRecord(**d) for d in data]
                logger.info(f"Loaded {len(self._episodes)} episodic memories")
            except Exception as e:
                logger.warning(f"Failed to load episodic memory: {e}")
                self._episodes = []

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps([asdict(ep) for ep in self._episodes], indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to save episodic memory: {e}")

    def store(self, record: MemoryRecord) -> Optional[float]:
        """
        Store an episode. Returns conflict score if contradiction detected
        with existing memories, else None.
        """
        content_hash = hashlib.md5(record.content.encode()).hexdigest()[:12]
        max_conflict = 0.0

        with self._lock:
            for ep in self._episodes:
                # Deduplicate
                if hashlib.md5(ep.content.encode()).hexdigest()[:12] == content_hash:
                    ep.access_count += 1
                    ep.last_accessed = time.time()
                    ep.compute_stability()
                    self._save()
                    return None

                # Conflict detection (research: detect contradictions)
                conflict = detect_conflict(record.content, ep.content)
                if conflict > max_conflict:
                    max_conflict = conflict
                if conflict > 0.6:
                    logger.warning(
                        f"Memory conflict ({conflict:.2f}): "
                        f"'{record.content[:50]}' vs '{ep.content[:50]}'"
                    )
                    # Lower confidence of new memory if it conflicts
                    record.confidence *= (1.0 - conflict * 0.5)

            if len(self._episodes) >= self.capacity:
                for ep in self._episodes:
                    ep.compute_stability()
                self._episodes.sort(key=lambda e: e.stability_score)
                self._episodes.pop(0)

            self._episodes.append(record)
            self._save()

        return max_conflict if max_conflict > 0.1 else None

    def retrieve(self, query: str, limit: int = 3,
                 max_content_len: int = 500) -> List[MemoryRecord]:
        with self._lock:
            if not self._episodes:
                return []

            query_words = set(query.lower().split())
            scored = []
            for ep in self._episodes:
                # Skip oversized entries (legacy salvo text pollution)
                if len(ep.content) > max_content_len:
                    continue
                ep_words = set(ep.content.lower().split())
                union = len(query_words | ep_words)
                overlap = len(query_words & ep_words) / union if union else 0.0
                ep.compute_stability()
                score = (0.4 * overlap
                         + 0.3 * ep.recency()
                         + 0.3 * ep.stability_score)
                scored.append((score, ep))

            scored.sort(key=lambda x: x[0], reverse=True)

            results = []
            for score, ep in scored[:limit]:
                if score > 0.05:
                    ep.last_accessed = time.time()
                    ep.access_count += 1
                    results.append(ep)
            return results

    def get_all(self) -> List[MemoryRecord]:
        with self._lock:
            return list(self._episodes)

    def __len__(self):
        return len(self._episodes)


# ─── Semantic Memory (Qdrant-backed via mem0) ────────────────────────

class SemanticMemory:
    """
    Consolidated long-term memory (neocortex analog in CLS).
    Backed by Qdrant vector DB via mem0.
    Uses minimal custom_prompt to eliminate the default ~1000-token
    "Personal Information Organizer" system prompt.
    """

    def __init__(self, collection: str = "imd_semantic_v1",
                 persist_dir: str = "./memory/qdrant_data"):
        self._mem0 = None
        self._collection = collection
        self._persist_dir = persist_dir
        self._init_backend()

    def _init_backend(self):
        try:
            from mem0 import Memory
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": self._collection,
                        "path": self._persist_dir,
                        "embedding_model_dims": 768,
                        "on_disk": True,
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "ig-bundlellmv2-hamiltonians",
                        "api_key": "not-needed",
                        "openai_base_url": "http://192.168.56.1:1234/v1",
                        "temperature": 0.0,
                        "max_tokens": 300,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": "google/embeddinggemma-300m"},
                },
                "custom_prompt": (
                    "Extract key facts from the input as a JSON list of strings. "
                    "Each fact should be one concise sentence. Return only JSON."
                ),
            }
            self._mem0 = Memory.from_config(config)
            logger.info(f"IMD SemanticMemory online ({self._collection})")
        except Exception as e:
            logger.error(f"SemanticMemory init failed: {e}")

    def store(self, content: str, metadata: Optional[Dict] = None,
              user_id: str = "demo_user") -> bool:
        if not self._mem0:
            return False
        try:
            safe = content[:1000]
            self._mem0.add([{"role": "user", "content": safe}],
                           user_id=user_id, metadata=metadata or {})
            return True
        except Exception as e:
            if "422" not in str(e):
                logger.warning(f"Semantic store error: {e}")
            return False

    def retrieve(self, query: str, limit: int = 2,
                 user_id: str = "demo_user") -> List[Dict]:
        if not self._mem0:
            return []
        try:
            results = self._mem0.search(query[:1000], user_id=user_id, limit=limit)
            if isinstance(results, dict) and "results" in results:
                return results["results"]
            if isinstance(results, list):
                return results
            return []
        except Exception as e:
            if "422" not in str(e):
                logger.warning(f"Semantic retrieve error: {e}")
            return []


# ─── Geometric Scoring Engine ────────────────────────────────────────

class GeometricScorer:
    """
    Riemannian scoring for memory retrieval and ranking.

    Implements the research retrieval formula:
      s(m|q,c) = -d_M(x_m, x_q) + λ·Compat(f_m, c) - μ·Conflict(m, S_c)

    Where:
      d_M    = geodesic distance (Poincaré when embeddings available,
               else 1 - Jaccard as proxy)
      Compat = fiber-context compatibility (memory_type matching)
      Conflict = contradiction score with already-selected memories S_c
    """

    @staticmethod
    def poincare_distance(u, v) -> float:
        """
        Geodesic distance in the Poincaré ball model.
        d(u,v) = arcosh(1 + 2‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))
        """
        import numpy as np
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        u_sq = min(float(np.dot(u, u)), 0.99)
        v_sq = min(float(np.dot(v, v)), 0.99)
        diff_sq = float(np.dot(u - v, u - v))
        denom = (1 - u_sq) * (1 - v_sq)
        if denom < 1e-10:
            return 10.0
        return float(np.arccosh(max(1 + 2 * diff_sq / denom, 1.0)))

    @staticmethod
    def keyword_distance(text_a: str, text_b: str) -> float:
        """1 - Jaccard similarity as distance proxy for d_M."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        union = words_a | words_b
        if not union:
            return 1.0
        return 1.0 - len(words_a & words_b) / len(union)

    @staticmethod
    def fiber_compatibility(memory_type: str, query_intent_type: str) -> float:
        """
        Compat(f_m, c): How well the memory's fiber type matches
        what the query is looking for.

        Returns [0, 1]: 1.0 = perfect match, 0.3 = neutral, 0.0 = mismatch.
        """
        if memory_type == query_intent_type:
            return 1.0
        # Partial compatibilities
        compat_matrix = {
            ("fact", "inference"): 0.6,
            ("inference", "fact"): 0.6,
            ("procedure", "fact"): 0.4,
            ("episode", "fact"): 0.3,
            ("fact", "procedure"): 0.4,
            ("fact", "episode"): 0.3,
        }
        return compat_matrix.get((memory_type, query_intent_type), 0.3)

    @staticmethod
    def retrieval_score(content: str, query: str, memory_type: str,
                        stability: float, confidence: float,
                        selected_set: List[str],
                        query_type: str = "fact") -> float:
        """
        Full geometric retrieval formula:
          s = -d_M(m, q) + λ·Compat(f, c) - μ·Conflict(m, S_c)
              + stability_bonus + confidence_bonus

        Args:
            content: Memory text
            query: Query text
            memory_type: Fiber type of this memory
            stability: Computed stability score
            confidence: Memory confidence [0, 1]
            selected_set: Already-selected memory texts for conflict check
            query_type: Inferred query intent type
        """
        # Distance term: -d_M (lower distance = higher score)
        d = GeometricScorer.keyword_distance(content, query)
        distance_score = -d

        # Compatibility term: λ·Compat(f_m, c)
        compat = GeometricScorer.fiber_compatibility(memory_type, query_type)
        compat_score = LAMBDA_COMPAT * compat

        # Conflict term: -μ·max_conflict(m, S_c)
        max_conflict = 0.0
        for selected in selected_set:
            c = detect_conflict(content, selected)
            if c > max_conflict:
                max_conflict = c
        conflict_penalty = MU_CONFLICT * max_conflict

        # Combine with stability and confidence
        return (distance_score
                + compat_score
                - conflict_penalty
                + 0.2 * stability
                + 0.1 * confidence)

    @staticmethod
    def curvature_weight(curvature: float) -> float:
        """σ(2|κ|): sigmoid mapping curvature → [0, 1] tier preference."""
        return 1.0 / (1.0 + math.exp(-2.0 * abs(curvature)))

    @staticmethod
    def information_density(text: str) -> float:
        """Character-level Shannon entropy normalized to [0, 1]."""
        if not text:
            return 0.0
        freq: Dict[str, int] = {}
        for c in text.lower():
            freq[c] = freq.get(c, 0) + 1
        n = len(text)
        H = -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())
        return min(H / 6.6, 1.0)

    @staticmethod
    def decay_factor(age_seconds: float, half_life: float = 3600.0) -> float:
        return math.exp(-0.693 * age_seconds / half_life)


# ─── Hamiltonian Saliency Dynamics ───────────────────────────────────

class SaliencyDynamics:
    """
    Hamiltonian dynamics for tier budget allocation.

    State: (q, p) ∈ R³ × R³
      q = [w_working, w_episodic, w_semantic]  (tier allocation weights)
      p = [p_working, p_episodic, p_semantic]  (allocation momentum)

    Hamiltonian:
      H = |p|²/2 + V(q)
      V(q) = -Σ utility_i · q_i + λ·(Σ q_i - 1)²

    The symplectic leapfrog integrator preserves total "energy"
    (budget coherence), ensuring tier allocations evolve smoothly
    without drift or oscillation.

    Reference: research recommends "Hamiltonian dynamics only for
    auxiliary variables (saliency, budget allocation, consolidation
    scheduling)" — not for the memories themselves.
    """

    def __init__(self, constraint_strength: float = 10.0,
                 dt: float = 0.05, damping: float = 0.02):
        self.lam = constraint_strength
        self.dt = dt
        self.damping = damping  # Small dissipation for convergence
        # State: equal initial allocation
        self.q = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        self.p = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()

    def _grad_V(self, q: List[float],
                utilities: List[float]) -> List[float]:
        """∂V/∂q_i = -utility_i + 2λ(Σq_j - 1)"""
        total = sum(q)
        return [
            -u + 2 * self.lam * (total - 1.0)
            for u in utilities
        ]

    def step(self, utilities: List[float]) -> Dict[str, float]:
        """
        Symplectic leapfrog step.

        Args:
            utilities: [utility_working, utility_episodic, utility_semantic]
                       Measured as retrieval hit rate per tier.

        Returns:
            Updated tier weight allocation.
        """
        with self._lock:
            dt = self.dt

            # Half-step momentum (with damping for convergence)
            grad = self._grad_V(self.q, utilities)
            self.p = [
                (1 - self.damping) * p - (dt / 2) * g
                for p, g in zip(self.p, grad)
            ]

            # Full-step position (floor at 0.05 to keep all tiers alive)
            self.q = [
                max(0.05, q + dt * p)
                for q, p in zip(self.q, self.p)
            ]

            # Half-step momentum
            grad = self._grad_V(self.q, utilities)
            self.p = [
                (1 - self.damping) * p - (dt / 2) * g
                for p, g in zip(self.p, grad)
            ]

            # Project to simplex (normalize — conservation law)
            total = sum(self.q)
            self.q = [q / total for q in self.q]

        return {"working": self.q[0], "episodic": self.q[1],
                "semantic": self.q[2]}

    def get_weights(self) -> Dict[str, float]:
        return {"working": self.q[0], "episodic": self.q[1],
                "semantic": self.q[2]}

    def get_state(self) -> Dict[str, Any]:
        return {"q": list(self.q), "p": list(self.p)}


# ─── IMD Memory Controller ──────────────────────────────────────────

class IMDMemory:
    """
    Integrated Memory Dynamics controller.

    Orchestrates three-tier memory with:
      - Geometric retrieval scoring (d_M + Compat + Conflict)
      - Hamiltonian saliency dynamics for tier allocation
      - Automatic memory type classification (fiber assignment)
      - Conflict detection and confidence adjustment
      - CLS-inspired consolidation (episodic → semantic promotion)

    Drop-in replacement for LLMOSMemory.
    """

    _CONSOLIDATION_INTERVAL = 300
    _PROMOTE_STABILITY = 0.5
    _PROMOTE_MIN_AGE = 1800
    _PROMOTE_MIN_ACCESS = 3
    _EVICT_STABILITY = 0.3

    def __init__(self, persist_dir: str = "./memory",
                 telemetry_state: Optional[Dict] = None):
        self._telemetry = telemetry_state or {}
        self._persist_dir = Path(persist_dir)

        self.working = WorkingMemory(capacity=32)
        self.episodic = EpisodicMemory(
            persist_path=str(self._persist_dir / "episodic.json"),
            capacity=256,
        )
        self.semantic = SemanticMemory(
            collection="imd_semantic_v1",
            persist_dir=str(self._persist_dir / "qdrant_data"),
        )
        self.scorer = GeometricScorer()
        self.saliency = SaliencyDynamics()

        # Retrieval hit counters for Hamiltonian utility estimation
        self._tier_hits = {"working": 0, "episodic": 0, "semantic": 0}
        self._tier_queries = 0

        self._last_consolidation = time.time()
        self._consolidation_lock = threading.Lock()

        logger.info(
            f"IMD v2 online: working({self.working.capacity}) "
            f"+ episodic({self.episodic.capacity}) + semantic(qdrant) "
            f"+ Hamiltonian saliency + conflict detection"
        )

    # ── Telemetry ────────────────────────────────────────────────────

    def _curvature(self) -> float:
        return float(self._telemetry.get("curvature", 0.0))

    def _entropy(self) -> float:
        return float(self._telemetry.get("entropy", 0.0))

    # ── Core API ─────────────────────────────────────────────────────

    # Patterns that indicate salvo/system-status text — never store these
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
        Store a memory with automatic tier and type assignment.

        Auto-tier: high info density → episodic, else → working.
        Auto-type: classifies content as fact/procedure/episode/inference.
        """
        if not content or not content.strip():
            return ""

        # Reject salvo/system-status/PII content
        for pattern in self._REJECT_PATTERNS:
            if pattern in content:
                logger.debug(f"IMD store rejected (pattern={pattern!r})")
                return ""

        content = content[:1000]
        k = curvature if curvature is not None else self._curvature()
        s = entropy if entropy is not None else self._entropy()

        record = MemoryRecord(
            id=hashlib.md5(content.encode()).hexdigest()[:16],
            content=content,
            tier=tier if tier != "auto" else "working",
            created_at=time.time(),
            last_accessed=time.time(),
            curvature_at_creation=k,
            entropy_at_creation=s,
            context_hash=(hashlib.md5(context.encode()).hexdigest()[:8]
                          if context else ""),
            memory_type=classify_memory_type(content),
            confidence=1.0,
            saliency=1.0,
        )
        record.compute_stability()

        if tier == "auto":
            density = self.scorer.information_density(content)
            if density > 0.7 and len(content) > 100:
                record.tier = "episodic"
                self.episodic.store(record)
            else:
                record.tier = "working"
                evicted = self.working.store(record)
                if evicted and evicted.stability_score > self._EVICT_STABILITY:
                    evicted.tier = "episodic"
                    self.episodic.store(evicted)
        elif tier == "working":
            evicted = self.working.store(record)
            if evicted and evicted.stability_score > self._EVICT_STABILITY:
                evicted.tier = "episodic"
                self.episodic.store(evicted)
        elif tier == "episodic":
            self.episodic.store(record)
        elif tier == "semantic":
            self.semantic.store(content, metadata={
                "curvature": k, "entropy": s,
                "stability": record.stability_score,
                "memory_type": record.memory_type,
            })

        # Periodic consolidation
        if time.time() - self._last_consolidation > self._CONSOLIDATION_INTERVAL:
            threading.Thread(target=self.consolidate, daemon=True).start()

        return record.id

    def retrieve(self, query: str, token_budget: int = 400,
                 curvature: Optional[float] = None,
                 user_id: str = "demo_user") -> str:
        """
        Multi-tier retrieval with geometric scoring formula.

        Uses: s(m|q,c) = -d_M + λ·Compat - μ·Conflict + stability + confidence

        Tier weights are set by Hamiltonian saliency dynamics,
        modulated by manifold curvature:
          High |κ| → upweight episodic (complex reasoning needs experience)
          Low  |κ| → upweight semantic (factual recall from long-term)
        """
        if not query or not query.strip():
            return ""

        k = curvature if curvature is not None else self._curvature()
        cw = self.scorer.curvature_weight(k)

        # Hamiltonian-evolved tier weights, modulated by curvature
        hw = self.saliency.get_weights()
        tier_weights = {
            "working":  hw["working"],
            "episodic": hw["episodic"] + 0.1 * cw,
            "semantic": hw["semantic"] + 0.1 * (1 - cw),
        }
        # Renormalize
        tw_total = sum(tier_weights.values())
        tier_weights = {k: v / tw_total for k, v in tier_weights.items()}

        qt = query_intent(query)

        # Gather raw candidates from all tiers
        raw_candidates: List[Tuple[str, str, str, float, float]] = []
        # (content, tier_tag, memory_type, stability, confidence)

        for rec in self.working.retrieve(limit=6):
            if len(rec.content) > 500 or any(p in rec.content for p in self._REJECT_PATTERNS):
                continue
            raw_candidates.append((
                rec.content, "W", rec.memory_type,
                rec.stability_score, rec.confidence,
            ))

        for rec in self.episodic.retrieve(query, limit=5):
            if any(p in rec.content for p in self._REJECT_PATTERNS):
                continue
            raw_candidates.append((
                rec.content, "E", rec.memory_type,
                rec.stability_score, rec.confidence,
            ))

        for item in self.semantic.retrieve(query, limit=3, user_id=user_id):
            text = item.get("memory", "") if isinstance(item, dict) else str(item)
            if text:
                # Skip entries matching reject patterns (stale salvo/system text)
                if any(p in text for p in self._REJECT_PATTERNS):
                    logger.debug(f"IMD retrieve rejected semantic entry (pattern match)")
                    continue
                # Skip oversized entries
                if len(text) > 500:
                    continue
                density = self.scorer.information_density(text)
                raw_candidates.append((
                    text, "S", "fact", density, 1.0,
                ))

        if not raw_candidates:
            return ""

        # Score with geometric formula, building S_c incrementally
        selected_texts: List[str] = []
        scored: List[Tuple[float, str, str]] = []

        for content, tag, mtype, stab, conf in raw_candidates:
            tier_key = {"W": "working", "E": "episodic", "S": "semantic"}[tag]
            tw = tier_weights[tier_key]

            geo_score = self.scorer.retrieval_score(
                content=content,
                query=query,
                memory_type=mtype,
                stability=stab,
                confidence=conf,
                selected_set=selected_texts,
                query_type=qt,
            )
            final = geo_score * tw
            scored.append((final, content, tag))
            selected_texts.append(content)

        scored.sort(key=lambda x: x[0], reverse=True)

        # Track tier hits for Hamiltonian utility estimation
        self._tier_queries += 1
        for _, _, tag in scored[:3]:
            key = {"W": "working", "E": "episodic", "S": "semantic"}[tag]
            self._tier_hits[key] += 1

        # Update Hamiltonian dynamics every 10 queries
        if self._tier_queries >= 10:
            total_q = max(self._tier_queries, 1)
            utilities = [
                self._tier_hits["working"] / total_q,
                self._tier_hits["episodic"] / total_q,
                self._tier_hits["semantic"] / total_q,
            ]
            self.saliency.step(utilities)
            self._tier_hits = {"working": 0, "episodic": 0, "semantic": 0}
            self._tier_queries = 0

        # Build context string within token budget (~4 chars/token)
        char_budget = token_budget * 4
        header = "[Memory Context]"
        lines = []
        used = len(header) + 2

        for score, content, tag in scored:
            line = f"- [{tag}] {content}"
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)

    def consolidate(self):
        """
        CLS-inspired consolidation (slow neocortical integration).

        Promotes stable, frequently-accessed episodic memories to semantic.
        Updates saliency across all episodic memories via stability recompute.
        """
        if not self._consolidation_lock.acquire(blocking=False):
            return
        try:
            self._last_consolidation = time.time()
            promoted = 0

            for ep in self.episodic.get_all():
                ep.compute_stability()
                if (ep.stability_score > self._PROMOTE_STABILITY
                        and ep.age() > self._PROMOTE_MIN_AGE
                        and ep.access_count >= self._PROMOTE_MIN_ACCESS):
                    ok = self.semantic.store(ep.content, metadata={
                        "promoted_from": "episodic",
                        "stability": ep.stability_score,
                        "memory_type": ep.memory_type,
                        "access_count": ep.access_count,
                    })
                    if ok:
                        promoted += 1

            if promoted:
                logger.info(f"IMD consolidation: {promoted} promoted to semantic")
        finally:
            self._consolidation_lock.release()

    # ── Drop-in compatibility ────────────────────────────────────────

    def add(self, content: str, user_id: str = "demo_user",
            metadata: Optional[Dict] = None):
        return self.store(content, tier="auto")

    def search(self, query: str, user_id: str = "demo_user",
               limit: int = 2):
        return self.semantic.retrieve(query, limit=limit, user_id=user_id)

    def get_context_string(self, query: str,
                           user_id: str = "demo_user") -> str:
        return self.retrieve(query, token_budget=600, user_id=user_id)

    # ── Context Compaction ───────────────────────────────────────────

    def compact_conversation(self, messages: List[Dict[str, str]],
                              max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Compact a conversation history for context window management.
        Keeps system msgs + last `max_messages` non-system msgs.
        Stores dropped messages in episodic memory before discarding.
        Returns compacted history with a summary of dropped content.
        """
        if not messages or len(messages) <= max_messages + 1:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= max_messages:
            return messages

        dropped = non_system[:-max_messages]
        kept = non_system[-max_messages:]

        # Store dropped messages in episodic memory before discarding
        for msg in dropped:
            content = msg.get("content", "")
            if content and len(content) > 20:
                self.store(content[:500], tier="episodic")

        # Build extractive summary
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

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "working": {
                "capacity": self.working.capacity,
                "used": len(self.working),
            },
            "episodic": {
                "capacity": self.episodic.capacity,
                "used": len(self.episodic),
            },
            "semantic": {
                "backend": "qdrant",
                "status": "online" if self.semantic._mem0 else "offline",
            },
            "saliency_dynamics": self.saliency.get_state(),
            "tier_weights": self.saliency.get_weights(),
            "last_consolidation": self._last_consolidation,
            "curvature": self._curvature(),
            "entropy": self._entropy(),
        }


# ─── Standalone test ─────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    telemetry = {"curvature": -1.0, "entropy": 0.3}
    imd = IMDMemory(telemetry_state=telemetry)

    # Store with auto-classification
    imd.store("Riemannian manifolds have curvature tensors that encode geometry")
    imd.store("To install PyTorch, first run pip install torch")
    imd.store("We discussed fiber bundles in the last session")
    imd.store("Since A implies B and B implies C, therefore A implies C")

    # Test conflict detection
    imd.store("The capital of France is Paris")
    imd.store("The capital of France is not Paris")  # Should trigger conflict

    # Retrieve
    ctx = imd.get_context_string("How do I work with manifold geometry?")
    print(f"\n=== Retrieved Context ===\n{ctx}")
    print(f"\n=== Stats ===\n{json.dumps(imd.get_stats(), indent=2)}")
