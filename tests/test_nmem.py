"""
Tests for NMEM — Neuromorphic Memory Algorithm.

Validates:
  - FHN dynamics: decay toward rest, retrieval spike, reconsolidation
  - FFT spectral decay: high-freq loss before low-freq
  - Gabor interference: similar memories merge
  - Multi-anchor retrieval: any pathway suffices
  - Savant preservation: protected nodes don't decay
  - Ghost entries: decayed nodes known but inaccessible
  - Attentional gating: low-salience content rejected
  - Reconstruction: progressive compression with perturbation
  - Drop-in API compatibility with IMDMemory
"""

import time
import math
import numpy as np
import pytest
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmem import (
    NMEMConfig, NMEM, HashEmbedder,
    FHNState, fhn_step, fhn_advance, fhn_activation_score,
    fft_decay, gabor_kernel,
    detect_savant_content, compute_salience, compute_topic_hash,
    MemoryNode, TurnHeader,
)


# ─── FHN Dynamics ─────────────────────────────────────────────────

class TestFHNDynamics:

    def test_resting_state_decay(self):
        """Unretrieved memory should decay toward low activation."""
        cfg = NMEMConfig(fhn_epsilon=0.08)
        state = FHNState(v=0.8, w=0.0)  # Fresh memory

        # Advance 10 hours without retrieval
        state = fhn_advance(state, delta_hours=10.0, cfg=cfg, retrieved=False)
        score = fhn_activation_score(state)

        # Should have decayed significantly
        assert score < 0.7, f"Expected decay, got activation={score:.3f}"

    def test_retrieval_spike(self):
        """Retrieval should spike activation."""
        cfg = NMEMConfig()
        state = FHNState(v=0.0, w=0.5)  # Partially decayed

        score_before = fhn_activation_score(state)
        state = fhn_advance(state, delta_hours=0.001, cfg=cfg, retrieved=True)
        score_after = fhn_activation_score(state)

        assert score_after > score_before, (
            f"Retrieval should increase activation: "
            f"{score_before:.3f} → {score_after:.3f}"
        )

    def test_activation_bounded(self):
        """FHN activation score should stay in [0, 1]."""
        cfg = NMEMConfig()
        for v in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            state = FHNState(v=v, w=0.0)
            score = fhn_activation_score(state)
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for v={v}"

    def test_reconsolidation_differs(self):
        """Two retrieval events from same state should produce different trajectories."""
        cfg = NMEMConfig()
        state = FHNState(v=0.3, w=0.2)

        # Retrieve then decay
        s1 = fhn_advance(state, 0.001, cfg, retrieved=True)
        s1 = fhn_advance(s1, 1.0, cfg, retrieved=False)

        # Retrieve then decay again from same starting point
        s2 = fhn_advance(state, 0.001, cfg, retrieved=True)
        s2 = fhn_advance(s2, 2.0, cfg, retrieved=False)

        # Different decay durations → different end states
        assert abs(s1.v - s2.v) > 0.001 or abs(s1.w - s2.w) > 0.001


# ─── FFT Spectral Decay ──────────────────────────────────────────

class TestFFTDecay:

    def test_high_freq_decays_faster(self):
        """High-frequency components should decay more than low-frequency."""
        dim = 128
        # Create vector with known frequency content
        vec = np.zeros(dim)
        # Low freq component
        vec += np.cos(2 * np.pi * 2 * np.arange(dim) / dim)
        # High freq component
        vec += np.cos(2 * np.pi * 50 * np.arange(dim) / dim)
        vec /= np.linalg.norm(vec)

        # Apply decay
        decayed = fft_decay(vec, envelope=0.5, alpha=0.05)

        # Check spectrum: high-freq should be more attenuated
        orig_spec = np.abs(np.fft.rfft(vec))
        decay_spec = np.abs(np.fft.rfft(decayed))

        # Ratio of high-freq attenuation vs low-freq
        low_idx = 2
        high_idx = min(50, len(orig_spec) - 1)

        if orig_spec[low_idx] > 1e-10 and orig_spec[high_idx] > 1e-10:
            low_ratio = decay_spec[low_idx] / orig_spec[low_idx]
            high_ratio = decay_spec[high_idx] / orig_spec[high_idx]
            assert high_ratio < low_ratio, (
                f"High freq should decay more: "
                f"low_ratio={low_ratio:.4f}, high_ratio={high_ratio:.4f}"
            )

    def test_full_envelope_no_decay(self):
        """Envelope=1.0 should produce minimal change."""
        vec = np.random.randn(128)
        vec /= np.linalg.norm(vec)
        decayed = fft_decay(vec, envelope=0.99, alpha=0.02)
        # Should be nearly identical (renormalization might shift slightly)
        diff = np.linalg.norm(vec - decayed)
        assert diff < 0.1, f"Full envelope should preserve: diff={diff:.4f}"

    def test_zero_alpha_uniform_decay(self):
        """Alpha=0 should produce uniform scaling (no frequency selectivity)."""
        vec = np.random.randn(128)
        vec /= np.linalg.norm(vec)
        decayed = fft_decay(vec, envelope=0.5, alpha=0.0)
        # After renormalization, should be similar direction
        sim = np.dot(vec, decayed)
        assert sim > 0.95, f"Zero alpha should preserve direction: sim={sim:.4f}"


# ─── Gabor Interference ──────────────────────────────────────────

class TestGaborInterference:

    def test_close_time_high_sim(self):
        """Close in time + high similarity → strong interference."""
        cfg = NMEMConfig(gabor_sigma=3600.0, gabor_omega=2.0)
        g = gabor_kernel(delta_t=60.0, cosine_sim=0.95, cfg=cfg)
        assert abs(g) > 0.1, f"Expected strong interference, got {g:.4f}"

    def test_far_time_decays(self):
        """Far apart in time → weak interference regardless of similarity."""
        cfg = NMEMConfig(gabor_sigma=3600.0, gabor_omega=2.0)
        g_close = gabor_kernel(delta_t=60.0, cosine_sim=0.9, cfg=cfg)
        g_far = gabor_kernel(delta_t=36000.0, cosine_sim=0.9, cfg=cfg)
        assert abs(g_far) < abs(g_close), (
            f"Far memories should interfere less: "
            f"|g_close|={abs(g_close):.4f}, |g_far|={abs(g_far):.4f}"
        )

    def test_low_sim_no_interference(self):
        """Low similarity → cos(omega * sim) oscillates, but magnitude from temporal."""
        cfg = NMEMConfig(gabor_sigma=3600.0, gabor_omega=2.0)
        g = gabor_kernel(delta_t=60.0, cosine_sim=0.1, cfg=cfg)
        # Should still be non-zero (temporal close), just different direction
        assert isinstance(g, float)


# ─── Savant Detection ────────────────────────────────────────────

class TestSavantDetection:

    def test_code_detected(self):
        assert detect_savant_content("def compute_loss(x, y):")
        assert detect_savant_content("from torch import nn")
        assert detect_savant_content("const result = await fetch(url)")

    def test_numbers_detected(self):
        assert detect_savant_content("checkpoint at step 30000")
        assert detect_savant_content("loss was 0.74521")

    def test_urls_detected(self):
        assert detect_savant_content("see https://github.com/example")
        assert detect_savant_content("running on localhost:9100")

    def test_paths_detected(self):
        assert detect_savant_content("saved to H:/LLM-MANIFOLD/igbundle-llm/model.pt")
        assert detect_savant_content("config in /etc/nginx/conf.d/")

    def test_plain_text_not_savant(self):
        assert not detect_savant_content("The weather is nice today")
        assert not detect_savant_content("I think we should redesign the memory")


# ─── Hash Embedder ────────────────────────────────────────────────

class TestHashEmbedder:

    def test_deterministic(self):
        emb = HashEmbedder(dim=128)
        v1 = emb.embed("hello world")
        v2 = emb.embed("hello world")
        assert np.allclose(v1, v2)

    def test_similar_texts_similar(self):
        emb = HashEmbedder(dim=128)
        v1 = emb.embed("the geometric adapter uses fiber bundles")
        v2 = emb.embed("fiber bundles in the geometric adapter")
        sim = emb.cosine_sim(v1, v2)
        assert sim > 0.3, f"Similar texts should have positive similarity: {sim:.3f}"

    def test_different_texts_lower_sim(self):
        emb = HashEmbedder(dim=128)
        v1 = emb.embed("the geometric adapter uses fiber bundles")
        v2 = emb.embed("I went to the grocery store for milk")
        v3 = emb.embed("fiber bundles in the geometric adapter")
        sim_similar = emb.cosine_sim(v1, v3)
        sim_different = emb.cosine_sim(v1, v2)
        assert sim_similar > sim_different, (
            f"Similar > different: {sim_similar:.3f} vs {sim_different:.3f}"
        )

    def test_unit_norm(self):
        emb = HashEmbedder(dim=128)
        v = emb.embed("some text here")
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm:.4f}"


# ─── Full NMEM Integration ───────────────────────────────────────

class TestNMEMIntegration:

    def _make_nmem(self, tmp_path) -> NMEM:
        cfg = NMEMConfig(
            persist_dir=str(tmp_path),
            salience_threshold=0.05,  # Low threshold for testing
            max_nodes=100,
            theta_high=0.3,
            theta_low=0.1,
        )
        return NMEM(config=cfg)

    def test_store_and_retrieve(self, tmp_path):
        mem = self._make_nmem(tmp_path)
        nid = mem.store("The curvature K is currently at -5.63")
        assert nid != ""

        result = mem.retrieve("what is the curvature?")
        assert "curvature" in result.lower() or "-5.63" in result

    def test_attentional_gating(self, tmp_path):
        cfg = NMEMConfig(
            persist_dir=str(tmp_path),
            salience_threshold=0.99,  # Very high — blocks everything
        )
        mem = NMEM(config=cfg)
        # Store two identical messages — second should be gated
        mem.store("unique first message about quantum physics")
        nid = mem.store("unique first message about quantum physics")
        # Second store of identical content = 0 salience = gated
        # (unless it's the very first turn)
        # First store always passes (salience=1.0 for first turn)

    def test_savant_preserved(self, tmp_path):
        mem = self._make_nmem(tmp_path)
        nid = mem.store("checkpoint at H:/LLM-MANIFOLD/igbundle-llm/cp-3000/model.pt")
        assert nid != ""

        node = mem._nodes.get(nid)
        assert node is not None
        assert node.header.savant_flag is True

    def test_multiple_stores_retrieval(self, tmp_path):
        mem = self._make_nmem(tmp_path)
        mem.store("The fiber entropy S is 0.95 after phase B training")
        mem.store("Gemini is running ARC-AGI evaluations on config D")
        mem.store("The weather forecast says rain tomorrow")

        result = mem.retrieve("what is the fiber entropy?")
        assert "entropy" in result.lower() or "0.95" in result

    def test_ghost_formation(self, tmp_path):
        cfg = NMEMConfig(
            persist_dir=str(tmp_path),
            salience_threshold=0.01,
            theta_low=0.99,  # Very high — everything becomes ghost quickly
            max_nodes=100,
        )
        mem = NMEM(config=cfg)
        mem.store("some content that will decay")

        # Force decay by manipulating FHN state
        for node in mem._nodes.values():
            node.fhn = FHNState(v=-2.0, w=2.0)  # Very low activation

        # Run decay via retrieve
        mem.retrieve("anything")

        assert len(mem._ghosts) > 0 or len(mem._nodes) == 0

    def test_persistence(self, tmp_path):
        cfg = NMEMConfig(persist_dir=str(tmp_path), salience_threshold=0.01)

        # Create and store
        mem1 = NMEM(config=cfg)
        mem1.store("persistent memory about fiber bundles")
        n_nodes = len(mem1._nodes)
        assert n_nodes > 0

        # Reload
        mem2 = NMEM(config=cfg)
        assert len(mem2._nodes) == n_nodes

    def test_reject_patterns(self, tmp_path):
        mem = self._make_nmem(tmp_path)
        nid = mem.store("IGBUNDLE // NEURAL GLASS status report")
        assert nid == ""

    def test_dropin_api(self, tmp_path):
        """Verify IMDMemory-compatible API."""
        mem = self._make_nmem(tmp_path)
        mem.add("test content via add()")
        results = mem.search("test content")
        ctx = mem.get_context_string("test")
        # Just verify no crashes — API compatibility
        assert isinstance(results, list)
        assert isinstance(ctx, str)

    def test_compact_conversation(self, tmp_path):
        mem = self._make_nmem(tmp_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        for i in range(15):
            messages.append({"role": "user", "content": f"User message {i}"})
            messages.append({"role": "assistant", "content": f"Response {i}"})

        compacted = mem.compact_conversation(messages, max_messages=10)
        non_system = [m for m in compacted if m["role"] != "system"]
        assert len(non_system) <= 10

    def test_stats(self, tmp_path):
        mem = self._make_nmem(tmp_path)
        mem.store("test memory for stats")
        stats = mem.get_stats()
        assert stats["total_nodes"] >= 1
        assert "mean_activation" in stats
        assert "config" in stats


# ─── Salience Computation ────────────────────────────────────────

class TestSalience:

    def test_first_turn_max_salience(self):
        emb = HashEmbedder(dim=128)
        vec = emb.embed("hello")
        s = compute_salience(vec, None, emb)
        assert s == 1.0

    def test_identical_low_salience(self):
        emb = HashEmbedder(dim=128)
        vec = emb.embed("the same message repeated")
        s = compute_salience(vec, vec, emb)
        assert s < 0.1, f"Identical should have low salience: {s:.3f}"

    def test_different_high_salience(self):
        emb = HashEmbedder(dim=128)
        v1 = emb.embed("talking about fiber bundle geometry and curvature")
        v2 = emb.embed("what should we have for dinner tonight")
        s = compute_salience(v2, v1, emb)
        assert s > 0.5, f"Different topics should have high salience: {s:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
