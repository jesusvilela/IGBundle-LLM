"""
ModelState — Persistent, evolvable model configuration.
========================================================
Single source of truth for generation params, checkpoint info,
telemetry snapshots, and salvo history. Both app_neural_glass.py
and run_salvos_v2.py load from and write to memory/model_state.json.
"""

import os
import re
import json
import time
import copy
import tempfile

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")
STATE_FILE = os.path.join(STATE_DIR, "model_state.json")

DEFAULTS = {
    "generation": {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.15,
    },
    "checkpoint": {
        "path": "",
        "step": 0,
        "adapter_type": "geometric",
    },
    "system_prompt": (
        "Answer directly with step-by-step reasoning. Be concise."
    ),
    "telemetry_snapshot": {},
    "evolution": {
        "generation_count": 0,
        "best_score": 0.0,
        "best_params": {},
    },
    "salvo_history": [],
}

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "igbundle_odyssey_v3",
)


class ModelState:
    """Persistent, evolvable model configuration."""

    def __init__(self, state_file=None):
        self.state_file = state_file or STATE_FILE
        self._state = None

    # ── Load / Save ──────────────────────────────────────────────────

    def load(self) -> dict:
        """Load state from disk, merge with defaults for missing keys."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                disk = json.load(f)
        else:
            disk = {}

        # Deep-merge defaults (only fill missing keys, never overwrite existing)
        state = copy.deepcopy(DEFAULTS)
        for key, default_val in DEFAULTS.items():
            if key in disk:
                if isinstance(default_val, dict) and isinstance(disk[key], dict):
                    state[key] = {**default_val, **disk[key]}
                else:
                    state[key] = disk[key]

        # Auto-resolve checkpoint if not set
        if not state["checkpoint"]["path"]:
            ckpt_path, step = self.find_latest_checkpoint()
            if ckpt_path:
                state["checkpoint"]["path"] = ckpt_path
                state["checkpoint"]["step"] = step

        self._state = state
        return state

    def save(self, state=None):
        """Atomic write to disk."""
        if state is not None:
            self._state = state
        if self._state is None:
            return

        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        # Write to temp file first, then rename (atomic on same filesystem)
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.state_file), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
            # On Windows, must remove target first if it exists
            if os.path.exists(self.state_file):
                os.replace(tmp_path, self.state_file)
            else:
                os.rename(tmp_path, self.state_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @property
    def state(self) -> dict:
        if self._state is None:
            self.load()
        return self._state

    # ── Generation Params ────────────────────────────────────────────

    @property
    def generation_params(self) -> dict:
        """Return generation hyperparameters dict."""
        return dict(self.state["generation"])

    # ── Checkpoint Resolution ────────────────────────────────────────

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir=None) -> tuple:
        """
        Scan checkpoint-* dirs, return (path_to_adapter_weights, step) for latest.
        Returns (None, 0) if no checkpoints found.
        """
        ckpt_dir = checkpoint_dir or CHECKPOINT_DIR
        if not os.path.isdir(ckpt_dir):
            return None, 0

        pattern = re.compile(r"checkpoint-(?:\w+-)?(\d+)$")
        candidates = []
        for name in os.listdir(ckpt_dir):
            m = pattern.match(name)
            if m:
                step = int(m.group(1))
                weights = os.path.join(ckpt_dir, name, "adapter_weights.pt")
                if os.path.exists(weights):
                    candidates.append((step, weights))

        if not candidates:
            return None, 0

        candidates.sort(key=lambda x: x[0])
        best_step, best_path = candidates[-1]
        return best_path, best_step

    # ── Evolution ────────────────────────────────────────────────────

    def evolve(self, mean_score: float, params: dict, run_detail: dict = None):
        """
        Compare run score with best. If improved, update generation params.
        Always appends to salvo_history.

        Args:
            mean_score: Mean score across all salvo questions
            params: The generation params used for this run
            run_detail: Optional dict with per-salvo scores, checkpoint info, etc.
        """
        state = self.state
        evo = state["evolution"]

        # Build history entry
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": dict(params),
            "mean_score": round(mean_score, 4),
            "generation": evo["generation_count"] + 1,
        }
        if run_detail:
            entry.update(run_detail)

        state["salvo_history"].append(entry)
        evo["generation_count"] += 1

        # Evolution: update if improved
        if mean_score > evo["best_score"]:
            print(f"[ModelState] Score improved: {evo['best_score']:.4f} -> {mean_score:.4f}")
            evo["best_score"] = round(mean_score, 4)
            evo["best_params"] = dict(params)
            # Promote to active generation params
            state["generation"].update(params)
        else:
            delta = evo["best_score"] - mean_score
            print(f"[ModelState] Score {mean_score:.4f} did not beat best {evo['best_score']:.4f} (delta={delta:.4f})")

        self.save()
        return evo["best_score"]

    # ── Telemetry Snapshot ───────────────────────────────────────────

    def snapshot_telemetry(self, telemetry_dict: dict):
        """Save telemetry state for restoration on restart."""
        import numpy as _np

        def _to_json_safe(val):
            """Convert numpy types to JSON-serializable Python types."""
            if isinstance(val, _np.ndarray):
                return val.tolist()
            if isinstance(val, (_np.integer,)):
                return int(val)
            if isinstance(val, (_np.floating,)):
                return float(val)
            return val

        # Only persist serializable scalar/list values
        snap = {}
        for k, v in telemetry_dict.items():
            v = _to_json_safe(v)
            if isinstance(v, (int, float, str, bool)):
                snap[k] = v
            elif isinstance(v, list):
                # Keep last 50 entries max for history arrays
                safe_list = [_to_json_safe(x) for x in (v[-50:] if len(v) > 50 else v)]
                snap[k] = safe_list
        self.state["telemetry_snapshot"] = snap
        self.save()

    def restore_telemetry(self) -> dict:
        """Return saved telemetry snapshot (or empty dict)."""
        return dict(self.state.get("telemetry_snapshot", {}))

    # ── Utility ──────────────────────────────────────────────────────

    def record_training_event(self, event_type: str, step: int, details: dict = None):
        """
        Log a training event (halt, lr_reduce, fiber_warning, checkpoint, etc.).

        Args:
            event_type: One of 'halt', 'lr_reduce', 'fiber_critical',
                        'fiber_warning', 'checkpoint', 'training_start', 'training_end'
            step: Training step when the event occurred
            details: Optional dict with additional context
        """
        state = self.state
        if "training_events" not in state:
            state["training_events"] = []

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event_type,
            "step": step,
        }
        if details:
            entry["details"] = details

        state["training_events"].append(entry)
        # Keep last 200 events max
        if len(state["training_events"]) > 200:
            state["training_events"] = state["training_events"][-200:]

        self.save()

    def update_checkpoint(self, path: str, step: int):
        """Update the active checkpoint."""
        self.state["checkpoint"]["path"] = path
        self.state["checkpoint"]["step"] = step
        self.save()

    def summary(self) -> str:
        """One-line summary for logging."""
        s = self.state
        evo = s["evolution"]
        gen = s["generation"]
        ckpt = s["checkpoint"]
        return (
            f"ModelState: cp-{ckpt['step']} | "
            f"T={gen['temperature']} P={gen['top_p']} K={gen['top_k']} R={gen['repetition_penalty']} | "
            f"best_score={evo['best_score']:.3f} | gen#{evo['generation_count']}"
        )


# ── Standalone test ──────────────────────────────────────────────────

if __name__ == "__main__":
    ms = ModelState()
    state = ms.load()
    print(ms.summary())
    print(f"State file: {ms.state_file}")
    print(f"Checkpoint: {state['checkpoint']}")
    print(f"Generation: {state['generation']}")
    print(f"Evolution: {state['evolution']}")
    ms.save()
    print(f"Saved to {ms.state_file}")
