"""IACS v2 File-Based Message Store — drop-in replacement for SQLite MessageStore.

Messages stored as individual JSON files in iacs/data/messages/.
No database, no locking, no corruption. Survives server crashes.
Both agents can read files directly when server is down (offline mode).
"""

import json
import os
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import Message


class FileMessageStore:
    """File-based message store with same interface as SQLite MessageStore."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / "data")
        self.data_dir = Path(data_dir)
        self.msg_dir = self.data_dir / "messages"
        self.dl_dir = self.data_dir / "dead-letters"
        self.offline_dir = self.data_dir / "offline-queue"
        self.telemetry_path = self.data_dir / "telemetry.jsonl"
        self.agents_path = self.data_dir / "agents.json"

        # Create directories
        for d in [self.msg_dir, self.dl_dir, self.offline_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast queries (rebuilt from files on init)
        self._index: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._rebuild_index()

    # ── Index Management ─────────────────────────────────────────────

    def _rebuild_index(self):
        """Scan message files and build in-memory index."""
        with self._lock:
            self._index = []
            for f in self.msg_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    data["_file"] = str(f)
                    self._index.append(data)
                except (json.JSONDecodeError, OSError):
                    continue
            # Also ingest any offline-queue messages
            for f in self.offline_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    data["_file"] = str(f)
                    # Move to messages dir
                    dest = self.msg_dir / f.name
                    f.rename(dest)
                    data["_file"] = str(dest)
                    self._index.append(data)
                except (json.JSONDecodeError, OSError):
                    continue
            self._index.sort(key=lambda m: m.get("created_at", 0), reverse=True)

    def _msg_filename(self, msg_id: str, created_at: float) -> str:
        """Generate sortable filename: timestamp_uuid.json"""
        ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(created_at))
        short_id = msg_id[:8]
        return f"{ts}_{short_id}.json"

    # ── Store / Retrieve ─────────────────────────────────────────────

    def store_message(self, msg: Message) -> None:
        created_at = time.time()
        data = {
            "id": msg.id,
            "timestamp": msg.timestamp,
            "sender": msg.sender,
            "recipient": msg.recipient,
            "type": msg.type.value if hasattr(msg.type, "value") else msg.type,
            "priority": msg.priority.value if hasattr(msg.priority, "value") else msg.priority,
            "ttl_seconds": msg.ttl_seconds,
            "correlation_id": msg.correlation_id,
            "payload": msg.payload,
            "version": msg.version,
            "acked": msg.acked,
            "retry_count": msg.retry_count,
            "created_at": created_at,
        }

        filename = self._msg_filename(msg.id, created_at)
        filepath = self.msg_dir / filename

        # Atomic write: write to .tmp then rename
        tmp = filepath.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.rename(filepath)

        with self._lock:
            # Remove old version if replacing (same id)
            self._index = [m for m in self._index if m["id"] != msg.id]
            data["_file"] = str(filepath)
            self._index.insert(0, data)

    def get_messages(
        self,
        recipient: Optional[str] = None,
        type_filter: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        unacked_only: bool = False,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for m in self._index:
                if recipient and m["recipient"] != recipient and m["recipient"] != "__broadcast__":
                    continue
                if type_filter and m["type"] != type_filter:
                    continue
                if since and m["timestamp"] <= since:
                    continue
                if unacked_only and m.get("acked", False):
                    continue
                results.append(self._clean(m))
                if len(results) >= limit:
                    break
            return results

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for m in self._index:
                if m["id"] == message_id:
                    return self._clean(m)
        return None

    def ack_message(self, message_id: str) -> bool:
        with self._lock:
            for m in self._index:
                if m["id"] == message_id:
                    m["acked"] = True
                    # Update file on disk
                    self._persist(m)
                    return True
        return False

    def sweep_expired(self) -> int:
        now = time.time()
        expired = []
        with self._lock:
            remaining = []
            for m in self._index:
                ttl = m.get("ttl_seconds", 300)
                created = m.get("created_at", 0)
                acked = m.get("acked", False)
                if ttl > 0 and not acked and (created + ttl) < now:
                    expired.append(m)
                else:
                    remaining.append(m)

            for m in expired:
                if m.get("retry_count", 0) >= 3:
                    self._move_to_dead_letter(m, "TTL expired after max retries")
                # Delete message file
                fpath = Path(m.get("_file", ""))
                if fpath.exists():
                    fpath.unlink()

            self._index = remaining
        return len(expired)

    # ── Dead Letters ─────────────────────────────────────────────────

    def _move_to_dead_letter(self, msg: Dict, reason: str):
        dl = {
            "id": msg["id"],
            "original_message": json.dumps(self._clean(msg)),
            "reason": reason,
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "retry_count": msg.get("retry_count", 0),
        }
        filepath = self.dl_dir / f"{msg['id'][:8]}.json"
        filepath.write_text(json.dumps(dl, indent=2), encoding="utf-8")

    def get_dead_letters(self) -> List[Dict[str, Any]]:
        results = []
        for f in sorted(self.dl_dir.glob("*.json"), reverse=True):
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
            if len(results) >= 100:
                break
        return results

    def retry_dead_letter(self, message_id: str) -> Optional[Dict[str, Any]]:
        for f in self.dl_dir.glob("*.json"):
            try:
                dl = json.loads(f.read_text(encoding="utf-8"))
                if dl["id"] == message_id:
                    original = json.loads(dl["original_message"])
                    original["retry_count"] = dl["retry_count"] + 1
                    original["acked"] = False
                    import uuid as _uuid
                    original["id"] = str(_uuid.uuid4())
                    msg = Message(**original)
                    self.store_message(msg)
                    f.unlink()
                    return msg.model_dump()
            except (json.JSONDecodeError, OSError, Exception):
                continue
        return None

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            total = len(self._index)
            last_min = sum(1 for m in self._index if m.get("created_at", 0) > now - 60)
            dead = len(list(self.dl_dir.glob("*.json")))

            by_type: Dict[str, int] = {}
            by_agent: Dict[str, int] = {}
            for m in self._index:
                t = m.get("type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1
                s = m.get("sender", "unknown")
                by_agent[s] = by_agent.get(s, 0) + 1

        return {
            "total_messages": total,
            "messages_last_minute": last_min,
            "dead_letters": dead,
            "messages_by_type": by_type,
            "messages_by_agent": by_agent,
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _clean(self, m: Dict) -> Dict[str, Any]:
        """Return message dict without internal fields."""
        return {k: v for k, v in m.items() if not k.startswith("_")}

    def _persist(self, m: Dict):
        """Write updated message back to its file."""
        fpath = Path(m.get("_file", ""))
        if fpath.exists():
            data = self._clean(m)
            fpath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def close(self):
        """No-op — no connection to close."""
        pass
