"""
IACS v2 Offline Client — file-based messaging that works WITHOUT the server.

When the IACS server (localhost:9100) is down, agents can still exchange messages
by reading/writing JSON files directly. Messages written offline are automatically
picked up by the server when it restarts.

Usage:
    from iacs.client.offline_client import OfflineClient

    iacs = OfflineClient("claude_code")
    iacs.send_finding("Step 200 reached, S=1.15")
    msgs = iacs.get_messages_for_me()
    iacs.send_task("gemini", "Update dashboard for file-based store")
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any


def _default_data_dir() -> Path:
    return Path(__file__).parent.parent / "data"


class OfflineClient:
    """File-based IACS client — no server dependency."""

    def __init__(self, agent_id: str, data_dir: Optional[str] = None):
        self.agent_id = agent_id
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir()
        self.msg_dir = self.data_dir / "messages"
        self.offline_dir = self.data_dir / "offline-queue"

        # Ensure dirs exist
        self.msg_dir.mkdir(parents=True, exist_ok=True)
        self.offline_dir.mkdir(parents=True, exist_ok=True)

    # ── Sending ──────────────────────────────────────────────────────

    def send_message(
        self,
        recipient: str,
        msg_type: str,
        payload: Dict[str, Any],
        priority: int = 2,
        ttl: int = 0,  # 0 = no expiry (persistent by default in v2)
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write a message as a JSON file. Returns the message dict."""
        msg_id = str(uuid.uuid4())
        created_at = time.time()
        ts_str = datetime.now(timezone.utc).isoformat()

        data = {
            "id": msg_id,
            "timestamp": ts_str,
            "sender": self.agent_id,
            "recipient": recipient,
            "type": msg_type,
            "priority": priority,
            "ttl_seconds": ttl,
            "correlation_id": correlation_id,
            "payload": payload,
            "version": "2.0.0",
            "acked": False,
            "retry_count": 0,
            "created_at": created_at,
        }

        # Write to messages dir (server will pick up on index rebuild)
        ts_file = time.strftime("%Y%m%dT%H%M%S", time.gmtime(created_at))
        filename = f"{ts_file}_{msg_id[:8]}.json"
        filepath = self.msg_dir / filename

        # Atomic write
        tmp = filepath.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.rename(filepath)

        return data

    def send_finding(self, content: str, recipient: str = "__broadcast__", priority: int = 2):
        return self.send_message(recipient, "finding", {"content": content}, priority)

    def send_task(self, recipient: str, description: str, context: Optional[Dict] = None, priority: int = 2):
        payload = {"description": description}
        if context:
            payload["context"] = context
        return self.send_message(recipient, "task_assignment", payload, priority)

    def send_question(self, recipient: str, question: str, context: Optional[Dict] = None):
        payload = {"question": question}
        if context:
            payload["context"] = context
        return self.send_message(recipient, "question", payload)

    def send_answer(self, recipient: str, answer: str, correlation_id: Optional[str] = None):
        return self.send_message(recipient, "answer", {"answer": answer}, correlation_id=correlation_id)

    def send_status(self, status: str):
        return self.send_message("__broadcast__", "status_update", {"status": status})

    # ── Receiving ────────────────────────────────────────────────────

    def get_messages_for_me(
        self,
        since: Optional[str] = None,
        type_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Read all messages addressed to this agent or broadcast."""
        results = []
        files = sorted(self.msg_dir.glob("*.json"), reverse=True)
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            recip = data.get("recipient", "")
            if recip != self.agent_id and recip != "__broadcast__":
                continue
            if type_filter and data.get("type") != type_filter:
                continue
            if since and data.get("timestamp", "") <= since:
                continue
            results.append(data)
            if len(results) >= limit:
                break
        return results

    def get_all_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read all messages (for dashboard/debugging)."""
        results = []
        files = sorted(self.msg_dir.glob("*.json"), reverse=True)
        for f in files[:limit]:
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def ack_message(self, message_id: str) -> bool:
        """Mark a message as acknowledged by updating its file."""
        for f in self.msg_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("id") == message_id:
                    data["acked"] = True
                    f.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                    return True
            except (json.JSONDecodeError, OSError):
                continue
        return False

    # ── Convenience ──────────────────────────────────────────────────

    def get_unread(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get unacked messages for this agent."""
        msgs = self.get_messages_for_me(limit=limit)
        return [m for m in msgs if not m.get("acked", False)]

    def print_inbox(self):
        """Pretty-print unread messages."""
        msgs = self.get_unread()
        if not msgs:
            print(f"[{self.agent_id}] No unread messages.")
            return
        print(f"[{self.agent_id}] {len(msgs)} unread message(s):\n")
        for m in msgs:
            sender = m.get("sender", "?")
            ts = m.get("timestamp", "?")[:16]
            mtype = m.get("type", "?")
            body = str(m.get("payload", {}).get("content", "")
                       or m.get("payload", {}).get("answer", "")
                       or m.get("payload", {}).get("description", "")
                       or m.get("payload", {}).get("question", ""))[:120]
            print(f"  {ts} | {sender} | {mtype}")
            print(f"    {body}")
            print()
