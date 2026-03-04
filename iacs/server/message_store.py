"""IACS SQLite Message Store — WAL mode, TTL sweep, dead letter queue."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import Message


class MessageStore:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "messages.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                type TEXT NOT NULL,
                priority INTEGER DEFAULT 2,
                ttl_seconds INTEGER DEFAULT 300,
                correlation_id TEXT,
                payload TEXT NOT NULL,
                version TEXT DEFAULT '1.0.0',
                acked INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_recipient ON messages(recipient);
            CREATE INDEX IF NOT EXISTS idx_msg_type ON messages(type);
            CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at);

            CREATE TABLE IF NOT EXISTS dead_letters (
                id TEXT PRIMARY KEY,
                original_message TEXT NOT NULL,
                reason TEXT NOT NULL,
                failed_at TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()

    def store_message(self, msg: Message) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO messages
               (id, timestamp, sender, recipient, type, priority,
                ttl_seconds, correlation_id, payload, version,
                acked, retry_count, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                msg.id, msg.timestamp, msg.sender, msg.recipient,
                msg.type.value if hasattr(msg.type, 'value') else msg.type,
                msg.priority.value if hasattr(msg.priority, 'value') else msg.priority,
                msg.ttl_seconds, msg.correlation_id,
                json.dumps(msg.payload), msg.version,
                int(msg.acked), msg.retry_count, time.time()
            )
        )
        self.conn.commit()

    def get_messages(
        self,
        recipient: Optional[str] = None,
        type_filter: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM messages WHERE 1=1"
        params: list = []
        if recipient:
            query += " AND (recipient = ? OR recipient = '__broadcast__')"
            params.append(recipient)
        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)
        if since:
            query += " AND timestamp > ?"
            params.append(since)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def ack_message(self, message_id: str) -> bool:
        cur = self.conn.execute(
            "UPDATE messages SET acked = 1 WHERE id = ?", (message_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def sweep_expired(self) -> int:
        now = time.time()
        # Find expired unacked messages within retry limit
        expired = self.conn.execute(
            """SELECT * FROM messages
               WHERE ttl_seconds > 0
                 AND acked = 0
                 AND (created_at + ttl_seconds) < ?""",
            (now,)
        ).fetchall()

        moved_to_dl = 0
        for row in expired:
            d = dict(row)
            if d["retry_count"] >= 3:
                # Move to dead letter
                self.conn.execute(
                    """INSERT OR REPLACE INTO dead_letters
                       (id, original_message, reason, failed_at, retry_count)
                       VALUES (?,?,?,?,?)""",
                    (
                        d["id"],
                        json.dumps(self._row_to_dict_raw(row)),
                        "TTL expired after max retries",
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        d["retry_count"]
                    )
                )
                moved_to_dl += 1

        # Delete all expired
        cur = self.conn.execute(
            """DELETE FROM messages
               WHERE ttl_seconds > 0
                 AND acked = 0
                 AND (created_at + ttl_seconds) < ?""",
            (now,)
        )
        self.conn.commit()
        return cur.rowcount

    def get_dead_letters(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM dead_letters ORDER BY failed_at DESC LIMIT 100"
        ).fetchall()
        return [dict(r) for r in rows]

    def retry_dead_letter(self, message_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM dead_letters WHERE id = ?", (message_id,)
        ).fetchone()
        if not row:
            return None
        original = json.loads(row["original_message"])
        original["retry_count"] = row["retry_count"] + 1
        original["acked"] = False
        msg = Message(**original)
        msg.id = str(__import__("uuid").uuid4())
        self.store_message(msg)
        self.conn.execute("DELETE FROM dead_letters WHERE id = ?", (message_id,))
        self.conn.commit()
        return msg.model_dump()

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        total = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        last_min = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE created_at > ?", (now - 60,)
        ).fetchone()[0]
        dead = self.conn.execute("SELECT COUNT(*) FROM dead_letters").fetchone()[0]

        by_type = {}
        for row in self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM messages GROUP BY type"
        ).fetchall():
            by_type[row["type"]] = row["cnt"]

        by_agent = {}
        for row in self.conn.execute(
            "SELECT sender, COUNT(*) as cnt FROM messages GROUP BY sender"
        ).fetchall():
            by_agent[row["sender"]] = row["cnt"]

        return {
            "total_messages": total,
            "messages_last_minute": last_min,
            "dead_letters": dead,
            "messages_by_type": by_type,
            "messages_by_agent": by_agent,
        }

    def _row_to_dict(self, row) -> Dict[str, Any]:
        d = dict(row)
        d["payload"] = json.loads(d["payload"]) if isinstance(d["payload"], str) else d["payload"]
        d["acked"] = bool(d["acked"])
        return d

    def _row_to_dict_raw(self, row) -> Dict[str, Any]:
        return dict(row)

    def close(self):
        self.conn.close()
