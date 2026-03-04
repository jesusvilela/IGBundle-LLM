"""
IACS Agent Client — Used by both Claude Code and Gemini.

Usage (sync):
    from iacs.client import SyncAgentClient
    iacs = SyncAgentClient("claude_code", "claude")
    iacs.connect()
    iacs.send_finding("Curvature proxy measures off-diagonal elements")
    msgs = iacs.poll_messages()

Usage (async):
    from iacs.client import AgentClient
    client = AgentClient("gemini", "gemini")
    await client.connect()
    await client.send_task("claude_code", "Review riemannian.py fix")
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import aiohttp
import yaml

logger = logging.getLogger("iacs.client")


def _load_server_url() -> str:
    """Auto-discover server URL from protocol_spec.yaml."""
    spec_path = Path(__file__).parent.parent / "protocol_spec.yaml"
    if spec_path.exists():
        with open(spec_path) as f:
            spec = yaml.safe_load(f)
        return spec["protocol"]["server"]["base_url"]
    return "http://localhost:9100"


class AgentClient:
    """Async IACS client with WebSocket support and auto-reconnect."""

    def __init__(
        self,
        agent_id: str,
        agent_type: str = "custom",
        server_url: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.server_url = server_url or _load_server_url()
        self.capabilities = capabilities or [
            "task_assignment", "finding", "question", "answer",
            "status_update", "file_lock", "capability_query",
        ]
        self.session_id: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._handlers: Dict[str, Callable] = {}
        self._reconnect_attempts = 0

    # ── Connection ──────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Negotiate protocol with server."""
        self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self.server_url}/api/v1/negotiate",
                json={
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "protocol_version": "1.0.0",
                    "capabilities": self.capabilities,
                },
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.session_id = data["session_id"]
                    self._reconnect_attempts = 0
                    logger.info(
                        f"Connected as {self.agent_id} (session={self.session_id[:8]}...)"
                    )
                    return True
                else:
                    logger.error(f"Negotiate failed: {resp.status}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def connect_ws(self) -> bool:
        """Establish WebSocket for real-time messaging."""
        if not self._session:
            await self.connect()
        ws_url = self.server_url.replace("http://", "ws://") + f"/ws/{self.agent_id}"
        try:
            self._ws = await self._session.ws_connect(ws_url)
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._listener_task = asyncio.create_task(self._ws_listener())
            logger.info(f"WebSocket connected: {ws_url}")
            return True
        except Exception as e:
            logger.error(f"WebSocket connect failed: {e}")
            return False

    async def disconnect(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._listener_task:
            self._listener_task.cancel()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Sending ─────────────────────────────────────────────────────

    async def send_message(
        self,
        recipient: str,
        msg_type: str,
        payload: Dict[str, Any],
        priority: int = 2,
        ttl: int = 300,
        correlation_id: Optional[str] = None,
    ) -> Optional[Dict]:
        if not self._session:
            await self.connect()
        msg = {
            "id": str(uuid.uuid4()),
            "sender": self.agent_id,
            "recipient": recipient,
            "type": msg_type,
            "priority": priority,
            "ttl_seconds": ttl,
            "correlation_id": correlation_id,
            "payload": payload,
            "version": "1.0.0",
        }
        try:
            async with self._session.post(
                f"{self.server_url}/api/v1/messages", json=msg
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error(f"Send failed: {resp.status}")
                return None
        except aiohttp.ClientError as e:
            logger.error(f"Send error: {e}")
            return None

    async def send_finding(
        self, content: str, recipient: str = "__broadcast__", priority: int = 2
    ):
        return await self.send_message(
            recipient, "finding", {"content": content}, priority
        )

    async def send_task(
        self,
        recipient: str,
        description: str,
        context: Optional[Dict] = None,
        priority: int = 2,
    ):
        payload = {"description": description}
        if context:
            payload["context"] = context
        return await self.send_message(recipient, "task_assignment", payload, priority)

    async def send_question(
        self, recipient: str, question: str, context: Optional[Dict] = None
    ):
        payload = {"question": question}
        if context:
            payload["context"] = context
        cid = str(uuid.uuid4())
        return await self.send_message(
            recipient, "question", payload, correlation_id=cid
        )

    async def send_answer(
        self, recipient: str, answer: str, correlation_id: Optional[str] = None
    ):
        return await self.send_message(
            recipient, "answer", {"answer": answer}, correlation_id=correlation_id
        )

    async def send_status(self, status: str):
        return await self.send_message(
            "__broadcast__", "status_update", {"status": status}
        )

    async def request_file_lock(self, filepath: str):
        return await self.send_message(
            "__broadcast__", "file_lock", {"action": "lock", "filepath": filepath}, priority=0
        )

    async def release_file_lock(self, filepath: str):
        return await self.send_message(
            "__broadcast__", "file_lock", {"action": "unlock", "filepath": filepath}
        )

    # ── Receiving ───────────────────────────────────────────────────

    async def poll_messages(
        self,
        since: Optional[str] = None,
        type_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        if not self._session:
            await self.connect()
        params: Dict[str, Any] = {"recipient": self.agent_id, "limit": limit}
        if since:
            params["since"] = since
        if type_filter:
            params["type"] = type_filter
        try:
            async with self._session.get(
                f"{self.server_url}/api/v1/messages", params=params
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except aiohttp.ClientError:
            return []

    async def ack_message(self, message_id: str) -> bool:
        try:
            async with self._session.post(
                f"{self.server_url}/api/v1/messages/{message_id}/ack"
            ) as resp:
                return resp.status == 200
        except aiohttp.ClientError:
            return False

    def on_message(self, msg_type: str, handler: Callable):
        self._handlers[msg_type] = handler

    # ── Internal ────────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        while True:
            try:
                await asyncio.sleep(10)
                if self._ws and not self._ws.closed:
                    await self._ws.send_json({"type": "heartbeat"})
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _ws_listener(self):
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    frame_type = data.get("type")
                    if frame_type == "message":
                        msg_data = data.get("data", {})
                        msg_type = msg_data.get("type")
                        handler = self._handlers.get(msg_type)
                        if handler:
                            handler(msg_data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"WS listener error: {e}")
        # Auto-reconnect
        await self._reconnect()

    async def _reconnect(self):
        self._reconnect_attempts += 1
        delay = min(2 ** self._reconnect_attempts, 30)
        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)
        try:
            await self.connect_ws()
        except Exception:
            await self._reconnect()


# ── Synchronous wrapper ─────────────────────────────────────────────

class SyncAgentClient:
    """Blocking wrapper for non-async contexts (e.g. Claude Code scripts)."""

    def __init__(self, agent_id: str, agent_type: str = "custom", **kwargs):
        self._async = AgentClient(agent_id, agent_type, **kwargs)
        self._loop = asyncio.new_event_loop()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def connect(self):
        return self._run(self._async.connect())

    def send_message(self, *a, **kw):
        return self._run(self._async.send_message(*a, **kw))

    def send_finding(self, *a, **kw):
        return self._run(self._async.send_finding(*a, **kw))

    def send_task(self, *a, **kw):
        return self._run(self._async.send_task(*a, **kw))

    def send_question(self, *a, **kw):
        return self._run(self._async.send_question(*a, **kw))

    def send_answer(self, *a, **kw):
        return self._run(self._async.send_answer(*a, **kw))

    def send_status(self, *a, **kw):
        return self._run(self._async.send_status(*a, **kw))

    def request_file_lock(self, *a, **kw):
        return self._run(self._async.request_file_lock(*a, **kw))

    def release_file_lock(self, *a, **kw):
        return self._run(self._async.release_file_lock(*a, **kw))

    def poll_messages(self, **kw):
        return self._run(self._async.poll_messages(**kw))

    def ack_message(self, *a, **kw):
        return self._run(self._async.ack_message(*a, **kw))

    def close(self):
        self._run(self._async.disconnect())
        self._loop.close()
