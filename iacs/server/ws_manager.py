"""IACS WebSocket Connection Manager."""

import logging
from typing import Dict, Optional

from fastapi import WebSocket

logger = logging.getLogger("iacs.ws")


class ConnectionManager:
    """Manages WebSocket connections for all agents."""

    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, agent_id: str, websocket: WebSocket) -> None:
        self.active[agent_id] = websocket
        logger.info(f"WS connected: {agent_id} (total: {len(self.active)})")

    async def disconnect(self, agent_id: str) -> None:
        self.active.pop(agent_id, None)
        logger.info(f"WS disconnected: {agent_id} (total: {len(self.active)})")

    async def send_to_agent(self, agent_id: str, data: dict) -> bool:
        ws = self.active.get(agent_id)
        if ws is None:
            return False
        try:
            await ws.send_json(data)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {agent_id}: {e}")
            await self.disconnect(agent_id)
            return False

    async def broadcast(self, data: dict, exclude: Optional[str] = None) -> None:
        disconnected = []
        for agent_id, ws in self.active.items():
            if agent_id == exclude:
                continue
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(agent_id)
        for agent_id in disconnected:
            await self.disconnect(agent_id)

    def is_connected(self, agent_id: str) -> bool:
        return agent_id in self.active
