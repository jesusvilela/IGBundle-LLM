"""IACS Health Monitor — heartbeat tracking, stale agent detection."""

import time
from datetime import datetime, timezone
from typing import Dict, List

from .models import AgentInfo, AgentStatus


class HealthMonitor:
    def __init__(self, timeout_ms: int = 30000, cleanup_interval_s: int = 60):
        self.agents: Dict[str, AgentInfo] = {}
        self.timeout_ms = timeout_ms
        self.cleanup_interval_s = cleanup_interval_s
        self._heartbeat_times: Dict[str, float] = {}

    def register_agent(self, info: AgentInfo) -> None:
        self.agents[info.agent_id] = info
        self._heartbeat_times[info.agent_id] = time.time()

    def record_heartbeat(self, agent_id: str) -> None:
        self._heartbeat_times[agent_id] = time.time()
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = (
                datetime.now(timezone.utc).isoformat()
            )

    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        if agent_id in self.agents:
            self.agents[agent_id].status = status

    def get_stale_agents(self) -> List[str]:
        now = time.time()
        threshold = self.timeout_ms / 1000.0
        stale = []
        for agent_id, last_hb in self._heartbeat_times.items():
            if (now - last_hb) > threshold:
                info = self.agents.get(agent_id)
                if info and info.status != AgentStatus.OFFLINE:
                    stale.append(agent_id)
        return stale

    def remove_agent(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)
        self._heartbeat_times.pop(agent_id, None)

    def get_all_agents(self) -> List[AgentInfo]:
        return list(self.agents.values())
