"""IACS Data Models — Pydantic v2"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    FINDING = "finding"
    QUESTION = "question"
    ANSWER = "answer"
    STATUS_UPDATE = "status_update"
    FILE_LOCK = "file_lock"
    CAPABILITY_QUERY = "capability_query"
    HEARTBEAT = "heartbeat"
    SYSTEM = "system"


class Priority(int, Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class AgentStatus(str, Enum):
    ONLINE = "online"
    IDLE = "idle"
    WORKING = "working"
    OFFLINE = "offline"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sender: str
    recipient: str
    type: MessageType
    priority: Priority = Priority.NORMAL
    ttl_seconds: int = 300
    correlation_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"
    acked: bool = False
    retry_count: int = 0


class NegotiateRequest(BaseModel):
    agent_id: str
    agent_type: str
    protocol_version: str = "1.0.0"
    capabilities: List[str] = Field(default_factory=list)


class NegotiateResponse(BaseModel):
    session_id: str
    accepted_version: str
    server_capabilities: List[str]
    heartbeat_interval_ms: int = 10000


class AgentInfo(BaseModel):
    agent_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.ONLINE
    session_id: str
    capabilities: List[str] = Field(default_factory=list)
    connected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_heartbeat: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ws_connected: bool = False


class ServerStats(BaseModel):
    total_messages: int = 0
    messages_last_minute: int = 0
    active_agents: int = 0
    dead_letters: int = 0
    uptime_seconds: float = 0.0
    messages_by_type: Dict[str, int] = Field(default_factory=dict)
    messages_by_agent: Dict[str, int] = Field(default_factory=dict)
    avg_latency_ms: float = 0.0
