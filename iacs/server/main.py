"""
IACS Server — Inter-Agent Communication System
FastAPI REST + WebSocket message bus on localhost:9100
"""

import asyncio
import collections
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    Message, MessageType, Priority,
    NegotiateRequest, NegotiateResponse,
    AgentInfo, AgentStatus, ServerStats,
)
from .message_store import MessageStore
from .ws_manager import ConnectionManager
from .health import HealthMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("iacs.server")

SUPPORTED_VERSIONS = ["1.0.0"]
ALL_CAPABILITIES = [t.value for t in MessageType]


# ── Background tasks ────────────────────────────────────────────────

async def ttl_sweep_loop(app: FastAPI):
    while True:
        await asyncio.sleep(30)
        try:
            swept = app.state.store.sweep_expired()
            if swept > 0:
                logger.info(f"TTL sweep: removed {swept} expired messages")
                await app.state.ws_mgr.broadcast({
                    "type": "system",
                    "event": "ttl_sweep",
                    "count": swept,
                })
        except Exception as e:
            logger.error(f"TTL sweep error: {e}")


async def health_check_loop(app: FastAPI):
    while True:
        await asyncio.sleep(15)
        try:
            stale = app.state.health.get_stale_agents()
            for agent_id in stale:
                app.state.health.update_status(agent_id, AgentStatus.OFFLINE)
                logger.warning(f"Agent {agent_id} timed out (no heartbeat)")
                await app.state.ws_mgr.broadcast({
                    "type": "agent_event",
                    "event": "timeout",
                    "agent_id": agent_id,
                })
        except Exception as e:
            logger.error(f"Health check error: {e}")


# ── Lifespan ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = MessageStore()
    app.state.ws_mgr = ConnectionManager()
    app.state.health = HealthMonitor()
    app.state.start_time = time.time()
    app.state.telemetry_ring = collections.deque(maxlen=1000)
    logger.info("IACS server starting on http://localhost:9100")

    sweep = asyncio.create_task(ttl_sweep_loop(app))
    hcheck = asyncio.create_task(health_check_loop(app))
    yield
    sweep.cancel()
    hcheck.cancel()
    app.state.store.close()
    logger.info("IACS server shut down")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="IACS — Inter-Agent Communication System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST: Negotiation ───────────────────────────────────────────────

@app.post("/api/v1/negotiate", response_model=NegotiateResponse)
async def negotiate(req: NegotiateRequest):
    version = req.protocol_version if req.protocol_version in SUPPORTED_VERSIONS else "1.0.0"
    session_id = str(uuid.uuid4())
    caps = [c for c in req.capabilities if c in ALL_CAPABILITIES] or ALL_CAPABILITIES

    info = AgentInfo(
        agent_id=req.agent_id,
        agent_type=req.agent_type,
        session_id=session_id,
        capabilities=caps,
    )
    app.state.health.register_agent(info)
    logger.info(f"Agent negotiated: {req.agent_id} ({req.agent_type}) v{version}")

    await app.state.ws_mgr.broadcast({
        "type": "agent_event",
        "event": "registered",
        "agent_id": req.agent_id,
        "agent_type": req.agent_type,
    })

    return NegotiateResponse(
        session_id=session_id,
        accepted_version=version,
        server_capabilities=ALL_CAPABILITIES,
        heartbeat_interval_ms=10000,
    )


# ── REST: Messages ──────────────────────────────────────────────────

@app.post("/api/v1/messages")
async def send_message(msg: Message):
    app.state.store.store_message(msg)
    logger.info(f"MSG [{msg.type.value}] {msg.sender} -> {msg.recipient}")

    data = {"type": "message", "data": msg.model_dump(mode="json")}
    if msg.recipient == "__broadcast__":
        await app.state.ws_mgr.broadcast(data, exclude=msg.sender)
    else:
        await app.state.ws_mgr.send_to_agent(msg.recipient, data)
    # Always send to dashboard observer
    await app.state.ws_mgr.send_to_agent("__dashboard__", data)

    return msg.model_dump(mode="json")


@app.get("/api/v1/messages")
async def get_messages(
    recipient: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    limit: int = Query(50),
):
    return app.state.store.get_messages(recipient, type, since, limit)


@app.get("/api/v1/messages/{message_id}")
async def get_message(message_id: str):
    msg = app.state.store.get_message(message_id)
    if not msg:
        raise HTTPException(404, "Message not found")
    return msg


@app.post("/api/v1/messages/{message_id}/ack")
async def ack_message(message_id: str):
    ok = app.state.store.ack_message(message_id)
    if not ok:
        raise HTTPException(404, "Message not found")
    return {"status": "acked", "message_id": message_id}


# ── REST: Agents ────────────────────────────────────────────────────

@app.get("/api/v1/agents")
async def list_agents():
    agents = app.state.health.get_all_agents()
    result = []
    for a in agents:
        d = a.model_dump(mode="json")
        d["ws_connected"] = app.state.ws_mgr.is_connected(a.agent_id)
        result.append(d)
    return result


# ── REST: Health & Stats ────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - app.state.start_time, 1),
        "active_agents": len(app.state.health.agents),
        "ws_connections": len(app.state.ws_mgr.active),
    }


@app.get("/api/v1/stats", response_model=ServerStats)
async def get_stats():
    db_stats = app.state.store.get_stats()
    return ServerStats(
        total_messages=db_stats["total_messages"],
        messages_last_minute=db_stats["messages_last_minute"],
        active_agents=len([
            a for a in app.state.health.agents.values()
            if a.status != AgentStatus.OFFLINE
        ]),
        dead_letters=db_stats["dead_letters"],
        uptime_seconds=round(time.time() - app.state.start_time, 1),
        messages_by_type=db_stats["messages_by_type"],
        messages_by_agent=db_stats["messages_by_agent"],
    )


# ── REST: Telemetry ────────────────────────────────────────────────

@app.post("/api/v1/telemetry")
async def post_telemetry(data: dict):
    """Ingest a telemetry snapshot. Expected keys: agent_id, metrics (dict of floats), step (int)."""
    agent_id = data.get("agent_id", "unknown")
    metrics = data.get("metrics", {})
    step = data.get("step")
    ts = time.time()

    entry = {"t": ts, "step": step, "agent": agent_id, **metrics}
    app.state.telemetry_ring.append(entry)

    # Broadcast to dashboard via WS
    await app.state.ws_mgr.broadcast({
        "type": "telemetry",
        "data": entry,
    })
    return {"status": "ok", "buffered": len(app.state.telemetry_ring)}


@app.get("/api/v1/telemetry")
async def get_telemetry(last: int = Query(200)):
    """Return the last N telemetry entries."""
    ring = app.state.telemetry_ring
    entries = list(ring)[-last:]
    return entries


# ── REST: Dead Letters ──────────────────────────────────────────────

@app.get("/api/v1/dead-letters")
async def get_dead_letters():
    return app.state.store.get_dead_letters()


@app.post("/api/v1/dead-letters/{message_id}/retry")
async def retry_dead_letter(message_id: str):
    result = app.state.store.retry_dead_letter(message_id)
    if not result:
        raise HTTPException(404, "Dead letter not found")
    return result


# ── WebSocket ───────────────────────────────────────────────────────

@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    mgr = app.state.ws_mgr
    health = app.state.health

    await mgr.connect(agent_id, websocket)
    if agent_id in health.agents:
        health.agents[agent_id].ws_connected = True

    await mgr.broadcast(
        {"type": "agent_event", "event": "ws_connected", "agent_id": agent_id},
        exclude=agent_id,
    )

    try:
        while True:
            data = await websocket.receive_json()
            frame_type = data.get("type")

            if frame_type == "heartbeat":
                health.record_heartbeat(agent_id)
                await websocket.send_json({"type": "heartbeat", "status": "ok"})

            elif frame_type == "message":
                msg = Message(**data.get("data", {}))
                msg.sender = agent_id
                app.state.store.store_message(msg)
                logger.info(f"WS MSG [{msg.type.value}] {msg.sender} -> {msg.recipient}")

                payload = {"type": "message", "data": msg.model_dump(mode="json")}
                if msg.recipient == "__broadcast__":
                    await mgr.broadcast(payload, exclude=agent_id)
                else:
                    await mgr.send_to_agent(msg.recipient, payload)
                await mgr.send_to_agent("__dashboard__", payload)

            elif frame_type == "status_update":
                status = data.get("status", "online")
                try:
                    health.update_status(agent_id, AgentStatus(status))
                except ValueError:
                    pass
                await mgr.broadcast(
                    {
                        "type": "agent_event",
                        "event": "status_change",
                        "agent_id": agent_id,
                        "status": status,
                    },
                    exclude=agent_id,
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WS error for {agent_id}: {e}")
    finally:
        await mgr.disconnect(agent_id)
        if agent_id in health.agents:
            health.agents[agent_id].ws_connected = False
            health.update_status(agent_id, AgentStatus.OFFLINE)
        await mgr.broadcast(
            {"type": "agent_event", "event": "ws_disconnected", "agent_id": agent_id}
        )


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "iacs.server.main:app",
        host="localhost",
        port=9100,
        log_level="info",
    )
