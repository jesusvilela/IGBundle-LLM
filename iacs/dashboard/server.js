/**
 * IACS Observability Dashboard
 *
 * Bridges Python IACS server (port 9100) to browser clients via Socket.IO (port 9110).
 * Polls REST stats, relays WebSocket events in real time.
 */

const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const WebSocket = require("ws");
const path = require("path");

const IACS_REST = "http://localhost:9100";
const IACS_WS = "ws://localhost:9100/ws/__dashboard__";
const PORT = 9110;

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

app.use(express.static(path.join(__dirname, "public")));

// ── State ─────────────────────────────────────────────────────────

let iacsConnected = false;
let wsConn = null;
let messageLog = [];
const MAX_LOG = 500;
let stats = {};
let agents = [];

// ── IACS WebSocket Bridge ─────────────────────────────────────────

function connectToIACS() {
  try {
    wsConn = new WebSocket(IACS_WS);
  } catch (e) {
    console.log("[Dashboard] Cannot create WS, retry in 5s...");
    setTimeout(connectToIACS, 5000);
    return;
  }

  wsConn.on("open", () => {
    iacsConnected = true;
    console.log("[Dashboard] Connected to IACS server");
    io.emit("iacs_status", { connected: true });
  });

  wsConn.on("message", (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      io.emit("iacs_event", msg);

      if (msg.type === "message" && msg.data) {
        messageLog.push({
          ...msg.data,
          _received: new Date().toISOString(),
        });
        if (messageLog.length > MAX_LOG) {
          messageLog = messageLog.slice(-MAX_LOG);
        }
      }
    } catch (e) {
      // ignore parse errors
    }
  });

  wsConn.on("close", () => {
    iacsConnected = false;
    console.log("[Dashboard] Disconnected from IACS. Reconnecting in 5s...");
    io.emit("iacs_status", { connected: false });
    setTimeout(connectToIACS, 5000);
  });

  wsConn.on("error", (err) => {
    console.error("[Dashboard] WS error:", err.message);
    iacsConnected = false;
  });
}

// ── REST Polling ──────────────────────────────────────────────────

async function pollStats() {
  try {
    const [sRes, aRes] = await Promise.all([
      fetch(`${IACS_REST}/api/v1/stats`),
      fetch(`${IACS_REST}/api/v1/agents`),
    ]);
    stats = await sRes.json();
    agents = await aRes.json();
    io.emit("stats_update", { stats, agents });
  } catch (e) {
    // server not up yet, that's fine
  }
}

// ── Socket.IO Handlers ───────────────────────────────────────────

io.on("connection", (socket) => {
  console.log("[Dashboard] Browser client connected");
  socket.emit("iacs_status", { connected: iacsConnected });
  socket.emit("stats_update", { stats, agents });
  socket.emit("message_log", messageLog.slice(-100));

  socket.on("request_log", (params) => {
    const { limit = 100, type_filter = null } = params || {};
    let filtered = messageLog;
    if (type_filter) {
      filtered = filtered.filter((m) => m.type === type_filter);
    }
    socket.emit("message_log", filtered.slice(-limit));
  });

  socket.on("retry_dead_letter", async (messageId) => {
    try {
      await fetch(`${IACS_REST}/api/v1/dead-letters/${messageId}/retry`, {
        method: "POST",
      });
    } catch (e) {
      socket.emit("error", { message: "Retry failed" });
    }
  });
});

// ── Start ─────────────────────────────────────────────────────────

server.listen(PORT, "localhost", () => {
  console.log(`\n  IACS Dashboard: http://localhost:${PORT}\n`);
  connectToIACS();
  setInterval(pollStats, 3000);
});
