/* IACS Dashboard — Frontend Logic */

const socket = io();

// ── DOM refs ──────────────────────────────────────────────────────
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const agentsContainer = document.getElementById("agents-container");
const statTotal = document.getElementById("stat-total");
const statPerMin = document.getElementById("stat-per-min");
const statAgents = document.getElementById("stat-agents");
const statDead = document.getElementById("stat-dead");
const typeChart = document.getElementById("type-chart");
const agentChart = document.getElementById("agent-chart");
const logBody = document.getElementById("log-body");
const logScroll = document.getElementById("log-scroll");
const typeFilter = document.getElementById("type-filter");
const autoScroll = document.getElementById("auto-scroll");

// ── Connection status ─────────────────────────────────────────────
socket.on("iacs_status", (data) => {
  if (data.connected) {
    statusDot.className = "status-dot ok";
    statusText.textContent = "IACS Server Connected";
  } else {
    statusDot.className = "status-dot err";
    statusText.textContent = "Disconnected — retrying...";
  }
});

// ── Stats update ──────────────────────────────────────────────────
socket.on("stats_update", (data) => {
  const { stats, agents } = data;
  if (stats) {
    statTotal.textContent = stats.total_messages || 0;
    statPerMin.textContent = stats.messages_last_minute || 0;
    statAgents.textContent = stats.active_agents || 0;
    statDead.textContent = stats.dead_letters || 0;
    renderBarChart(typeChart, stats.messages_by_type || {}, "var(--accent)");
    renderBarChart(agentChart, stats.messages_by_agent || {}, "var(--purple)");
  }
  if (agents) {
    renderAgents(agents);
  }
});

// ── Live events ───────────────────────────────────────────────────
socket.on("iacs_event", (event) => {
  if (event.type === "message" && event.data) {
    addLogRow(event.data);
  }
  if (event.type === "agent_event") {
    // Will refresh on next stats_update poll
  }
});

// ── Message log (initial batch) ───────────────────────────────────
socket.on("message_log", (msgs) => {
  logBody.innerHTML = "";
  msgs.forEach(addLogRow);
});

// ── Filter ────────────────────────────────────────────────────────
typeFilter.addEventListener("change", () => {
  socket.emit("request_log", {
    limit: 200,
    type_filter: typeFilter.value || null,
  });
});

// ── Renderers ─────────────────────────────────────────────────────

function renderAgents(agents) {
  if (!agents || agents.length === 0) {
    agentsContainer.innerHTML =
      '<div class="agent-card"><div class="name" style="color:var(--text-dim)">No agents connected</div></div>';
    return;
  }
  agentsContainer.innerHTML = agents
    .filter((a) => a.agent_id !== "__dashboard__")
    .map((a) => {
      const st = a.status || "offline";
      return `
      <div class="agent-card">
        <div class="name">${esc(a.agent_id)}</div>
        <div class="type">${esc(a.agent_type)} ${a.ws_connected ? "// WS" : "// REST"}</div>
        <span class="badge ${st}">${st.toUpperCase()}</span>
      </div>`;
    })
    .join("");
}

function renderBarChart(container, data, color) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    container.innerHTML = '<div style="color:var(--text-dim);font-size:11px">No data yet</div>';
    return;
  }
  const max = Math.max(...entries.map(([, v]) => v), 1);
  container.innerHTML = entries
    .map(([label, val]) => {
      const h = Math.max((val / max) * 90, 2);
      const shortLabel = label.length > 8 ? label.slice(0, 8) : label;
      return `
      <div class="bar-col">
        <div class="bar" style="height:${h}px;background:${color}" title="${label}: ${val}"></div>
        <div class="bar-label">${esc(shortLabel)}</div>
      </div>`;
    })
    .join("");
}

function addLogRow(msg) {
  const tr = document.createElement("tr");
  const ts = msg.timestamp
    ? new Date(msg.timestamp).toLocaleTimeString()
    : "";
  const pClass = `priority-${msg.priority ?? 2}`;
  const preview = msg.payload
    ? JSON.stringify(msg.payload).slice(0, 80)
    : "";
  tr.innerHTML = `
    <td>${esc(ts)}</td>
    <td>${esc(msg.sender || "")}</td>
    <td>${esc(msg.recipient || "")}</td>
    <td>${esc(msg.type || "")}</td>
    <td class="${pClass}">${priorityLabel(msg.priority)}</td>
    <td title="${esc(preview)}">${esc(preview)}</td>
  `;
  logBody.appendChild(tr);

  // Trim to 200 rows
  while (logBody.children.length > 200) {
    logBody.removeChild(logBody.firstChild);
  }

  if (autoScroll.checked) {
    logScroll.scrollTop = logScroll.scrollHeight;
  }
}

function priorityLabel(p) {
  return ["CRIT", "HIGH", "NORM", "LOW"][p ?? 2] || "NORM";
}

function esc(s) {
  if (!s) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

// ── Telemetry real-time graphs ───────────────────────────────────

const TELEM_MAX = 200;
const telemData = { S: [], K: [], loss_llm: [], loss_geo: [], fiber_diversity: [] };
const telemSteps = [];
const telemStatus = document.getElementById("telem-status");

socket.on("telemetry", (event) => {
  const d = event.data || event;
  const step = d.step ?? telemSteps.length;
  telemSteps.push(step);
  if (telemSteps.length > TELEM_MAX) telemSteps.shift();

  for (const key of Object.keys(telemData)) {
    telemData[key].push(d[key] ?? null);
    if (telemData[key].length > TELEM_MAX) telemData[key].shift();
  }
  telemStatus.textContent = `step ${step} | ${telemData.S.length} pts`;
  drawTelemCharts();
});

// Also handle telemetry arriving as iacs_event
socket.on("iacs_event", (event) => {
  if (event.type === "telemetry" && event.data) {
    socket.emit("telemetry", event);  // Re-dispatch
    const d = event.data;
    const step = d.step ?? telemSteps.length;
    telemSteps.push(step);
    if (telemSteps.length > TELEM_MAX) telemSteps.shift();
    for (const key of Object.keys(telemData)) {
      telemData[key].push(d[key] ?? null);
      if (telemData[key].length > TELEM_MAX) telemData[key].shift();
    }
    telemStatus.textContent = `step ${step} | ${telemData.S.length} pts`;
    drawTelemCharts();
  }
});

function drawTelemCharts() {
  drawLineChart("chart-sk", [
    { data: telemData.S, color: "#3fb950", label: "S" },
    { data: telemData.K, color: "#f85149", label: "K" },
  ]);
  drawLineChart("chart-loss", [
    { data: telemData.loss_llm, color: "#58a6ff", label: "loss_llm" },
    { data: telemData.loss_geo, color: "#bc8cff", label: "loss_geo" },
    { data: telemData.fiber_diversity, color: "#d29922", label: "fiber_div" },
  ]);
}

function drawLineChart(canvasId, series) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 50, r: 10, t: 10, b: 20 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  ctx.clearRect(0, 0, W, H);

  // Compute global min/max across all series
  let allVals = [];
  for (const s of series) {
    for (const v of s.data) { if (v !== null && isFinite(v)) allVals.push(v); }
  }
  if (allVals.length === 0) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "12px monospace";
    ctx.fillText("No telemetry data yet", W / 2 - 70, H / 2);
    return;
  }
  let yMin = Math.min(...allVals), yMax = Math.max(...allVals);
  if (yMax - yMin < 0.001) { yMin -= 0.5; yMax += 0.5; }
  const yRange = yMax - yMin;
  const n = Math.max(...series.map((s) => s.data.length));

  // Grid lines
  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.fillStyle = "#8b949e";
    ctx.font = "10px monospace";
    const val = yMax - (i / 4) * yRange;
    ctx.fillText(val.toFixed(2), 2, y + 4);
  }

  // Draw each series
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i];
      if (v === null || !isFinite(v)) continue;
      const x = pad.l + (i / Math.max(n - 1, 1)) * pw;
      const y = pad.t + (1 - (v - yMin) / yRange) * ph;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Label at end
    if (s.data.length > 0) {
      const lastV = s.data[s.data.length - 1];
      if (lastV !== null && isFinite(lastV)) {
        const lx = pad.l + ((s.data.length - 1) / Math.max(n - 1, 1)) * pw;
        const ly = pad.t + (1 - (lastV - yMin) / yRange) * ph;
        ctx.fillStyle = s.color;
        ctx.font = "bold 10px monospace";
        ctx.fillText(`${s.label}=${lastV.toFixed(3)}`, lx - 60, ly - 6);
      }
    }
  }
}
