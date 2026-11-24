"""
AUTORESEARCH Dashboard Generator
=================================
Generates a self-contained HTML dashboard from research_status.json,
topic files, and output directories.
"""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
STATUS_FILE = ROOT / "research_status.json"
TOPICS_DIR = ROOT / "topics"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"
DASHBOARD_DIR = ROOT / "dashboard"
DASHBOARD_FILE = DASHBOARD_DIR / "index.html"

TOPICS = {
    1: ("fiber_bundles_neural_nets", "Fiber Bundle Geometry in Neural Networks"),
    2: ("information_geometry_llm", "Information Geometry of Language Models"),
    3: ("neuromorphic_memory", "Neuromorphic Memory Systems for AI"),
    4: ("geometric_steering", "Geometric Steering and Controllable Generation"),
    5: ("poincare_hyperbolic_ml", "Poincaré Embeddings and Hyperbolic ML"),
}

STATUS_COLORS = {
    "completed": "#22c55e",
    "running": "#3b82f6",
    "failed": "#ef4444",
    "timeout": "#f59e0b",
    "error": "#ef4444",
    "pending": "#6b7280",
}


def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {"runs": [], "watches": [], "last_updated": None}


def count_outputs() -> dict:
    """Count output files per topic slug."""
    counts = {}
    if not OUTPUTS_DIR.exists():
        return counts
    for d in OUTPUTS_DIR.iterdir():
        if d.is_dir():
            slug = "_".join(d.name.split("_")[:-2])  # strip timestamp
            files = list(d.iterdir())
            counts[slug] = counts.get(slug, 0) + len(files)
    return counts


def load_topic_questions() -> dict:
    """Load research questions from topic files."""
    questions = {}
    for num, (slug, _title) in TOPICS.items():
        fname = f"{num:02d}_{slug}.md"
        fpath = TOPICS_DIR / fname
        if fpath.exists():
            text = fpath.read_text(encoding="utf-8")
            qs = []
            in_questions = False
            for line in text.splitlines():
                if "## Research Questions" in line:
                    in_questions = True
                    continue
                if in_questions and line.startswith("##"):
                    break
                if in_questions and line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")):
                    qs.append(line.strip().split(". ", 1)[-1])
            questions[slug] = qs
    return questions


def build_dashboard():
    """Generate the HTML dashboard."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    status = load_status()
    output_counts = count_outputs()
    topic_questions = load_topic_questions()

    runs = status.get("runs", [])
    watches = status.get("watches", [])
    last_updated = status.get("last_updated", "never")

    # Compute per-topic stats
    topic_stats = {}
    for num, (slug, title) in TOPICS.items():
        topic_runs = [r for r in runs if r.get("topic") == slug]
        completed = sum(1 for r in topic_runs if r.get("status") == "completed")
        failed = sum(1 for r in topic_runs if r.get("status") in ("failed", "error", "timeout"))
        total = len(topic_runs)
        outputs = output_counts.get(slug, 0)
        questions = topic_questions.get(slug, [])
        topic_stats[num] = {
            "slug": slug,
            "title": title,
            "total_runs": total,
            "completed": completed,
            "failed": failed,
            "outputs": outputs,
            "questions": questions,
        }

    # Build run log rows
    run_rows = ""
    for r in reversed(runs[-50:]):
        color = STATUS_COLORS.get(r.get("status", "pending"), "#6b7280")
        started = r.get("started_at", "")[:19].replace("T", " ")
        run_rows += f"""
        <tr>
            <td><span class="status-dot" style="background:{color}"></span>{r.get('status','?')}</td>
            <td>{r.get('mode','?')}</td>
            <td>{r.get('topic','?')}</td>
            <td class="mono">{started}</td>
            <td class="mono">{r.get('exit_code','—')}</td>
        </tr>"""

    # Build topic cards
    topic_cards = ""
    for num in sorted(topic_stats):
        ts = topic_stats[num]
        pct = (ts["completed"] / ts["total_runs"] * 100) if ts["total_runs"] > 0 else 0
        questions_html = "".join(f"<li>{q}</li>" for q in ts["questions"])
        topic_cards += f"""
        <div class="card">
            <h3>Axis {num}: {ts['title']}</h3>
            <div class="stats-row">
                <div class="stat">
                    <span class="stat-value">{ts['total_runs']}</span>
                    <span class="stat-label">Runs</span>
                </div>
                <div class="stat">
                    <span class="stat-value" style="color:#22c55e">{ts['completed']}</span>
                    <span class="stat-label">Completed</span>
                </div>
                <div class="stat">
                    <span class="stat-value" style="color:#ef4444">{ts['failed']}</span>
                    <span class="stat-label">Failed</span>
                </div>
                <div class="stat">
                    <span class="stat-value" style="color:#3b82f6">{ts['outputs']}</span>
                    <span class="stat-label">Outputs</span>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{pct}%"></div>
            </div>
            <details>
                <summary>Research Questions ({len(ts['questions'])})</summary>
                <ol class="questions">{questions_html}</ol>
            </details>
        </div>"""

    # Watch list
    watch_items = ""
    for w in watches:
        watch_items += f"<li><strong>{w.get('topic','?')}</strong> — {w.get('query','')[:80]}</li>"
    if not watch_items:
        watch_items = "<li class='muted'>No active watches</li>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AUTORESEARCH Dashboard — IGBundle</title>
<style>
:root {{
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1a1a24;
    --border: #2a2a3a;
    --text: #e0e0e8;
    --text-muted: #8888a0;
    --accent: #7c3aed;
    --accent2: #3b82f6;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #f59e0b;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.6;
    padding: 24px;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{
    font-size: 1.8rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}}
.subtitle {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 24px; }}
.header-stats {{
    display: flex;
    gap: 24px;
    margin-bottom: 32px;
    flex-wrap: wrap;
}}
.header-stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 24px;
    min-width: 140px;
}}
.header-stat .value {{
    font-size: 2rem;
    font-weight: 700;
}}
.header-stat .label {{
    color: var(--text-muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.section {{ margin-bottom: 32px; }}
.section h2 {{
    font-size: 1.2rem;
    color: var(--accent2);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s;
}}
.card:hover {{ border-color: var(--accent); }}
.card h3 {{ font-size: 1rem; margin-bottom: 12px; }}
.stats-row {{ display: flex; gap: 16px; margin-bottom: 12px; }}
.stat {{ text-align: center; }}
.stat-value {{ display: block; font-size: 1.4rem; font-weight: 700; }}
.stat-label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }}
.progress-bar {{
    height: 4px;
    background: var(--surface2);
    border-radius: 2px;
    margin-bottom: 12px;
    overflow: hidden;
}}
.progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--green));
    border-radius: 2px;
    transition: width 0.3s;
}}
details {{ margin-top: 8px; }}
summary {{
    cursor: pointer;
    color: var(--text-muted);
    font-size: 0.85rem;
}}
.questions {{
    margin-top: 8px;
    padding-left: 20px;
    font-size: 0.85rem;
    color: var(--text-muted);
}}
.questions li {{ margin-bottom: 4px; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}
th {{
    text-align: left;
    padding: 8px 12px;
    border-bottom: 2px solid var(--border);
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
td {{
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
}}
tr:hover td {{ background: var(--surface2); }}
.status-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}}
.mono {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.8rem; }}
.muted {{ color: var(--text-muted); }}
.watch-list {{ list-style: none; }}
.watch-list li {{
    padding: 8px 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}}
.footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
}}
@media (max-width: 768px) {{
    body {{ padding: 12px; }}
    .header-stats {{ gap: 12px; }}
    .cards {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">

<h1>AUTORESEARCH</h1>
<p class="subtitle">IGBundle Manifold LLM — Feynman Research Orchestrator</p>

<div class="header-stats">
    <div class="header-stat">
        <div class="value">{len(runs)}</div>
        <div class="label">Total Runs</div>
    </div>
    <div class="header-stat">
        <div class="value" style="color:var(--green)">{sum(1 for r in runs if r.get('status')=='completed')}</div>
        <div class="label">Completed</div>
    </div>
    <div class="header-stat">
        <div class="value" style="color:var(--red)">{sum(1 for r in runs if r.get('status') in ('failed','error','timeout'))}</div>
        <div class="label">Failed</div>
    </div>
    <div class="header-stat">
        <div class="value">{len(watches)}</div>
        <div class="label">Watches</div>
    </div>
    <div class="header-stat">
        <div class="value">5</div>
        <div class="label">Research Axes</div>
    </div>
</div>

<div class="section">
    <h2>Research Axes</h2>
    <div class="cards">
        {topic_cards}
    </div>
</div>

<div class="section">
    <h2>Run History</h2>
    {"<table><thead><tr><th>Status</th><th>Mode</th><th>Topic</th><th>Started</th><th>Exit</th></tr></thead><tbody>" + run_rows + "</tbody></table>" if run_rows else "<p class='muted'>No runs recorded yet. Launch with: <code>python AUTORESEARCH/run_research.py --topic 1 --mode deep</code></p>"}
</div>

<div class="section">
    <h2>Active Watches</h2>
    <ul class="watch-list">
        {watch_items}
    </ul>
</div>

<div class="footer">
    Dashboard generated: {now} &middot; Status last updated: {last_updated or 'never'}<br>
    AUTORESEARCH &middot; Powered by Feynman + Claude
</div>

</div>
</body>
</html>"""

    DASHBOARD_FILE.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {DASHBOARD_FILE}")
    return str(DASHBOARD_FILE)


if __name__ == "__main__":
    build_dashboard()
