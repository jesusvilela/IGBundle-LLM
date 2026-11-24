"""
AUTORESEARCH Orchestrator
=========================
Launches Feynman research sessions for each topic axis,
logs progress, and updates the dashboard.

Usage:
    python AUTORESEARCH/run_research.py [--topic N] [--mode deep|lit|watch|auto]
    python AUTORESEARCH/run_research.py --all
    python AUTORESEARCH/run_research.py --dashboard
"""

import os
import subprocess
import json
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
TOPICS_DIR = ROOT / "topics"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"
DASHBOARD_DIR = ROOT / "dashboard"
STATUS_FILE = ROOT / "research_status.json"

TOPICS = {
    1: ("fiber_bundles_neural_nets", "Fiber Bundle Geometry in Neural Networks"),
    2: ("information_geometry_llm", "Information Geometry of Language Models"),
    3: ("neuromorphic_memory", "Neuromorphic Memory Systems for AI"),
    4: ("geometric_steering", "Geometric Steering and Controllable Generation"),
    5: ("poincare_hyperbolic_ml", "Poincare Embeddings and Hyperbolic ML"),
}

FEYNMAN_WORKFLOWS = {
    "deep": "deepresearch",
    "lit": "lit",
    "watch": "watch",
    "auto": "autoresearch",
    "review": "review",
}


def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {"runs": [], "watches": [], "last_updated": None}


def save_status(status: dict):
    status["last_updated"] = datetime.now().isoformat()
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def resolve_feynman_executable() -> str | None:
    candidates = ["feynman.cmd", "feynman"] if os.name == "nt" else ["feynman"]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def resolve_feynman_package_root(feynman_executable: str) -> Path:
    return Path(feynman_executable).resolve().parent / "node_modules" / "@companion-ai" / "feynman"


def build_feynman_invocation(mode: str, query: str, session_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Build a working Feynman command for the current platform."""
    workflow = FEYNMAN_WORKFLOWS.get(mode, "deepresearch")
    feynman_executable = resolve_feynman_executable()
    if not feynman_executable:
        raise FileNotFoundError("feynman not found in PATH")

    env = os.environ.copy()
    if os.name != "nt":
        return [feynman_executable, workflow, query, "--session-dir", str(session_dir)], env

    # Windows-specific fallback: invoke the embedded Pi runtime directly.
    # The current Feynman launcher passes absolute paths to `node --import`,
    # which trips Node's ESM loader on Windows for research workflows.
    node_executable = shutil.which("node")
    if not node_executable:
        raise FileNotFoundError("node not found in PATH")

    package_root = resolve_feynman_package_root(feynman_executable)
    promise_polyfill = package_root / "dist" / "system" / "promise-polyfill.js"
    pi_cli = package_root / "node_modules" / "@mariozechner" / "pi-coding-agent" / "dist" / "cli.js"
    research_tools = package_root / "extensions" / "research-tools.ts"
    prompt_templates = package_root / "prompts"
    system_prompt_file = package_root / ".feynman" / "SYSTEM.md"

    required_paths = [promise_polyfill, pi_cli, research_tools, prompt_templates, system_prompt_file]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Feynman installation is incomplete: " + ", ".join(missing))

    node_modules_bin = package_root / "node_modules" / ".bin"
    env["PATH"] = f"{node_modules_bin}{os.pathsep}{env.get('PATH', '')}"
    env["FEYNMAN_SESSION_DIR"] = str(session_dir)
    env["FEYNMAN_NODE_EXECUTABLE"] = node_executable
    env["FEYNMAN_BIN_PATH"] = str(package_root / "bin" / "feynman.js")
    env.setdefault("PI_SKIP_VERSION_CHECK", "1")
    env.setdefault("PI_HARDWARE_CURSOR", "1")

    prompt = f"/{workflow} {query}"
    cmd = [
        node_executable,
        "--import",
        promise_polyfill.resolve().as_uri(),
        str(pi_cli),
        "--session-dir",
        str(session_dir),
        "--extension",
        str(research_tools),
        "--prompt-template",
        str(prompt_templates),
        "--system-prompt",
        system_prompt_file.read_text(encoding="utf-8"),
        "-p",
        prompt,
    ]
    return cmd, env


def summarize_failure(log_file: Path) -> str | None:
    if not log_file.exists():
        return None

    text = log_file.read_text(encoding="utf-8", errors="replace")
    if "ERR_UNSUPPORTED_ESM_URL_SCHEME" in text:
        return "Feynman hit a Windows/Node ESM loader failure while starting the workflow."
    if "[feynman] no supported package manager found" in text:
        return "Feynman setup is incomplete: no supported package manager detected."

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1][:280]


def run_feynman(mode: str, topic_text: str, topic_slug: str) -> dict:
    """Launch a Feynman research session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output subdirectory
    out_dir = OUTPUTS_DIR / f"{topic_slug}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"{topic_slug}_{mode}_{timestamp}.log"

    # Build the Feynman CLI command
    query = f"{topic_text} — focus on recent advances (2023-2026), "
    query += "empirical results, and connections to fiber bundle geometry in transformers"

    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH: {mode.upper()} on Topic: {topic_text}")
    print(f"  Output: {out_dir}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}\n")

    result = {
        "topic": topic_slug,
        "mode": mode,
        "query": query,
        "output_dir": str(out_dir),
        "log_file": str(log_file),
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }

    try:
        cmd, env = build_feynman_invocation(mode, query, out_dir)
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=1800,  # 30 min max per topic
                cwd=str(ROOT.parent),
                env=env,
            )
            result["exit_code"] = proc.returncode
            result["status"] = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                result["error"] = summarize_failure(log_file)
                if result["error"]:
                    print(f"  Failure: {result['error']}")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["exit_code"] = -1
    except FileNotFoundError as exc:
        message = str(exc) or "feynman not found in PATH"
        print(f"  ERROR: {message}")
        if "feynman" in message.lower():
            print("  Hint: run `npm install -g @companion-ai/feynman` and restart the shell.")
        result["status"] = "error"
        result["exit_code"] = -2
        result["error"] = message

    result["finished_at"] = datetime.now().isoformat()
    return result


def run_topic(topic_num: int, mode: str):
    if topic_num not in TOPICS:
        print(f"Unknown topic {topic_num}. Available: {list(TOPICS.keys())}")
        return

    slug, title = TOPICS[topic_num]
    status = load_status()
    result = run_feynman(mode, title, slug)
    status["runs"].append(result)
    save_status(status)
    print(f"\n  Result: {result['status']}")


def run_all(mode: str):
    """Run all topics sequentially."""
    status = load_status()
    for num, (slug, title) in TOPICS.items():
        result = run_feynman(mode, title, slug)
        status["runs"].append(result)
        save_status(status)
        print(f"  Topic {num} ({slug}): {result['status']}")
        time.sleep(5)  # Brief pause between topics


def setup_watches():
    """Set up recurring watches for all topics."""
    status = load_status()
    for num, (slug, title) in TOPICS.items():
        query = f"New papers and developments in {title} applied to language models"
        print(f"  Setting watch: {title}")
        try:
            watch_dir = OUTPUTS_DIR / "_watches" / slug
            watch_dir.mkdir(parents=True, exist_ok=True)
            cmd, env = build_feynman_invocation("watch", query, watch_dir)
            subprocess.run(cmd, timeout=30, capture_output=True, cwd=str(ROOT.parent), env=env)
            status["watches"].append({
                "topic": slug,
                "query": query,
                "created_at": datetime.now().isoformat(),
            })
        except Exception as e:
            print(f"    Failed: {e}")
    save_status(status)


def show_status():
    """Print current research status."""
    status = load_status()
    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH STATUS")
    print(f"  Last updated: {status.get('last_updated', 'never')}")
    print(f"{'='*60}")

    runs = status.get("runs", [])
    if runs:
        print(f"\n  Completed runs: {len(runs)}")
        for r in runs[-10:]:
            print(f"    [{r['status']:>10}] {r['mode']:>5} | {r['topic']}")
    else:
        print("\n  No runs yet.")

    watches = status.get("watches", [])
    if watches:
        print(f"\n  Active watches: {len(watches)}")
        for w in watches:
            print(f"    {w['topic']}")

    # Count outputs
    output_dirs = list(OUTPUTS_DIR.iterdir()) if OUTPUTS_DIR.exists() else []
    print(f"\n  Output directories: {len(output_dirs)}")
    for d in output_dirs[:10]:
        if d.is_dir():
            files = list(d.iterdir())
            print(f"    {d.name}/  ({len(files)} files)")

    print()


def main():
    parser = argparse.ArgumentParser(description="AUTORESEARCH Orchestrator")
    parser.add_argument("--topic", type=int, help="Topic number (1-5)")
    parser.add_argument("--mode", default="deep",
                        choices=["deep", "lit", "watch", "auto", "review"])
    parser.add_argument("--all", action="store_true", help="Run all topics")
    parser.add_argument("--watches", action="store_true", help="Setup recurring watches")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--dashboard", action="store_true", help="Generate dashboard")

    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        show_status()
    elif args.dashboard:
        generate_dashboard()
    elif args.watches:
        setup_watches()
    elif args.all:
        run_all(args.mode)
    elif args.topic:
        run_topic(args.topic, args.mode)
    else:
        print("AUTORESEARCH — Feynman Research Orchestrator")
        print()
        print("Topics:")
        for num, (slug, title) in TOPICS.items():
            print(f"  {num}. {title}")
        print()
        print("Usage:")
        print("  python AUTORESEARCH/run_research.py --topic 1 --mode deep")
        print("  python AUTORESEARCH/run_research.py --all --mode lit")
        print("  python AUTORESEARCH/run_research.py --watches")
        print("  python AUTORESEARCH/run_research.py --status")
        print("  python AUTORESEARCH/run_research.py --dashboard")


def generate_dashboard():
    """Generate the HTML dashboard from current status."""
    # Import here to avoid circular dependency
    from AUTORESEARCH.dashboard_gen import build_dashboard
    build_dashboard()


if __name__ == "__main__":
    main()
