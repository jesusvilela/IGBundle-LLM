import argparse
import sys
import subprocess
import threading
import time
import os
import signal
import queue
import re
from datetime import datetime
import psutil

# Rich imports
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.syntax import Syntax
from rich.align import Align

console = Console()

# Configuration
SERVER_PORT = 8000
MODEL_PATH = ""
BENCHMARKS = []
LIMIT = 10
PYTHON_EXE = sys.executable

# Queues for inter-thread communication
log_queue = queue.Queue()

class BenchmarkMonitor:
    def __init__(self):
        self.server_process = None
        self.bench_process = None
        self.running = True
        self.layout = Layout()
        self.console_logs = []
        self.model_activity = []
        self.current_task = "Initializing..."
        self.progress_str = "Waiting to start..."
        self.system_stats = {"cpu": 0, "ram": 0, "tps": 0.0}
        
        # Init layout
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8)
        )
        self.layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2) # Give logs more space
        )
        self.layout["left"].split(
            Layout(name="progress", ratio=1),
            Layout(name="system", size=8)
        )

    def start_server(self):
        """Starts the llama-cpp-python server (Only in Local Mode)."""
        if self.base_url:
            self.log_message(f"[yellow]Remote Mode[/]: Skipping local server start (Target: {self.base_url})")
            return

        cmd = [
            PYTHON_EXE, "-m", "llama_cpp.server",
            "--model", MODEL_PATH,
            "--port", str(SERVER_PORT),
            "--n_gpu_layers", "99",
            "--n_ctx", "8192"
        ]
        self.log_message(f"Starting server on port {SERVER_PORT}...")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        # Reader thread
        t = threading.Thread(target=self._read_stream, args=(self.server_process.stdout, "SERVER"))
        t.daemon = True
        t.start()

    def start_benchmark(self):
        """Starts the benchmark script (Local Backend)."""
        self.log_message("[green]Using Local llama-cpp-python Backend (Probs Enabled)...[/]")
        
        # Run benchmarks
        # We assume run_benchmarks_server.py is in the same dir
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_benchmarks_server.py")
        
        cmd = [
            PYTHON_EXE, script_path,
        ]
        
        # Always pass model_path as it is required by run_benchmarks_server.py
        # Use 'qwen2.5:7b' as sensible default for Ollama if not specified
        m_path = MODEL_PATH if MODEL_PATH and MODEL_PATH != "Remote Model" else "qwen2.5:7b"
        cmd.extend(["--model_path", m_path])
             
        if self.base_url:
             cmd.extend(["--base_url", self.base_url])
             
        # Add benchmarks
        cmd.append("--benchmarks")
        cmd.extend(BENCHMARKS) # Global list
        
        # Add limit
        cmd.extend(["--limit", str(LIMIT).strip()])
        
        self.log_message(f"Run Cmd: {' '.join(cmd)}")
        
        self.bench_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout for simplicity
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        t = threading.Thread(target=self._read_stream, args=(self.bench_process.stdout, "BENCH"))
        t.daemon = True
        t.start()

    def _read_stream(self, stream, source):
        if stream:
            for line in iter(stream.readline, ''):
                if not line: break
                stripped = line.strip()
                if stripped: # Only queue non-empty lines
                    log_queue.put((source, stripped))
            stream.close()

    def log_message(self, msg):
        log_queue.put(("SYSTEM", msg))


        self.tps_history = []
        self.max_history = 60
        self.current_qna = ("Thinking...", "...")

    def _generate_sparkline(self, data):
        if not data: return ""
        chars = " ▂▃▄▅▆▇█"
        min_v, max_v = min(data), max(data)
        if max_v == min_v: max_v += 1
        
        line = ""
        for x in data:
            idx = int((x - min_v) / (max_v - min_v) * (len(chars) - 1))
            line += chars[idx]
        return line

    def read_monitor_state(self):
        """
        Reads the monitor state. 
        Now extracts TPS directly from logs (stderr) since we removed the monkey patch.
        """
        tps = 0.0
        latest_q = "Running native benchmarks..."
        latest_a = "Monitoring stderr..."
        
        try:
            # Fallback: Read JSON if it exists/was updated recently (within 5 seconds)
            if os.path.exists(STATE_FILE):
                 try:
                     if time.time() - os.path.getmtime(STATE_FILE) < 5:
                         with open(STATE_FILE, 'r') as f:
                            data = json.load(f)
                            return data
                 except:
                     pass
            
            # Primary: Parse logs for tqdm progress
            # Looking for pattern: "1.23it/s" or "5.67s/it"
            # We need to read from the console_logs list
            logs = self.console_logs[-20:] # Get last 20 logs
            for line in reversed(logs):
                # Check for it/s
                match_it = re.search(r'(\d+\.\d+)it/s', line)
                if match_it:
                    tps = float(match_it.group(1))
                    latest_a = f"Rate: {tps} it/s"
                    break
                
                # Check for s/it (inverse TPS)
                match_sit = re.search(r'(\d+\.\d+)s/it', line)
                if match_sit:
                    val = float(match_sit.group(1))
                    if val > 0:
                        tps = 1.0 / val
                        latest_a = f"Rate: {val} s/it"
                    break
                    
        except Exception:
            pass
            
        return {
            "tps": tps,
            "latest_q": latest_q,
            "latest_a": latest_a,
            "timestamp": time.time()
        }

    def update_data(self):
        # Read file state
        try:
            state = self.read_monitor_state()
            tps = state.get("tps", 0)
            self.system_stats["tps"] = tps
            self.tps_history.append(tps)
            if len(self.tps_history) > self.max_history: 
                self.tps_history.pop(0)
                
            self.current_qna = (state.get("latest_q", ""), state.get("latest_a", ""))
        except: pass

        try:
            while True:
                source, line = log_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                if source == "SERVER":
                    pass 
                elif source == "BENCH":
                    if "%|" in line: self.progress_str = line
                    elif "Selected tasks:" in line: self.current_task = line.replace("Selected tasks:", "").strip()
                    elif "QnA_LOG:" in line: pass # Handled by file now
                    elif "Error:" in line: self.console_logs.append(f"[bold red]ERR[/] {line}")
                    else: self.console_logs.append(f"[blue]BENCH[/] {line}")
                elif source == "SYSTEM":
                     self.console_logs.append(f"[yellow]SYS[/] {line}")
        except queue.Empty: pass
        
        if len(self.console_logs) > 20: self.console_logs = self.console_logs[-20:]
        self.system_stats["cpu"] = psutil.cpu_percent()
        self.system_stats["ram"] = psutil.virtual_memory().percent

    def generate_layout(self):
        self.update_data()
        
        # Header
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(f"[b]LLM Benchmark Monitor (v2.0)[/] | Model: [cyan]{os.path.basename(MODEL_PATH)}[/] | Port: [green]{SERVER_PORT}[/]")
        self.layout["header"].update(Panel(grid, style="white on blue"))
        
        # Performance Graph + Scores
        spark = self._generate_sparkline(self.tps_history)
        avg_tps = sum(self.tps_history)/len(self.tps_history) if self.tps_history else 0
        
        # Try to read incremental scores
        scores_text = "[dim]Waiting for results...[/]"
        try:
             if os.path.exists("server_benchmark_results.json"):
                with open("server_benchmark_results.json", "r") as f:
                    res = json.load(f).get("results", {})
                    if res:
                        scores_list = []
                        for k, v in res.items():
                            acc = v.get("acc,none") or v.get("acc") or 0
                            scores_list.append(f"{k}: [bold white]{acc:.2%}[/]")
                        scores_text = "\n".join(scores_list)
        except: pass

        perf_text = f"[bold cyan]TPS History:[/]\n[green]{spark}[/]\n\nCurrent: {self.system_stats['tps']:.1f} t/s | Avg: {avg_tps:.1f} t/s\n\n[bold yellow]Current Scores:[/]\n{scores_text}"
        self.layout["progress"].update(Panel(Text.from_markup(perf_text), title="Real-time Performance & Scores", border_style="cyan"))

        # QnA Panel (Replaces System)
        q_safe = self.current_qna[0].replace("[", r"\[")
        a_safe = self.current_qna[1].replace("[", r"\[")
        q_text = f"[bold yellow]Q:[/]\n{q_safe}\n\n[bold green]A:[/]\n{a_safe}"
        self.layout["system"].update(Panel(Text.from_markup(q_text), title="Live Inference Stream", border_style="magenta"))

        # Logs - Fix Markup Rendering
        log_str = "\n".join(self.console_logs)
        # Escape brackets that aren't markup tags? Hard with regex.
        # But we control the tags mostly ([blue], [red]).
        # If the log message itself has [brackets], it might break.
        # For now, trust Text.from_markup handles balanced tags.
        self.layout["right"].update(Panel(Text.from_markup(log_str), title="Logs", border_style="white"))
        
        # Footer
        res_grid = Table.grid(expand=True, padding=(0,2))
        res_grid.add_column("Stat", justify="right")
        res_grid.add_column("Val", justify="left")
        res_grid.add_row("CPU:", f"{self.system_stats['cpu']}%")
        res_grid.add_row("RAM:", f"{self.system_stats['ram']}%")
        res_grid.add_row("Task:", self.current_task)
        res_grid.add_row("Prog:", self.progress_str)
        self.layout["footer"].update(Panel(res_grid, title="System Status", border_style="dim"))
        
        return self.layout

    def run(self):
        if not getattr(self, 'passive', False):
            self.start_server()
            self.start_benchmark()
        else:
            self.log_message("[yellow]Passive Mode[/]: Monitoring internal state...")
            
        with Live(self.generate_layout(), refresh_per_second=4, screen=True) as live:
            while True:
                live.update(self.generate_layout())
                time.sleep(0.25)
                if self.bench_process and self.bench_process.poll() is not None: pass # Keep display open

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, help="Path to GGUF model (required only for local mode)")
    parser.add_argument("--base_url", default=None, help="Base URL for remote API (e.g. http://localhost:11434/v1)")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu"])
    parser.add_argument("--limit", default=10)
    parser.add_argument("--passive", action="store_true", help="Passive mode: Do not spawn workers, just monitor state.")
    args = parser.parse_args()

    MODEL_PATH = args.model_path or "Remote Model"
    SERVER_PORT = 8000 # Unused if remote
    BENCHMARKS = args.benchmarks
    LIMIT = args.limit
    
    # If using base_url, we don't need model_path strictly
    # If passive, we require neither
    if not args.passive and not args.base_url and not args.model_path:
        parser.error("Either --model_path (Local) or --base_url (Remote) must be specified.")

    monitor = BenchmarkMonitor()
    # Inject args into instance if needed, or rely on globals?
    # The class uses globals MODEL_PATH, etc. (Bad practice but existing code).
    # I set globals above.
    
    # Passing base_url to instance
    monitor.base_url = args.base_url
    monitor.passive = args.passive
    
    monitor.run()

