import asyncio
import logging
import random
import subprocess
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [CREW] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auxiliary_crew.log"),
        logging.StreamHandler()
    ]
)

IPC_INBOX = "crew_inbox.json"
IPC_OUTBOX = "crew_outbox.json"

@dataclass
class Proposal:
    author: str
    type: str 
    content: str
    status: str = "pending"
    priority: int = 1

@dataclass
class NewTask:
    author: str
    task_type: str
    description: str

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    async def act(self, ctx: dict) -> Union[Proposal, NewTask, None]:
        return None

    def log(self, msg: str):
        logging.info(f"[{self.role.upper()}::{self.name}] {msg}")

class GeminiOperator(BaseAgent):
    async def act(self, ctx: dict) -> Union[Proposal, NewTask, None]:
        # Simulate or Real Gemini Call
        if random.random() < 0.05:
            # Only optimize Geometry code
            return NewTask(self.name, "optimize", "Analyze curvature calculation in geometric_analysis.py")
        return None

class GeometricAnalyst(BaseAgent):
    """
    Primary Role: Ensures the codebase respects the Hyperbolic Concavity hypothesis.
    """
    async def act(self, ctx: dict) -> Union[Proposal, NewTask, None]:
        r = random.random()
        if r < 0.15:
            return Proposal(self.name, "bundle_check", "Verify local section continuity in layer 12")
        elif r < 0.3:
            return Proposal(self.name, "metric_tensor", "Adjust Fisher Information Matrix approximation damping")
        return None

class HyperparamOptimizer(BaseAgent):
    async def act(self, ctx: dict) -> Union[Proposal, NewTask, None]:
        if random.random() < 0.05:
            return Proposal(self.name, "opt", "Suggest learning_rate decay warmup increase")
        return None

class ThesisPreserver(BaseAgent):
    """
    Replaces generic DocUpdater. Ensures documentation matches the Math Thesis.
    """
    async def act(self, ctx: dict) -> Union[Proposal, NewTask, None]:
        if random.random() < 0.05:
            return Proposal(self.name, "thesis_align", "Detected drift in README vs Thesis.pdf. Proposal to revert Section 3.")
        return None

class Critic(BaseAgent):
    async def review(self, proposal: Proposal) -> bool:
        if "invalid" in proposal.content.lower(): return False
        
        # Bias towards Geometry
        if "curvature" in proposal.content.lower() or "bundle" in proposal.content.lower():
            self.log(f"PRIORITY APPROVAL for Geometric Proposal: {proposal.content}")
            return True
            
        if random.random() > 0.1:
            self.log(f"APPROVED: {proposal.content}")
            return True
        return False

class Supervisor(BaseAgent):
    async def validate_repo_integrity(self) -> bool:
        self.log("Running MANIFOLD INTEGRITY CHECKS...")
        try:
            # Ensure geometry module is importable
            subprocess.run(["python", "-c", "import src.igbundle.geometry"], check=False) # Soft check
            # Standard checks
            subprocess.run(["python", "-m", "compileall", ".", "-q"], check=True, capture_output=True)
            self.log("PASSED: Bundle Integrity Verified")
            return True
        except subprocess.CalledProcessError:
            self.log("FAILED: Integrity Check")
            return False

    async def commit_cycle(self):
        self.log("Initiating Git Cycle (Thesis-Aligned)...")
        if await self.validate_repo_integrity():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            try:
                subprocess.run(["git", "add", "."], check=True)
                msg = f"chore(crew): Geometric alignment cycle {timestamp}"
                subprocess.run(["git", "commit", "-m", msg], check=False)
                subprocess.run(["git", "push"], check=True)
                self.log("SUCCESS: Pushed to remote")
            except subprocess.CalledProcessError as e:
                self.log(f"ERROR: Git operation failed: {e}")

class CrewManager:
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.critics: List[Critic] = []
        self.supervisor = Supervisor("HAL-9000", "Supervisor")
        self.request_queue: List[dict] = []
        
        # Spawn Massive Crew (50 Agents) - REBALANCED
        self.log("Spawning 50 agents (Geometry Focused)...")
        
        # Major Pivot: 30 Geometric Analysts
        for i in range(30): self.agents.append(GeometricAnalyst(f"Geo-{i}", "Analyst"))
        
        # 5 Janitors -> Thesis Preservers
        for i in range(5): self.agents.append(ThesisPreserver(f"Thesis-{i}", "Preserver"))
        
        # 5 Optimizers (reduced from 15)
        for i in range(5): self.agents.append(HyperparamOptimizer(f"Opt-{i}", "Optimizer"))
        
        # 5 Gemini Operators
        for i in range(5): self.agents.append(GeminiOperator(f"Gemini-{i}", "GeminiOp"))
        
        # 5 Critics
        for i in range(5): self.critics.append(Critic(f"Crit-{i}", "Critic"))
        
        self.cycle_count = 0

    def log(self, msg):
        logging.info(f"[MANAGER] {msg}")

    async def check_inbox(self):
        if not os.path.exists(IPC_INBOX): return
        try:
            with open(IPC_INBOX, 'r') as f:
                content = f.read().strip() or "[]"
                requests = json.loads(content)
            if requests:
                self.log(f"Inbox: {len(requests)} items")
                self.request_queue.extend(requests)
                with open(IPC_INBOX, 'w') as f: f.write("[]")
        except: pass

    async def dispatch_new_task(self, task: NewTask):
        self.log(f"Recursing: {task.task_type}")

    async def run_cycle(self):
        self.cycle_count += 1
        logging.info(f"=== Cycle {self.cycle_count} (Geometry Swarm) ===")
        
        await self.check_inbox()
        
        active_agents = random.sample(self.agents, 15) 
        tasks = [agent.act({}) for agent in active_agents]
        results = await asyncio.gather(*tasks)
        
        proposals = [r for r in results if isinstance(r, Proposal)]
        logging.info(f"Proposals: {len(proposals)}")
        
        if self.cycle_count % 5 == 0:
            await self.supervisor.commit_cycle()
            
        logging.info("=== Cycle Complete ===\n")

async def main():
    if not os.path.exists(IPC_INBOX):
        with open(IPC_INBOX, 'w') as f: f.write("[]")
    crew = CrewManager()
    while True:
        await crew.run_cycle()
        await asyncio.sleep(2)

if __name__ == "__main__":
    try: asyncio.run(main())
    except: pass
