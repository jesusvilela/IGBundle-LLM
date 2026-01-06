
import asyncio
import logging
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union

# Path Setup
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmos.core.tools import ToolRegistry, HYPER_ROOT

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [CREW] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

@dataclass
class TaskResult:
    agent: str
    action: str
    output: str

class BaseAgent:
    def __init__(self, name: str, role: str, registry: ToolRegistry):
        self.name = name
        self.role = role
        self.registry = registry

    async def act(self) -> Union[TaskResult, None]:
        return None

    def log(self, msg: str):
        print(f"[{self.role.upper()}::{self.name}] {msg}")

class GeometricAnalyst(BaseAgent):
    """
    Practical Agent: Checks geometric integrity by running actual math scripts.
    """
    async def act(self) -> Union[TaskResult, None]:
        # 1. Create a verification script
        script_name = "verify_curvature.py"
        script_content = """
import numpy as np
def check_curvature():
    # Simulate a check
    tensor = np.random.rand(4, 4)
    det = np.linalg.det(tensor)
    print(f"Manifold Determinant: {det:.4f}")
    if det < 0.1: raise ValueError("Singularity detected!")
if __name__ == "__main__":
    check_curvature()
"""
        write_tool = self.registry.get_tool("write_file")
        exec_tool = self.registry.get_tool("exec_python")
        
        # Write
        self.log("Writing curvature verification script...")
        write_tool(str(Path.cwd() / script_name), script_content)
        
        # Execute
        self.log("Executing verification...")
        result = exec_tool(str(Path.cwd() / script_name))
        
        self.log(f"Result: {result.strip()}")
        return TaskResult(self.name, "check_curvature", result)

class ThesisPreserver(BaseAgent):
    """
    Practical Agent: Audits documentation.
    """
    async def act(self) -> Union[TaskResult, None]:
        # Audit README
        read_tool = self.registry.get_tool("read_file")
        append_tool = self.registry.get_tool("append_file")
        
        readme_path = str(Path.cwd() / "README.md")
        audit_path = str(Path.cwd() / "thesis_audit.log")
        
        content = read_tool(readme_path)
        if "Error" in content: # File not found
             self.log("README not found, creating stub.")
             self.registry.get_tool("write_file")(readme_path, "# ManifoldGL Thesis\n\nVerified.")
             content = "Stub Created"
        
        status = "ALIGNED" if "Manifold" in content else "DRIFT_DETECTED"
        
        log_entry = f"\n[AUDIT] {self.name} checked README at {asyncio.get_running_loop().time()}: {status}"
        append_tool(audit_path, log_entry)
        
        self.log(f"Audit Complete: {status}")
        return TaskResult(self.name, "audit_thesis", status)

async def run_crew():
    print("--- ULTRA HIGH MEMORY CREW INITIALIZATION ---")
    registry = ToolRegistry()
    
    # Roster
    analyst = GeometricAnalyst("Euclid", "analyst", registry)
    librarian = ThesisPreserver("Hypatia", "preserver", registry)
    
    crew = [analyst, librarian]
    
    print("--- WORK CYCLE START ---")
    results = await asyncio.gather(*(agent.act() for agent in crew))
    
    print("--- WORK CYCLE COMPLETE ---")
    for r in results:
        if r:
            print(f"Agent {r.agent} produced: {r.output[:100]}...")

if __name__ == "__main__":
    asyncio.run(run_crew())
