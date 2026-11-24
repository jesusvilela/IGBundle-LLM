import sys, os
sys.path.append(os.getcwd())
from iacs.client.agent_client import SyncAgentClient

def main():
    iacs = SyncAgentClient("gemini", "gemini")
    iacs.connect()
    
    msg = """Claude, I've monitored the training run. It unfortunately hit an Out Of Memory (OOM) error around step 495.
`monitor_state.json` reports:
CUDA error: out of memory.

`dashboard_state.json` ended abruptly with:
"Step 495 | Loss 9.6075 | Geo 234.70"

It looks like the 500 step resume didn't quite make it. We need to investigate this OOM, maybe adjust batch size or gradient accumulation? Please advise or make the necessary adjustments to `train_odyssey_v3.py`."""
    
    iacs.send_answer("claude_code", msg)
    iacs.send_task("claude_code", "Investigate the OOM error at step 495 and adjust training parameters. Let me know when you relaunch.")
    
    print("OOM Notification sent to Claude.")
    iacs.close()

if __name__ == "__main__":
    main()