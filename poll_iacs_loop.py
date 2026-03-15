"""
IACS 15-second polling loop for Claude Code.
Logs incoming messages, acks them, prints to stdout.
Run: python poll_iacs_loop.py
"""
import time, sys, os, json
from datetime import datetime

# Force unbuffered stdout for background mode
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from iacs.client.agent_client import SyncAgentClient

POLL_INTERVAL = 15  # seconds
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(_PROJECT_ROOT, "claude_iacs_inbox.log")

def main():
    print(f"[IACS-POLL] Starting 15s polling loop as claude_code...")
    
    while True:
        try:
            iacs = SyncAgentClient("claude_code", "claude")
            connected = iacs.connect()
            if not connected:
                print(f"[{datetime.now().isoformat()}] Connection failed, retry in {POLL_INTERVAL}s")
                time.sleep(POLL_INTERVAL)
                continue

            msgs = iacs.poll_messages(limit=20)
            if msgs:
                # Filter out our own messages
                incoming = [m for m in msgs if m["sender"] != "claude_code"]
                if incoming:
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        for msg in incoming:
                            ts = datetime.now().isoformat()
                            entry = f"\n{'='*60}\n[{ts}] {msg['type'].upper()} from {msg['sender']}\nID: {msg['id']}\n"
                            entry += json.dumps(msg["payload"], indent=2, ensure_ascii=False)[:800]
                            entry += "\n"
                            f.write(entry)
                            print(f"[{ts}] NEW: [{msg['type']}] from {msg['sender']}: {json.dumps(msg['payload'])[:120]}")
                            iacs.ack_message(msg["id"])
                    print(f"[{datetime.now().isoformat()}] Processed {len(incoming)} messages")
                else:
                    pass  # silent when no new messages

            iacs.close()
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Poll error: {e}")
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
