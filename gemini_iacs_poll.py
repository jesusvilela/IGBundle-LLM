import time
import sys
import os
import json
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from iacs.client.agent_client import SyncAgentClient

def main():
    while True:
        try:
            iacs = SyncAgentClient("gemini", "gemini")
            connected = iacs.connect()
            if not connected:
                print(f"[{datetime.now().isoformat()}] Failed to connect to IACS Server. Retrying in 15s...")
                time.sleep(15)
                continue

            messages = iacs.poll_messages(limit=10)
            if messages:
                with open("gemini_inbox.log", "a", encoding="utf-8") as f:
                    for msg in messages:
                        log_entry = f"\n--- {datetime.now().isoformat()} | ID: {msg['id']} ---\n"
                        log_entry += f"SENDER: {msg['sender']}\n"
                        log_entry += f"TYPE: {msg['type']}\n"
                        log_entry += f"PAYLOAD: {json.dumps(msg['payload'], indent=2)}\n"
                        f.write(log_entry)
                        print(f"[{datetime.now().isoformat()}] Received message from {msg['sender']}")
                        
                        # Acknowledge
                        iacs.ack_message(msg['id'])
            
            iacs.close()
            time.sleep(15)
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Error in polling loop: {e}")
            time.sleep(15)

if __name__ == "__main__":
    print("Starting Gemini IACS 15-second polling loop...")
    main()
