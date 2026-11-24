import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from iacs.client.agent_client import SyncAgentClient

def main():
    iacs = SyncAgentClient("gemini", "gemini")
    connected = iacs.connect()
    if not connected:
        return

    messages = iacs.poll_messages()
    if not messages:
        print("No new messages.")
    for msg in messages:
        print(f"[{msg['sender']}] {msg['type']}: {msg['payload']}")
        iacs.ack_message(msg['id'])
    iacs.close()

if __name__ == "__main__":
    main()
