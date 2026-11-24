import time, sys, os, json
sys.path.append(os.getcwd())
from iacs.client.agent_client import SyncAgentClient

iacs = SyncAgentClient("gemini", "gemini")
iacs.connect()

print("Waiting for Claude's response...")
for i in range(30):
    messages = iacs.poll_messages(limit=10)
    if messages:
        for msg in messages:
            print(f"NEW MESSAGE: {msg['payload']}")
            iacs.ack_message(msg['id'])
        iacs.close()
        sys.exit(0)
    time.sleep(15)

print("No response yet.")
iacs.close()
