import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class IACSv2Client:
    """
    Lightweight serverless file-based IACS client.
    Reads/writes JSON files in shared directories to communicate safely.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.base_dir = Path("iacs")
        self.inbox_dir = self.base_dir / "inbox" / agent_id
        self.broadcast_dir = self.base_dir / "inbox" / "broadcast"
        self.archive_dir = self.base_dir / "archive"
        
        # Ensure directories exist
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.broadcast_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _write_message(self, recipient: str, msg_type: str, payload: Dict[str, Any], priority: int = 2) -> str:
        msg_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        
        # Format payload to match expected structure
        message = {
            "id": msg_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sender": self.agent_id,
            "recipient": recipient,
            "type": msg_type,
            "priority": priority,
            "payload": payload,
            "version": "2.0.0"
        }
        
        filename = f"{timestamp}_{msg_id}_{msg_type}.json"
        
        if recipient == "__broadcast__":
            target_dir = self.broadcast_dir
        else:
            target_dir = self.base_dir / "inbox" / recipient
            target_dir.mkdir(parents=True, exist_ok=True)
            
        target_path = target_dir / filename
        
        # Atomic write: write to .tmp then rename
        tmp_path = target_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=2)
            
        os.rename(tmp_path, target_path)
        return msg_id

    def send_task(self, recipient: str, content: str, topic: Optional[str] = None) -> str:
        return self._write_message(recipient, "task", {"content": content, "topic": topic})

    def send_answer(self, recipient: str, content: str, topic: Optional[str] = None) -> str:
        return self._write_message(recipient, "answer", {"content": content, "topic": topic})

    def send_finding(self, content: str, topic: Optional[str] = None) -> str:
        return self._write_message("__broadcast__", "finding", {"content": content, "topic": topic})

    def poll_messages(self, process_callback=None) -> List[Dict[str, Any]]:
        """
        Poll for new messages in inbox and broadcast.
        If process_callback is provided, it processes and archives the message.
        """
        messages = []
        
        # Check specific inbox and broadcast
        for scan_dir in [self.inbox_dir, self.broadcast_dir]:
            if not scan_dir.exists():
                continue
                
            for file_path in scan_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        msg = json.load(f)
                        
                    # If this is broadcast but sent by ourselves, ignore
                    if msg.get("sender") == self.agent_id and scan_dir == self.broadcast_dir:
                        continue
                        
                    messages.append({"path": file_path, "message": msg})
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
        # Sort by timestamp
        messages.sort(key=lambda x: x["message"].get("timestamp", ""))
        
        processed_msgs = []
        for item in messages:
            msg = item["message"]
            file_path = item["path"]
            
            if process_callback:
                # Process the message
                process_callback(msg)
                # Archive it
                archive_path = self.archive_dir / file_path.name
                try:
                    os.rename(file_path, archive_path)
                except FileExistsError:
                    # Rename with timestamp suffix if exists
                    new_name = f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
                    os.rename(file_path, self.archive_dir / new_name)
                    
            processed_msgs.append(msg)
            
        return processed_msgs
