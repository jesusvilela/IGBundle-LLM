"""
Claude Code — Dedicated mem0 Store
Separate qdrant collection for Claude's investigation and diagnostic work.
Does NOT share state with Gemini's igbundle_memories_v5_gemma collection.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

# Redirect MEM0_DIR to avoid qdrant lock conflict with Gemini's running instance
_CLAUDE_MEM0_DIR = os.path.join(os.path.dirname(__file__), "memory", "claude_mem0_home")
os.makedirs(_CLAUDE_MEM0_DIR, exist_ok=True)
os.environ["MEM0_DIR"] = _CLAUDE_MEM0_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude.memory")

CLAUDE_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "claude_investigation_store",
            "path": "./memory/claude_qdrant_data",
            "embedding_model_dims": 768,
            "on_disk": True
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "ig-bundlellmv2-hamiltonians",
            "api_key": "not-needed",
            "openai_base_url": "http://192.168.56.1:1234/v1",
            "temperature": 0.0,
            "max_tokens": 1000
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "google/embeddinggemma-300m"
        }
    }
}

USER_ID = "claude_code"


class ClaudeMemory:
    """Claude Code's private mem0 store for investigation tracking."""

    def __init__(self):
        self.memory = self._init()

    def _init(self):
        try:
            from mem0 import Memory
            return Memory.from_config(CLAUDE_CONFIG)
        except Exception as e:
            logger.error(f"Claude mem0 init failed: {e}")
            return None

    def add(self, content: str, metadata: Optional[Dict] = None):
        if not self.memory:
            return {"error": "not initialized"}
        try:
            safe = content[:1500] if isinstance(content, str) else str(content)[:1500]
            messages = [{"role": "user", "content": safe}]
            return self.memory.add(messages, user_id=USER_ID, metadata=metadata)
        except Exception as e:
            logger.error(f"Claude mem0 add error: {e}")
            return {"error": str(e)}

    def search(self, query: str, limit: int = 5):
        if not self.memory:
            return []
        try:
            safe = query[:1000]
            return self.memory.search(safe, user_id=USER_ID, limit=limit)
        except Exception as e:
            logger.error(f"Claude mem0 search error: {e}")
            return []

    def log_finding(self, tag: str, finding: str):
        """Convenience: log a tagged investigation finding."""
        content = f"[{tag}] {finding}"
        return self.add(content, metadata={"tag": tag, "source": "claude_code"})


if __name__ == "__main__":
    m = ClaudeMemory()
    if m.memory:
        m.log_finding("INIT", "Claude mem0 store initialized. Collection: claude_investigation_store")
        results = m.search("initialization")
        print(f"Store active. Search test returned {len(results)} results.")
    else:
        print("ERROR: mem0 store failed to initialize.")
