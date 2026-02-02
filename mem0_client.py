"""
LLMOS Memory Client (Ported for App.py)
Wraps mem0 for persistent memory in the ManifoldGL demo.
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llmos.memory")

class LLMOSMemory:
    """
    LLMOS Memory Backend powered by mem0.
    """
    
    def __init__(self, config_path: str = "memory_config.yaml"):
        """Initialize memory backend."""
        self.config_path = config_path
        self.config = self._default_config() # Use defaults primarily for demo
        self.memory = self._init_mem0()
        logger.info("LLMOS Memory initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration optimized for local Ollama."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "igbundle_memories_v2",
                    "path": "./memory/qdrant_data"
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "qwen2.5:0.5b",
                    "temperature": 0.1
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text:latest"
                }
            },
            "memory": {
                "user_id": "demo_user",
                "auto_extract": False
            }
        }
    
    def _init_mem0(self):
        """Initialize mem0 Memory instance."""
        try:
            from mem0 import Memory
            
            mem0_config = {
                "vector_store": self.config.get("vector_store"),
                "llm": self.config.get("llm"),
                "embedder": self.config.get("embedder"),
            }
            return Memory.from_config(mem0_config)
        except ImportError:
            logger.error("mem0 not installed. Please `pip install mem0ai`.")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize mem0: {e}")
            return None
    
    def add(self, content: str, user_id: str = "demo_user", metadata: Optional[Dict] = None):
        if not self.memory: return {"error": "Memory not initialized"}
        try:
            # Truncate content to avoid embedding context overflow (Nomic limit ~8k, safe 4k)
            safe_content = content[:4000] if isinstance(content, str) else str(content)[:4000]
            
            # mem0's add method signature varies by version, but usually takes message or string
            # We wrap it as a message list for best compatibility
            messages = [{"role": "user", "content": safe_content}]
            return self.memory.add(messages, user_id=user_id, metadata=metadata)
        except Exception as e:
            logger.error(f"Memory Add Error: {e}")
            return {"error": str(e)}

    def search(self, query: str, user_id: str = "demo_user", limit: int = 2):
        if not self.memory: return []
        try:
            # Truncate query
            safe_query = query[:4000] if isinstance(query, str) else str(query)[:4000]
            
            results = self.memory.search(safe_query, user_id=user_id, limit=limit)
            return results if results else [] # mem0 returns dict or list? usually dict with 'results' key or list
        except Exception as e:
            logger.warning(f"Memory Search Failed (Non-fatal): {e}")
            return []

    def get_context_string(self, query: str, user_id: str = "demo_user") -> str:
        """Get formatted context string for LLM prompts."""
        results = self.search(query, user_id=user_id)
        if not results:
            return ""
        
        # Parse results (handle dict/list variations)
        memories = []
        if isinstance(results, dict) and "results" in results:
            item_list = results["results"]
        elif isinstance(results, list):
            item_list = results
        else:
            item_list = []

        for item in item_list:
            text = item.get("memory", "")
            if text:
                memories.append(f"- {text}")
        
        if not memories:
            return ""
            
        return "Relevant Memories:\n" + "\n".join(memories)

if __name__ == "__main__":
    # Test
    m = LLMOSMemory()
    m.add("I am researching Riemannian Manifolds.")
    print(m.search("What am I researching?"))
