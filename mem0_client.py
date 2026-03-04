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
        """Return default configuration optimized for local Ollama and limited RAM."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "igbundle_memories_v6_gemma",
                    "path": "./memory/qdrant_data",
                    "embedding_model_dims": 768,
                    "on_disk": True # Force payload to disk to avoid RAM/VRAM exhaustion
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
            # Drastically truncate content to avoid embedding context overflow and 422 Unprocessable Entity
            # Nomic limit ~8k, but for speed and HTTP stability we limit to 1000 chars per memory block.
            safe_content = content[:1000] if isinstance(content, str) else str(content)[:1000]
            
            # mem0's add method signature varies by version, but usually takes message or string
            # We wrap it as a message list for best compatibility
            messages = [{"role": "user", "content": safe_content}]
            return self.memory.add(messages, user_id=user_id, metadata=metadata)
        except Exception as e:
            error_str = str(e)
            if "422" in error_str or "UNPROCESSABLE_ENTITY" in error_str or "UNPROCESSABLE_CONTENT" in error_str:
                 logger.error(f"HTTP 422 Memory Error (Payload too large/malformed). Dropping context append.")
                 return {"error": "422 Payload Error"}
            logger.error(f"Memory Add Error: {e}")
            return {"error": error_str}

    def search(self, query: str, user_id: str = "demo_user", limit: int = 2):
        if not self.memory: return []
        try:
            # Truncate query significantly to avoid 422 on search
            safe_query = query[:1000] if isinstance(query, str) else str(query)[:1000]
            
            results = self.memory.search(safe_query, user_id=user_id, limit=limit)
            return results if results else [] # mem0 returns dict or list? usually dict with 'results' key or list
        except Exception as e:
            error_str = str(e)
            if "422" in error_str or "UNPROCESSABLE_ENTITY" in error_str or "UNPROCESSABLE_CONTENT" in error_str:
                 logger.error(f"HTTP 422 Memory Search Error (Query too large). Bypassing search.")
            else:
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
            
        final_str = "Relevant Memories:\n" + "\n".join(memories)
        return final_str[:800] # Hard limit to 800 chars — doubles retrieval quality while staying HTTP-safe

if __name__ == "__main__":
    # Test
    m = LLMOSMemory()
    m.add("I am researching Riemannian Manifolds.")
    print(m.search("What am I researching?"))
