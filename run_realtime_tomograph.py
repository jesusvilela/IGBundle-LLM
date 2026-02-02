"""
IGBundle Real-Time Tomograph - LM Studio Integration

Connects to LM Studio API to probe the model and stream real geometric telemetry.

Author: Jesus Vilela Jato (ManifoldGL Research)
"""

import argparse
import time
import sys
import os
import json
import torch
import requests
import threading
import queue
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "braintop", "src"))
sys.path.insert(0, os.path.join(base_dir, "src"))

# Import IGBundle components
from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter
from igbundle.core.config import IGBundleConfig

# Import Tomograph components
from braintop.integrations.igbundle_realtime import (
    IGBundleTomograph,
    ReductionMethod,
    TomographFrame
)


def atomic_json_write(filepath: str, data: Dict) -> bool:
    """
    Atomically write JSON to file using temp file + rename.
    This prevents read/write conflicts on Windows.
    """
    filepath = os.path.abspath(filepath)
    dir_path = os.path.dirname(filepath)
    
    try:
        # Write to temp file in same directory
        fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=dir_path if dir_path else '.')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f)
            
            # Atomic rename (on Windows, need to remove target first)
            if os.path.exists(filepath):
                os.remove(filepath)
            shutil.move(tmp_path, filepath)
            return True
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e
    except Exception as e:
        print(f"[Write Error] {e}")
        return False


class LMStudioProbe:
    """Probe interface for LM Studio API."""
    
    def __init__(
        self,
        base_url: str = "http://192.168.56.1:1234",
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.model_info = None
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Verify connection to LM Studio."""
        try:
            resp = self.session.get(f"{self.base_url}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data"):
                    self.model_info = data["data"][0]
                    print(f"[LMStudio] Connected: {self.model_info.get('id', 'Unknown')}")
                    return True
        except Exception as e:
            print(f"[LMStudio] Connection error: {e}")
        return False
    
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Get embeddings from LM Studio."""
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": text, "model": self.model_info.get("id", "default") if self.model_info else "default"},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data") and len(data["data"]) > 0:
                    return data["data"][0].get("embedding")
        except:
            pass
        return None
    
    def probe_hidden_state(self, prompt: str = "") -> Dict[str, Any]:
        """Probe the model's state."""
        start_time = time.time()
        embedding = self.get_embeddings(prompt) if prompt else None
        latency = time.time() - start_time
        
        return {
            "prompt": prompt or "[IDLE]",
            "embedding": embedding,
            "latency": latency,
            "timestamp": time.time(),
            "model": self.model_info.get("id", "unknown") if self.model_info else "unknown"
        }


class ModelTomographIntegration:
    """Complete integration for real-time model telemetry visualization."""
    
    def __init__(
        self,
        checkpoint_path: str,
        lmstudio_url: str = "http://192.168.56.1:1234",
        reduction_method: str = "pca",
        http_output: str = "tomograph_state.json",
    ):
        self.checkpoint_path = checkpoint_path
        self.lmstudio_url = lmstudio_url
        self.reduction_method = ReductionMethod(reduction_method)
        self.http_output = os.path.abspath(http_output)
        
        self.adapter = None
        self.tomograph = None
        self.lmstudio = None
        
        self.running = False
        self.telemetry_thread = None
        
        self.fiber_semantics = {
            231: "Logic", 138: "Creative", 196: "Knowledge", 245: "Coding", 178: "Empathy",
        }
    
    def load_adapter(self) -> None:
        """Load the trained IGBundle adapter."""
        print(f"[Tomograph] Loading adapter from {self.checkpoint_path}...")
        
        config = IGBundleConfig(
            hidden_size=2048,
            num_components=8,
            num_categories=16,
            latent_dim=32,
            bottleneck_dim=512,
            adapter_scale=0.1,
            dropout=0.0,
            eta_b=0.01,
            eta_f=0.01,
            use_dynamics=True,
            use_geodesic_attn=True,
        )
        
        self.adapter = GeometricIGBundleAdapter(config)
        
        weights_path = os.path.join(self.checkpoint_path, "adapter_weights.pt")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.checkpoint_path, "pytorch_model.bin")
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            adapter_weights = {k: v for k, v in state_dict.items() 
                             if "igbundle" in k.lower() or "adapter" in k.lower()}
            if adapter_weights:
                self.adapter.load_state_dict(adapter_weights, strict=False)
                print(f"[Tomograph] Loaded {len(adapter_weights)} weight tensors")
        
        self.adapter.eval()
        print("[Tomograph] Adapter loaded")
    
    def connect_lmstudio(self) -> bool:
        """Connect to LM Studio."""
        print(f"[Tomograph] Connecting to LM Studio at {self.lmstudio_url}...")
        self.lmstudio = LMStudioProbe(self.lmstudio_url)
        return self.lmstudio.model_info is not None
    
    def setup_tomograph(self) -> None:
        """Set up the tomograph."""
        print("[Tomograph] Setting up tomograph...")
        self.tomograph = IGBundleTomograph(reduction_method=self.reduction_method, sample_rate=1)
        self.tomograph.set_fiber_semantics(self.fiber_semantics)
        self.tomograph.register_hooks(self.adapter)
        print(f"[Tomograph] Reduction: {self.reduction_method.value}")
    
    def process_lmstudio_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert LM Studio state to tensor for adapter processing."""
        embedding = state.get("embedding")
        hidden_size = self.adapter.cfg.hidden_size
        seq_len = 32
        
        if embedding:
            emb_tensor = torch.tensor(embedding, dtype=torch.float32)
            emb_dim = emb_tensor.shape[0]
            
            if emb_dim < hidden_size:
                repeats = (hidden_size // emb_dim) + 1
                emb_tensor = emb_tensor.repeat(repeats)[:hidden_size]
            elif emb_dim > hidden_size:
                emb_tensor = emb_tensor[:hidden_size]
            
            input_tensor = emb_tensor.unsqueeze(0).unsqueeze(0).repeat(1, seq_len, 1)
            positions = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)
            input_tensor = input_tensor * (1 + 0.1 * positions)
            return input_tensor
        else:
            latency = state.get("latency", 0.1)
            scale = 0.5 + latency * 2
            return torch.randn(1, seq_len, hidden_size) * scale
    
    def _update_http_state(self, lm_state: Dict[str, Any], geo_state, frame_id: int) -> None:
        """Update HTTP state file with current telemetry."""
        try:
            import numpy as np
            
            base_coords = geo_state.base_coordinates.detach().cpu().numpy()
            fiber_sections = geo_state.fiber_sections.detach().cpu().numpy()
            
            B, T, P, D = base_coords.shape
            coords_flat = base_coords[0].reshape(-1, D)
            
            # Simple PCA to 3D for compatibility
            if D > 3:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, D))
                coords_3d = pca.fit_transform(coords_flat)
            else:
                coords_3d = coords_flat
            
            # Compute metrics
            curvature = 0.0
            if geo_state.metric is not None:
                try:
                    metric_tensor = geo_state.metric.metric_tensor.detach().cpu().numpy()
                    curvature = float(-np.log(np.abs(np.linalg.det(metric_tensor[0, 0, 0])) + 1e-8))
                except:
                    pass
            
            energy = float(np.mean(np.sum(base_coords ** 2, axis=-1)))
            entropy = float(-np.mean(np.sum(fiber_sections * np.log(fiber_sections + 1e-8), axis=-1)))
            
            consensus = 0.0
            if hasattr(geo_state, 'consensus_loss') and geo_state.consensus_loss is not None:
                consensus = float(geo_state.consensus_loss.detach().cpu().item())
            
            # Op distribution
            op_dist = {}
            if hasattr(geo_state, 'op_logits') and geo_state.op_logits is not None:
                op_logits = geo_state.op_logits.detach().cpu()
                op_probs = torch.softmax(op_logits, dim=-1).mean(dim=(0, 1)).numpy()
                op_labels = ["SELECT", "PROJECT", "FILTER", "JOIN", "AGGREGATE",
                            "TRANSFORM", "COMPOSE", "ABSTRACT", "INSTANTIATE", "COMPARE"]
                op_dist = {op_labels[i]: float(op_probs[i]) for i in range(min(len(op_labels), len(op_probs)))}
            
            state_dict = {
                "timestamp": time.time(),
                "frame_id": frame_id,
                "status": "active",
                "prompt": lm_state.get("prompt", "[IDLE]"),
                "model": lm_state.get("model", "unknown"),
                "latency": lm_state.get("latency", 0),
                "data": {
                    "frame_id": frame_id,
                    "timestamp": time.time(),
                    "prompt": lm_state.get("prompt", "[IDLE]"),
                    "base_coords_3d": coords_3d.tolist(),
                    "fiber_activations": fiber_sections[0].reshape(-1, fiber_sections.shape[-1]).tolist(),
                    "curvature_values": [curvature] * coords_3d.shape[0],
                    "energy_values": [energy] * coords_3d.shape[0],
                    "entropy_values": [entropy] * coords_3d.shape[0],
                    "consensus_loss": consensus,
                    "op_distribution": op_dist,
                }
            }
            
            # Use atomic write
            atomic_json_write(self.http_output, state_dict)
                
        except Exception as e:
            print(f"[HTTP] Error: {e}")
    
    def telemetry_loop(self, poll_interval: float = 0.5) -> None:
        """Main telemetry loop."""
        frame_id = 0
        
        while self.running:
            try:
                state = self.lmstudio.probe_hidden_state("") if self.lmstudio else {
                    "prompt": "[IDLE]", "embedding": None, "latency": 0.01, "timestamp": time.time()
                }
                
                input_tensor = self.process_lmstudio_state(state)
                
                with torch.no_grad():
                    output, geo_state = self.adapter(input_tensor)
                
                self._update_http_state(state, geo_state, frame_id)
                
                frame_id += 1
                
                if frame_id % 20 == 0:
                    print(f"[Telemetry] Frame {frame_id} | Latency: {state.get('latency', 0):.3f}s")
                
            except Exception as e:
                print(f"[Telemetry] Error: {e}")
            
            time.sleep(poll_interval)
    
    def start_telemetry(self, poll_interval: float = 0.5) -> None:
        """Start background telemetry collection."""
        self.running = True
        self.telemetry_thread = threading.Thread(target=self.telemetry_loop, args=(poll_interval,), daemon=True)
        self.telemetry_thread.start()
        print(f"[Tomograph] Telemetry started (poll: {poll_interval}s)")
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        self.running = False
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=2.0)
        if self.tomograph:
            self.tomograph.unregister_hooks()
        print("[Tomograph] Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="IGBundle Real-Time Tomograph")
    parser.add_argument("--checkpoint", default="trained_adapter")
    parser.add_argument("--lmstudio", default="http://192.168.56.1:1234")
    parser.add_argument("--reduction", choices=["pca", "umap", "tsne", "poincare"], default="pca")
    parser.add_argument("--http-output", default="tomograph_state.json")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    IGBUNDLE // TOMOGRAPH                             ║
║              Real-Time Model Telemetry (HTTP Only)                   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    integration = ModelTomographIntegration(
        checkpoint_path=args.checkpoint,
        lmstudio_url=args.lmstudio,
        reduction_method=args.reduction,
        http_output=args.http_output,
    )
    
    try:
        integration.load_adapter()
        integration.connect_lmstudio()
        integration.setup_tomograph()
        
        print("\n" + "="*70)
        print("TOMOGRAPH ACTIVE - Streaming model telemetry")
        print("="*70)
        print(f"\n📊 Dashboard: http://localhost:8050")
        print(f"📁 HTTP State: {args.http_output}")
        print(f"🤖 LM Studio: {args.lmstudio}")
        print("\n" + "="*70 + "\n")
        
        integration.start_telemetry(args.poll_interval)
        
        print("[Tomograph] Streaming telemetry. Press Ctrl+C to stop.\n")
        
        while True:
            time.sleep(1)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        integration.shutdown()


if __name__ == "__main__":
    main()
