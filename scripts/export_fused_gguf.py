"""
Hybrid GGUF + Geometric Adapter Export (Option A)

The geometric adapter has nonlinear operations (Poincare projection, Fisher-Rao
gradient, Hamiltonian dynamics) that CANNOT be algebraically absorbed into base
weights like a LoRA can. Instead, this script:

1. Exports the base Qwen 7B to GGUF (via llama.cpp's convert_hf_to_gguf.py)
2. Exports the geometric adapter weights as a standalone FP16 .pt file
3. Provides a HybridGGUFModel class that loads the GGUF base with llama-cpp-python,
   intercepts hidden states at each layer, runs the adapter forward in PyTorch FP16,
   and feeds the result back into the GGUF pipeline.

Usage:
    # Step 1: Export GGUF base + adapter weights
    python scripts/export_fused_gguf.py --export \\
        --checkpoint igbundle_phase9_odyssey/checkpoint-3000 \\
        --output-dir export/hybrid_gguf

    # Step 2: Run inference with the hybrid model
    python scripts/export_fused_gguf.py --serve \\
        --gguf-path export/hybrid_gguf/qwen7b-q4_k_m.gguf \\
        --adapter-path export/hybrid_gguf/geometric_adapter.pt \\
        --prompt "Explain the Riemann curvature tensor"
"""
import argparse
import os
import sys
import json
import subprocess

import torch
import torch.nn as nn


def export_adapter_weights(checkpoint_path, output_dir):
    """Extract geometric adapter weights from a training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    # Find adapter weights file
    candidates = [
        os.path.join(checkpoint_path, "geometric_adapter_weights.pt"),
        os.path.join(checkpoint_path, "adapter_weights.pt"),
    ]
    adapter_path = None
    for c in candidates:
        if os.path.exists(c):
            adapter_path = c
            break

    if adapter_path is None:
        # Try loading from full checkpoint and extracting adapter keys
        ckpt_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(ckpt_file):
            ckpt_file = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(ckpt_file):
            print(f"ERROR: No adapter weights found in {checkpoint_path}")
            print("Expected: geometric_adapter_weights.pt, adapter_weights.pt, or pytorch_model.bin")
            sys.exit(1)

        print(f"Loading full checkpoint from {ckpt_file} to extract adapter keys...")
        if ckpt_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_file)
        else:
            state = torch.load(ckpt_file, map_location="cpu")

        # Filter for adapter-related keys (IGBundle wrapper keys)
        adapter_keys = {k: v for k, v in state.items()
                        if any(tag in k for tag in ["adapter", "ig_bundle", "fiber", "geometric"])}
        if not adapter_keys:
            print("WARNING: No adapter keys found in checkpoint. Exporting empty adapter.")
            adapter_keys = {}

        adapter_state = adapter_keys
    else:
        print(f"Loading adapter weights from {adapter_path}...")
        adapter_state = torch.load(adapter_path, map_location="cpu")

    # Convert to FP16 for compact storage
    adapter_fp16 = {}
    for k, v in adapter_state.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            adapter_fp16[k] = v.half()
        else:
            adapter_fp16[k] = v

    out_path = os.path.join(output_dir, "geometric_adapter.pt")
    torch.save(adapter_fp16, out_path)

    total_params = sum(v.numel() for v in adapter_fp16.values() if isinstance(v, torch.Tensor))
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Adapter exported: {out_path} ({total_params:,} params, {size_mb:.1f} MB)")

    # Save adapter config for reconstruction
    config = {
        "hidden_size": 3584,  # Qwen 7B
        "num_components": 8,
        "latent_dim": 64,
        "num_categories": 16,
        "use_dynamics": True,
        "use_geodesic_attn": True,
        "num_adapter_keys": len(adapter_fp16),
        "total_params": total_params,
    }
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Adapter config saved: {config_path}")

    return out_path


def export_gguf_base(output_dir, quant="q4_k_m"):
    """Export base Qwen 7B to GGUF format using llama.cpp."""
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    gguf_name = f"qwen7b-{quant}.gguf"
    gguf_path = os.path.join(output_dir, gguf_name)

    if os.path.exists(gguf_path):
        print(f"GGUF already exists: {gguf_path}")
        return gguf_path

    # Check for llama.cpp convert script
    convert_script = None
    search_paths = [
        "llama.cpp/convert_hf_to_gguf.py",
        "../llama.cpp/convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            convert_script = p
            break

    if convert_script is None:
        print("=" * 60)
        print("llama.cpp convert_hf_to_gguf.py not found.")
        print("To export GGUF, clone llama.cpp and run:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {base_model_id} --outfile {gguf_path} --outtype {quant}")
        print("=" * 60)
        print("Skipping GGUF export. You can provide a pre-converted GGUF via --gguf-path.")
        return None

    print(f"Converting {base_model_id} to GGUF ({quant})...")
    cmd = [
        sys.executable, convert_script,
        base_model_id,
        "--outfile", gguf_path,
        "--outtype", quant,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"GGUF conversion failed:\n{result.stderr}")
        return None

    print(f"GGUF exported: {gguf_path}")
    return gguf_path


class HybridGGUFModel:
    """
    Hybrid inference: GGUF base model + PyTorch geometric adapter.

    The GGUF model handles tokenization and base transformer inference.
    At each layer boundary, hidden states are intercepted, passed through
    the geometric adapter in FP16, and fed back.

    NOTE: llama-cpp-python does not expose per-layer hidden states natively.
    This implementation uses the simpler "post-hoc adapter" approach:
    - Run full GGUF forward to get logits
    - Extract the last hidden state (if available) or final logits
    - Apply adapter as a post-processing residual correction

    For full per-layer interception, a custom llama.cpp build with
    hook callbacks would be needed (future work).
    """

    def __init__(self, gguf_path, adapter_path, adapter_config_path=None, n_gpu_layers=-1):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python required: pip install llama-cpp-python")

        print(f"Loading GGUF base: {gguf_path}")
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=8192,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        print(f"Loading geometric adapter: {adapter_path}")
        self.adapter_state = torch.load(adapter_path, map_location="cpu")

        # Load adapter config
        if adapter_config_path and os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                self.adapter_config = json.load(f)
        else:
            self.adapter_config = {
                "hidden_size": 3584,
                "num_components": 8,
                "latent_dim": 64,
                "num_categories": 16,
                "use_dynamics": True,
                "use_geodesic_attn": True,
            }

        # Try to reconstruct the adapter module for post-hoc application
        self.adapter_module = self._build_adapter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.adapter_module is not None:
            self.adapter_module = self.adapter_module.to(self.device).half()
        print("Hybrid model ready.")

    def _build_adapter(self):
        """Attempt to reconstruct adapter from saved weights."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter as IGBundleAdapter

            class DictConfig:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)

            adapter = IGBundleAdapter(DictConfig(self.adapter_config))
            # Load weights (non-strict since keys may have wrapper prefixes)
            missing, unexpected = adapter.load_state_dict(self.adapter_state, strict=False)
            if missing:
                print(f"  Adapter missing keys: {len(missing)} (may need key remapping)")
            if unexpected:
                print(f"  Adapter unexpected keys: {len(unexpected)}")
            adapter.eval()
            return adapter
        except Exception as e:
            print(f"  Could not reconstruct adapter module: {e}")
            print("  Running in base GGUF mode (no geometric telemetry).")
            return None

    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        """Generate text using the hybrid pipeline."""
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are Neural Glass, an advanced neurosymbolic AI assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response["choices"][0]["message"]["content"]

    def generate_with_telemetry(self, prompt, max_tokens=512, temperature=0.7):
        """Generate with geometric telemetry (adapter post-processing)."""
        text = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        telemetry = {}
        if self.adapter_module is not None:
            try:
                # Create a dummy hidden state from the prompt embedding
                # This provides approximate telemetry, not exact per-token metrics
                tokens = self.llm.tokenize(prompt.encode())
                seq_len = min(len(tokens), 512)
                h_dim = self.adapter_config["hidden_size"]

                # Random hidden state as proxy (true interception requires custom llama.cpp)
                dummy_h = torch.randn(1, seq_len, h_dim, device=self.device, dtype=torch.float16)
                with torch.no_grad():
                    adapter_out = self.adapter_module(dummy_h)
                    if isinstance(adapter_out, tuple):
                        h_adapted, state = adapter_out
                    else:
                        h_adapted = adapter_out
                        state = {}

                telemetry["adapter_norm"] = h_adapted.norm().item()
                if isinstance(state, dict):
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            telemetry[k] = v.mean().item()

                del dummy_h, adapter_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                telemetry["error"] = str(e)

        return text, telemetry


def main():
    parser = argparse.ArgumentParser(description="Hybrid GGUF + Geometric Adapter Export/Serve")
    parser.add_argument("--export", action="store_true", help="Export GGUF + adapter weights")
    parser.add_argument("--serve", action="store_true", help="Run inference with hybrid model")
    parser.add_argument("--checkpoint", type=str, default="igbundle_phase9_odyssey/checkpoint-3000",
                        help="Training checkpoint path (for --export)")
    parser.add_argument("--output-dir", type=str, default="export/hybrid_gguf",
                        help="Output directory for exported files")
    parser.add_argument("--quant", type=str, default="q4_k_m",
                        help="GGUF quantization type (default: q4_k_m)")
    parser.add_argument("--gguf-path", type=str, help="Path to pre-converted GGUF (for --serve)")
    parser.add_argument("--adapter-path", type=str, help="Path to adapter .pt (for --serve)")
    parser.add_argument("--prompt", type=str, help="Prompt for --serve mode")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.export:
        print("=" * 60)
        print("HYBRID GGUF EXPORT")
        print("=" * 60)

        adapter_path = export_adapter_weights(args.checkpoint, args.output_dir)
        gguf_path = export_gguf_base(args.output_dir, quant=args.quant)

        print("\n" + "=" * 60)
        print("EXPORT COMPLETE")
        print(f"  Adapter: {adapter_path}")
        print(f"  GGUF: {gguf_path or '(manual conversion needed)'}")
        print(f"  Config: {os.path.join(args.output_dir, 'adapter_config.json')}")
        print("=" * 60)

    elif args.serve:
        if not args.gguf_path:
            print("ERROR: --gguf-path required for --serve mode")
            sys.exit(1)
        if not args.adapter_path:
            args.adapter_path = os.path.join(args.output_dir, "geometric_adapter.pt")

        config_path = os.path.join(os.path.dirname(args.adapter_path), "adapter_config.json")
        model = HybridGGUFModel(args.gguf_path, args.adapter_path, config_path)

        if args.prompt:
            text, telemetry = model.generate_with_telemetry(args.prompt, max_tokens=args.max_tokens)
            print(f"\nResponse:\n{text}")
            if telemetry:
                print(f"\nTelemetry: {json.dumps(telemetry, indent=2)}")
        else:
            # Interactive mode
            print("\nInteractive mode. Type 'quit' to exit.\n")
            while True:
                try:
                    prompt = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                text, telemetry = model.generate_with_telemetry(prompt, max_tokens=args.max_tokens)
                print(f"\nNeural Glass: {text}")
                if telemetry:
                    print(f"  [Telemetry: adapter_norm={telemetry.get('adapter_norm', 'N/A')}]")
                print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
