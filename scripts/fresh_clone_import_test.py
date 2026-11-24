"""E2: fresh-clone import simulation.

Extracts src/igbundle from origin/dev-multimodal-physics into a temp dir
(via git archive, simulating a clean clone) and probes whether the package
and its key submodules can be located by the import system.

This is the gold-standard test for the HF MODEL_CARD's reproducibility claim:
"Code commit required to instantiate: 579dd7c ... instantiates the architecture".
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

REPO = r"H:\LLM-MANIFOLD\igbundle-llm"
COMMIT = "origin/dev-multimodal-physics"

# Modules the HF card's loading recipe + train_refined_hf.py require
REQUIRED_MODULES = [
    "igbundle",
    "igbundle.core.config",
    "igbundle.geometry.poincare",
    "igbundle.geometry.riemannian",
    "igbundle.geometry.hyperbolic",
    "igbundle.modules.geometric_adapter",
    "igbundle.modules.ops",
    "igbundle.dynamics.hamiltonian",
    "igbundle.dynamics.potential",
    "igbundle.fibers.executor",
    "igbundle.fibers.latent_store",
    "igbundle.cognition.meta",
    "igbundle.telemetry.manifold_stack",
    "igbundle.steering.gsp",
    "igbundle.integrations.hf_patch",
]


def main():
    tmp = os.path.join(tempfile.gettempdir(), "igbundle_clone_check")
    if os.path.exists(tmp):
        import shutil
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    # git archive extracts exactly what's in the pushed commit
    subprocess.run(
        ["git", "archive", COMMIT, "src/igbundle"],
        cwd=REPO, check=True,
        stdout=open(os.path.join(tmp, "_pkg.tar"), "wb"),
    )
    subprocess.run(["tar", "-xf", "_pkg.tar"], cwd=tmp, check=True)

    src_path = os.path.join(tmp, "src")
    sys.path.insert(0, src_path)
    print(f"sys.path[0] = {src_path}")
    print(f"Files present: {sum(1 for _ in os.walk(os.path.join(src_path, 'igbundle')) for f in _[2] if f.endswith('.py'))}")
    print()

    missing = []
    for m in REQUIRED_MODULES:
        try:
            spec = importlib.util.find_spec(m)
            status = "OK" if spec is not None else "MISSING"
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            status = f"ERROR ({type(exc).__name__})"
            spec = None
        if spec is None:
            missing.append(m)
        print(f"  {status:8} {m}")

    print()
    print(f"RESULT: {len(missing)} missing out of {len(REQUIRED_MODULES)} required")
    if missing:
        print("REPRODUCIBILITY GAP CONFIRMED — the HF card's loading recipe "
              "CANNOT work against this commit.")
        sys.exit(1)
    else:
        print("All required modules importable — reproducibility claim holds.")
        sys.exit(0)


if __name__ == "__main__":
    main()
