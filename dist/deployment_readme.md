# IG-Bundle Integrated Deployment Package

This folder contains the complete runtime environment for the Neural Glass manifold interaction system.

## Contents
*   `base_model.gguf`: Quantized Qwen-7B base model.
*   `adapter_refined.pt`: Geometric IGBundle Adapter (Refined).
*   `app_neural_glass.py`: The Gradio User Interface.
*   `run_inference_integrated.py`: CLI inference script.
*   `igbundle/`: The core library with fiber refinement logic.

## Setup

1.  **Install Requirements**
    Ensure you have Python 3.10+ and CUDA drivers installed.
    Install dependencies (if not already in your environment):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install gradio llama-cpp-python transformers peft
    ```

2.  **Run the GUI (Neural Glass)**
    Launch the web interface:
    ```bash
    python app_neural_glass.py
    ```
    Access it at `http://localhost:7865`.

    *Note: The app will automatically detect `adapter_refined.pt` in this directory.*

3.  **Run CLI Inference**
    For batch processing or testing:
    ```bash
    python run_inference_integrated.py --mode refined --prompt "Explain the geometry of AI."
    ```

## Features
*   **Fiber Refinement**: The system uses Option-A Fiber Latent Refinement.
*   **Constraint Discipline**: It automatically detects and enforces semantic constraints (e.g., "Riemannian").
*   **Dual Mode**: Use `--mode fast` for raw GGUF speed, or `--mode refined` (default in App) for geometric depth.
