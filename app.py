"""Experimental inference UI; telemetry is not evidence of learned geometry."""

import os
import sys

os.environ["DISABLE_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
sys.modules["torchao"] = None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import io
import threading

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from igbundle.demo_evidence import DemoTelemetry, format_metrics
from igbundle.integrations.checkpoint import load_weights, require_checkpoint_file
from igbundle.integrations.hf_patch import wrap_hf_candidate


TELEMETRY = DemoTelemetry()
GENERATION_LOCK = threading.Lock()
MODEL = None
TOKENIZER = None
EVIDENCE_URL = "https://jesusvilela.github.io/IGBundle-LLM/wiki.html"
EXAMPLE_TASKS = {
    "Abstract Reasoning": [
        "If red squares transform into blue circles, what happens to a green triangle?",
        "Complete the pattern: A→B→C, D→E→?, where each step adds one element.",
        "A 3x3 grid has symmetry along the diagonal. If top-right is red, what is bottom-left?",
    ],
    "Logical Inference": [
        "All mammals are warm-blooded. Whales are mammals. Therefore, ___?",
        "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
        "Socrates is a man. All men are mortal. What can we infer about Socrates?",
    ],
    "Compositional Tasks": [
        "Translate to French then reverse the words: 'Hello world'",
        "Take the first letter of each word in 'Artificial General Intelligence' and form an acronym.",
        "Count the vowels in 'Riemannian Geometry' then multiply by 3.",
    ],
    "Mathematical Reasoning": [
        "If f(x) = 2x + 3, what is f(f(5))?",
        "A hyperbolic space has constant negative curvature. True or false?",
        "Complete: 1, 1, 2, 3, 5, 8, ___",
    ],
}


class HookManager:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def _capture(self, module, inputs, output):
        state = output[1]
        TELEMETRY.add(
            state.sigma.detach().float().cpu().numpy(),
            state.p.detach().float().cpu().numpy(),
        )

    def attach(self):
        from igbundle.modules.adapter import IGBundleAdapter
        self.detach()
        for module in self.model.modules():
            if isinstance(module, IGBundleAdapter):
                self.hooks.append(module.register_forward_hook(self._capture))
        if not self.hooks:
            raise ValueError("No default adapters found for telemetry capture")

    def detach(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


def _figure_image(fig):
    try:
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
            buffer.seek(0)
            with Image.open(buffer) as image:
                return image.copy()
    finally:
        plt.close(fig)


def _unavailable_plot(title, reason):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title(title)
    ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True)
    ax.axis("off")
    return _figure_image(fig)


def plot_dispersion():
    if TELEMETRY.summary() is None:
        return _unavailable_plot("Dispersion unavailable", "No adapter states captured")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(TELEMETRY.sigma_samples, bins=40)
    ax.set_title("Observed Gaussian standard deviations (not curvature)")
    ax.set_xlabel("Sigma")
    ax.set_ylabel("Sample count")
    fig.text(0.5, 0.01, "First 4096 values at most; prefix sample, not the full distribution",
             ha="center", fontsize=8)
    return _figure_image(fig)


def plot_categorical():
    data = TELEMETRY.mean_probabilities()
    if data is None:
        return _unavailable_plot("Categorical probabilities unavailable", "No adapter states captured")
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Mean categorical probability")
    ax.set_title("Observed categorical probabilities (not pairwise affinity)")
    ax.set_xlabel("Category index")
    ax.set_ylabel("Component index")
    return _figure_image(fig)


def plot_training_metrics():
    return _unavailable_plot(
        "Training evidence unavailable",
        "No checkpoint-bound training log is configured.\n"
        "Synthetic convergence curves are not substituted.",
    )


def plot_ablation_comparison():
    return _unavailable_plot(
        "Matched ablation evidence unavailable",
        "No matched ablation run is configured.\n"
        "The former hard-coded accuracy bars were withdrawn.",
    )


def load_topo_stats():
    return [
        ["Task-performance advantage", "Not established"],
        ["Learned curvature / hyperbolicity", "Not established"],
        ["Hypercomplex runtime integration", "Not established"],
        ["Telemetry meaning", "Sigma and categorical probabilities only"],
        ["Checkpoint identity", "Not certified by tensor coverage"],
    ]


def load_model(config_path, checkpoint_path):
    adapter_w_path = require_checkpoint_file(os.path.join(checkpoint_path, "adapter_weights.pt"))
    with open(config_path, "r") as stream:
        cfg = yaml.safe_load(stream)
    base_model_id = cfg["base_model_id"]

    if torch.cuda.is_available():
        # Optional backend; importing the CPU demo does not require Unsloth.
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=1024,
            load_in_4bit=True,
            trust_remote_code=True,
            device_map={"": 0},
        )
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map="cpu", trust_remote_code=True,
        )
        if os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, checkpoint_path)

    cfg["ig_adapter"]["hidden_size"] = model.config.hidden_size
    from types import SimpleNamespace
    model = wrap_hf_candidate(model, SimpleNamespace(**cfg["ig_adapter"]))
    load_weights(model, adapter_w_path)
    model.eval()
    return model, tokenizer


def generate_response(message, history):
    if MODEL is None:
        raise RuntimeError("Model is not loaded")
    # This legacy single-turn prompt does not consume chat history.
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{message}\n\n### Response:\n"
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs, max_new_tokens=256, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=TOKENIZER.eos_token_id,
        )
    return TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def generate_with_viz(message, history):
    # Global model hooks must not mix observations from concurrent requests.
    with GENERATION_LOCK:
        TELEMETRY.reset()
        response = generate_response(message, history)
        return (
            response, plot_dispersion(), plot_categorical(),
            format_metrics(TELEMETRY.summary()),
        )


def launch_app(config_path, checkpoint_path):
    global MODEL, TOKENIZER
    MODEL, TOKENIZER = load_model(config_path, checkpoint_path)
    hooks = HookManager(MODEL)

    with gr.Blocks(title="IGBundle research demo") as demo:
        gr.Markdown(f"""
        # IGBundle research demo

        Experimental single-turn inference with descriptive adapter telemetry.
        **No task-performance gain, learned hyperbolicity, geometric faithfulness,
        or efficiency advantage is established.**

        Sigma is a Gaussian standard deviation, not curvature.
        Categorical probabilities are not pairwise affinities.
        [Published evidence and corrections]({EVIDENCE_URL})
        """)

        with gr.Tab("Inference"):
            gr.Markdown("Each prompt is independent; previous turns are displayed but not sent to the model.")
            chatbot = gr.Chatbot(type="messages", height=350)
            message = gr.Textbox(label="Prompt", lines=2)
            submit = gr.Button("Generate", variant="primary")
            clear = gr.Button("Clear")
            gr.Examples(examples=[[ex] for group in EXAMPLE_TASKS.values() for ex in group],
                        inputs=[message])
            metrics = gr.Markdown("Telemetry unavailable until adapter states are captured.")
            with gr.Row():
                dispersion = gr.Image(label="Observed Gaussian dispersion")
                categorical = gr.Image(label="Observed categorical probabilities")

            def respond(text, history):
                reply, p1, p2, summary = generate_with_viz(text, history)
                return "", (history or []) + [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                ], p1, p2, summary

            message.submit(respond, [message, chatbot],
                           [message, chatbot, dispersion, categorical, metrics])
            submit.click(respond, [message, chatbot],
                         [message, chatbot, dispersion, categorical, metrics])
            clear.click(
                lambda: ([], None, None, "Telemetry unavailable."),
                outputs=[chatbot, dispersion, categorical, metrics], queue=False,
            )

        with gr.Tab("Evidence availability"):
            gr.Markdown("""
            Training curves and ablation bars are unavailable here.
            No arbitrary local JSON file is treated as verified evidence.
            Previously displayed synthetic trajectories and hard-coded gains were withdrawn.
            """)
            gr.Image(value=plot_training_metrics(), label="Training evidence")
            gr.Image(value=plot_ablation_comparison(), label="Ablation evidence")
            gr.DataFrame(value=load_topo_stats(), headers=["Question", "Status"],
                         interactive=False)

        with gr.Tab("Interpretation"):
            gr.Markdown(f"""
            The default adapter uses mixture parameters, categorical distributions
            and a prescribed Poincare-distance operator. Those implementation choices
            do not demonstrate learned hierarchy, a mathematical fiber bundle,
            sheaf laws, or hypercomplex multiplication.

            The histogram is a bounded prefix sample. Summary values aggregate all
            captured observations, not independent prompts or evaluation examples.
            No topology viewer or semantic labeling is validated by this demo.

            Tensor restoration checks do not certify the base model, PEFT component,
            training provenance or archived checkpoint compatibility.
            [Evidence ledger]({EVIDENCE_URL})
            """)

    hooks.attach()
    try:
        demo.queue(default_concurrency_limit=1).launch(share=False, server_name="0.0.0.0")
    finally:
        hooks.detach()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Experimental IGBundle inference demo")
    parser.add_argument("--config", default=os.path.join(
        base_dir, "configs", "qwen25_7b_igbundle_lora.yaml"))
    parser.add_argument("--checkpoint", required=True,
                        help="Directory containing complete adapter_weights.pt")
    args = parser.parse_args()
    launch_app(args.config, args.checkpoint)
