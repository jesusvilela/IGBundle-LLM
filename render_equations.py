import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

# Use standard MathText
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def render_eq(latex, filename):
    fig = plt.figure(figsize=(6, 1))
    # Add text centered
    fig.text(0.5, 0.5, f"${latex}$", fontsize=20, ha='center', va='center')
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Rendered {filename}")

EQUATIONS = {
    "eq_bundle": r"E \to^\pi M",
    "eq_curvature": r"\sigma(x) = \nabla_\mu \nabla_\nu \phi(x)",
    "eq_kl": r"D_{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}",
    "eq_sheaf_loss": r"\mathcal{L}_{Sheaf} = \sum \Omega_{UV} JS(s_U || s_V)",
    "eq_lambda": r"\Gamma \vdash \lambda x. M : \tau \rightarrow \sigma",
    "eq_bottleneck": r"z_{bot} = \phi(W_{in} h + b_{in})",
    "eq_js": r"JS(P,Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render equation images for thesis assets.")
    parser.add_argument(
        "--output-dir",
        default="output/thesis/equations",
        help="Directory to write rendered equations",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, tex in EQUATIONS.items():
        render_eq(tex, str(output_dir / f"{name}.png"))
