"""Analytic reference illustration, independent of all checkpoints (stdlib only)."""

import argparse
from fractions import Fraction
import math
from pathlib import Path


def graph_curvature(x, y, saddle=False):
    """Induced Gaussian curvature for z=-(x^2+y^2)/4 or z=(x^2-y^2)/4."""
    x, y = Fraction(x), Fraction(y)
    determinant = Fraction(-1 if saddle else 1, 4)
    return determinant / (1 + (x*x + y*y) / 4)**2


def disk_curvature(x, y):
    """K=-exp(-2u) Delta u for u=log(2)-log(1-x^2-y^2)."""
    x, y = Fraction(x), Fraction(y)
    a = 1 - x*x - y*y
    if a <= 0:
        raise ValueError("Point must lie inside the unit disk")
    laplacian = 4 / a + 4 * (x*x + y*y) / a**2
    return -(a*a / 4) * laplacian


def render():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="450" style="max-width:100%;height:auto" viewBox="0 0 1020 450" role="img" aria-labelledby="title desc">',
        '<title id="title">Prescribed geometry is not learned geometry</title>',
        '<desc id="desc">Analytic reference: a downward paraboloid has positive Gaussian curvature, '
        'a saddle has variable negative curvature, and the Poincare disk metric has constant curvature minus one. '
        'No checkpoint data is used.</desc>',
        '<rect width="1020" height="450" fill="#10182b"/>',
        '<g font-family="Arial, sans-serif" fill="#eef2ff">',
        '<text x="510" y="34" text-anchor="middle" font-size="24">Prescribed geometry is not learned geometry</text>',
        '<text x="510" y="61" text-anchor="middle" font-size="15">Analytic reference only - no checkpoint data</text>',
    ]
    for center, saddle, title in ((170, False, "Downward paraboloid"), (510, True, "Saddle surface")):
        parts.append(f'<text x="{center}" y="100" text-anchor="middle" font-size="19">{title}</text>')
        for axis in (0, 1):
            for j in range(-4, 5):
                points = []
                for k in range(41):
                    coords = [j / 2, -2 + k / 10]
                    x, y = coords if axis == 0 else coords[::-1]
                    z = (x*x-y*y)/4 if saddle else -(x*x+y*y)/4
                    points.append(f"{center+33*(x-y):.2f},{205+14*(x+y)-36*z:.2f}")
                parts.append('<polyline points="' + " ".join(points) + '" fill="none" stroke="#73b9ff" stroke-width="1.2"/>')
        equation = "z = (x^2 - y^2) / 4" if saddle else "z = -(x^2 + y^2) / 4"
        sign = "-1" if saddle else "+1"
        parts.extend([
            f'<text x="{center}" y="340" text-anchor="middle" font-size="16">{equation}</text>',
            f'<text x="{center}" y="367" text-anchor="middle" font-size="15">K = {sign} / [4(1 + r^2/4)^2]</text>',
            f'<text x="{center}" y="394" text-anchor="middle" font-size="15">'
            + ("Negative, but not constant" if saddle else "Positive despite the downward opening") + '</text>',
        ])
    parts.extend([
        '<text x="850" y="100" text-anchor="middle" font-size="19">Poincare disk</text>',
        '<defs><clipPath id="disk"><circle cx="850" cy="215" r="105"/></clipPath></defs>',
        '<circle cx="850" cy="215" r="105" fill="none" stroke="#73b9ff" stroke-width="2"/>',
        '<g clip-path="url(#disk)" fill="none" stroke="#83e6ba" stroke-width="1.6">',
        '<path d="M745 215H955 M850 110V320"/>',
    ])
    for a in (-2, -1.5, 1.5, 2):
        radius = 105 * math.sqrt(a*a - 1)
        parts.append(f'<circle cx="{850+105*a}" cy="215" r="{radius:.4f}"/>')
    parts.extend([
        '</g>',
        '<text x="850" y="340" text-anchor="middle" font-size="15">ds^2 = 4(dx^2 + dy^2)/(1-r^2)^2</text>',
        '<text x="850" y="367" text-anchor="middle" font-size="16">K = -1 for r &lt; 1</text>',
        '<text x="850" y="394" text-anchor="middle" font-size="15">Orthogonal-circle geodesics</text>',
        '<text x="510" y="433" text-anchor="middle" font-size="14">Hypercomplex algebra additionally needs a specified product and verified laws; a picture does not supply them.</text>',
        '</g></svg>',
    ])
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(), encoding="utf-8", newline="\n")
