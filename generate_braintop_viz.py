"""Bounded parameter display; retained filename is not a topology certificate."""

import argparse
from collections.abc import Mapping
import hashlib
import html
import io
from pathlib import Path


MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_PROJECTION_VALUES = 4 * 1024 * 1024
NOTICE = (
    "Parameter values only: not measured topology, curvature, learned "
    "hyperbolicity, hypercomplex execution or task performance."
)


def read_projection(checkpoint_path, tensor_key=None):
    """Inspect one projection from a trusted local tensor-only checkpoint."""
    path = Path(checkpoint_path) / "adapter_weights.pt"
    # Bound the byte read before importing tensor libraries or allocating tensors.
    with path.open("rb") as stream:
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("Checkpoint exceeds the 64 MiB inspection limit")
        payload = stream.read(MAX_CHECKPOINT_BYTES + 1)
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds the 64 MiB inspection limit")
    import torch

    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise ValueError("Expected a string-keyed state dictionary")
    candidates = sorted(key for key in state
                        if key == "input_proj.weight" or key.endswith(".input_proj.weight"))
    if tensor_key is None:
        if len(candidates) != 1:
            raise ValueError(f"Select --tensor-key explicitly; candidates: {candidates}")
        tensor_key = candidates[0]
    if tensor_key not in candidates:
        raise ValueError("Tensor key must identify an exact input_proj.weight")
    value = state[tensor_key]
    if (not isinstance(value, torch.Tensor) or value.is_meta
            or value.layout != torch.strided or value.is_quantized
            or not value.is_floating_point()):
        raise ValueError("Projection must be a materialized dense floating tensor")

    if value.ndim != 2 or not value.numel() or value.numel() > MAX_PROJECTION_VALUES:
        raise ValueError("Projection must be a nonempty matrix with at most 4,194,304 values")
    if not torch.isfinite(value).all():
        raise ValueError("Projection contains nonfinite values")
    return value.detach(), tensor_key, hashlib.sha256(payload).hexdigest()


def generate_viz(checkpoint_path, output_file, lite_mode=False, node_metadata=None,
                 *, tensor_key=None):
    """Write a deterministic, self-contained HTML/SVG projection heatmap.

    No model is constructed and no complete-adapter restoration is attempted.
    Arbitrary node/activation metadata is no longer accepted as evidence.
    """
    if node_metadata is not None:
        raise ValueError("Unbound node metadata is not supported")
    output = Path(output_file)
    if output.suffix.lower() != ".html":
        raise ValueError("Output must be an .html file")
    if output.resolve() == (Path(checkpoint_path) / "adapter_weights.pt").resolve():
        raise ValueError("Output must not replace the input checkpoint")
    value, key, digest = read_projection(checkpoint_path, tensor_key)
    limit = 32 if lite_mode else 64
    rows, cols = min(limit, value.shape[0]), min(limit, value.shape[1])
    sample = value[:rows, :cols].double()
    scale = sample.abs().max().item()
    cells = []
    for row in range(rows):
        for col in range(cols):
            number = sample[row, col].item()
            strength = abs(number) / scale if scale else 0.0
            pale = round(255 * (1 - strength))
            color = f"rgb(255,{pale},{pale})" if number >= 0 else f"rgb({pale},{pale},255)"
            cells.append(
                f'<rect x="{col * 8}" y="{row * 8}" width="8" height="8" fill="{color}">'
                f'<title>row {row}, column {col}: {number:.17g}</title></rect>'
            )
    warning = html.escape(NOTICE)
    title = "Input-projection parameter prefix"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'viewBox="0 0 760 {rows * 8 + 100}" aria-labelledby="plot-title plot-desc">'
        f'<title id="plot-title">{title}</title><desc id="plot-desc">{warning}</desc>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="8" y="20" font-size="16">{title} (not topology)</text>'
        f'<g transform="translate(8,32)">{"".join(cells)}</g>'
        f'<text x="8" y="{rows * 8 + 55}" font-size="12">'
        'Parameter values, not learned geometry or performance.</text></svg>'
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title></head><body>
<h1>{title}</h1>
<p><strong>{warning}</strong></p>
<dl>
<dt>Tensor key</dt><dd>{html.escape(key)}</dd>
<dt>Full matrix shape</dt><dd>{value.shape[0]} x {value.shape[1]}</dd>
<dt>Source dtype</dt><dd>{html.escape(str(value.dtype))}</dd>
<dt>Displayed prefix</dt><dd>{rows} x {cols}; first rows and columns, not a random sample</dd>
<dt>Source file SHA-256</dt><dd>{digest}</dd>
<dt>Color scale</dt><dd>Blue negative, white zero, red positive;
displayed maximum absolute value {scale:.17g}. Each image uses its own scale.</dd>
</dl>
<p>Rows are output channels and columns are input coordinates of a stored linear
projection. No distance, neighborhood, curvature or semantic labels are inferred.
The hash binds this file's bytes, not its training history, base model, complete
adapter coverage or runtime compatibility.</p>
{svg}
</body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8", newline="\n")
    print(f"Saved parameter-only display: {output}")
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Trusted local directory containing adapter_weights.pt (max 64 MiB)")
    parser.add_argument("--tensor-key", help="Exact input projection key; required if ambiguous")
    parser.add_argument("--output", default="output/igbundle_parameters.html")
    parser.add_argument("--lite", action="store_true", help="Display at most 32 x 32 instead of 64 x 64")
    args = parser.parse_args()
    generate_viz(args.checkpoint, args.output, args.lite, tensor_key=args.tensor_key)
