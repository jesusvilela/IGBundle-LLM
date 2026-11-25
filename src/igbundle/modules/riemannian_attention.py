"""Opt-in distance attention on the prescribed Poincare ball."""

import math

import torch
from torch import nn

from .ops import compute_poincare_distance


class PoincareDistanceAttention(nn.Module):
    """Score supplied ball coordinates by negative squared distance.

    Inputs are query (B, H, T, D), key (B, H, S, D), and value (B, H, S, V).
    The optional broadcastable additive mask uses negative infinity for
    excluded keys. A fully excluded row returns zero. Values are averaged
    in their supplied vector space.
    """

    def __init__(self, curvature=1.0, temperature=1.0):
        super().__init__()
        if not math.isfinite(curvature) or curvature <= 0:
            raise ValueError("curvature must be finite and positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.curvature = curvature
        self.temperature = temperature

    def forward(self, query, key, value, attention_mask=None):
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Expected four dimensions for query, key and value")
        if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
            raise ValueError("Batch, head, and key/value sequence dimensions must agree")
        if key.shape[-2] == 0:
            raise ValueError("Expected at least one key")
        for tensor in (key, value):
            if tensor.dtype != query.dtype or tensor.device != query.device:
                raise ValueError("Query, key and value must share dtype and device")
        if not query.is_floating_point():
            raise ValueError("Query, key and value must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError("Value vectors must be finite")

        root = math.sqrt(self.curvature)
        q = query.float() if query.dtype in (torch.float16, torch.bfloat16) else query
        k = key.to(q.dtype)
        distance = compute_poincare_distance(root * q, root * k) / root
        scores = -distance.square() / self.temperature
        if not torch.isfinite(scores).all():
            raise ValueError("Attention scores overflowed at this precision")
        if attention_mask is not None:
            if not attention_mask.is_floating_point():
                raise ValueError("Expected a floating-point additive attention mask")
            scores = scores + torch.broadcast_to(attention_mask.to(scores), scores.shape)
            if torch.isnan(scores).any() or torch.isposinf(scores).any():
                raise ValueError("Attention mask produced invalid scores")
            empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
            weights = scores.masked_fill(empty, 0).softmax(dim=-1).masked_fill(empty, 0)
        else:
            weights = scores.softmax(dim=-1)
        return (weights @ value.to(scores.dtype)).to(value.dtype)
