import math
import unittest

import torch

from igbundle.modules.riemannian_attention import PoincareDistanceAttention


class PoincareDistanceAttentionTest(unittest.TestCase):
    def test_rectangular_attention_matches_analytic_radial_distances(self):
        # All points lie on a diameter, so the signed radial coordinate
        # 2 atanh(sqrt(c) x) / sqrt(c) measures geodesic length.
        c, temperature = 2.0, 0.7
        q = torch.tensor([[[[0.0, 0.0], [0.2, 0.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[-0.1, 0.0], [0.2, 0.0], [0.45, 0.0]]]], dtype=q.dtype)
        v = torch.eye(3, dtype=q.dtype).reshape(1, 1, 3, 3)
        q_radial = 2 * (math.sqrt(c) * q[..., 0]).atanh() / math.sqrt(c)
        k_radial = 2 * (math.sqrt(c) * k[..., 0]).atanh() / math.sqrt(c)
        expected = (-(q_radial.unsqueeze(-1) - k_radial.unsqueeze(-2)).square()
                    / temperature).softmax(-1)
        actual = PoincareDistanceAttention(c, temperature)(q, k, v)
        self.assertEqual(actual.shape, (1, 1, 2, 3))
        torch.testing.assert_close(actual, expected)

    def test_additive_mask_with_empty_row_is_finite_in_forward_and_backward(self):
        q = torch.zeros(2, 1, 2, 2, requires_grad=True)
        k = torch.tensor([[[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]]]).expand(2, -1, -1, -1)
        v = torch.arange(6, dtype=q.dtype).reshape(1, 1, 3, 2).expand(2, -1, -1, -1)
        mask = torch.tensor([[0., -torch.inf, -torch.inf], [-torch.inf, -torch.inf, -torch.inf]])
        actual = PoincareDistanceAttention()(q, k, v, mask)
        torch.testing.assert_close(actual[:, :, 0], v[:, :, 0])
        torch.testing.assert_close(actual[:, :, 1], torch.zeros_like(actual[:, :, 1]))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_outside_curvature_scaled_ball_fails_closed(self):
        point = torch.tensor([[[[0.6, 0.0]]]])
        with self.assertRaisesRegex(ValueError, "inside the unit ball"):
            PoincareDistanceAttention(curvature=4)(point, point, point)

    def test_bfloat16_uses_float32_scores_and_preserves_value_dtype(self):
        q = torch.tensor([[[[0.25, 0.0]]]], dtype=torch.bfloat16)
        k = torch.tensor([[[[0.0, 0.0], [0.5, 0.0]]]], dtype=q.dtype)
        v = torch.tensor([[[[1.0], [2.0]]]], dtype=q.dtype)
        attention = PoincareDistanceAttention()
        expected = attention(q.float(), k.float(), v.float()).to(v.dtype)
        torch.testing.assert_close(attention(q, k, v), expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
