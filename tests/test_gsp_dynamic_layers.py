import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from igbundle.steering.gsp import create_gsp


class DummyModel(nn.Module):
    def __init__(self, layer_count=42):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(layer_count)])


class TestGSPDynamicLayers(unittest.TestCase):
    def test_non_qwen_layer_band_maps_to_gsp_weights(self):
        gsp, _ = create_gsp(hidden_dim=8)
        attached = gsp.attach(DummyModel(), range(15, 28))

        self.assertEqual(attached[0], 15)
        self.assertEqual(attached[-1], 27)
        self.assertEqual(gsp._layer_weight_index[15], 0)
        self.assertEqual(gsp._layer_weight_index[27], len(gsp.genome.layer_weights) - 1)

        gsp._cached_lambda = 0.05
        delta = gsp._delta_h(torch.randn(1, 2, 8), 21)
        self.assertGreater(float(delta.norm().item()), 0.0)


if __name__ == "__main__":
    unittest.main()