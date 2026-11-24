import unittest

import torch

from igbundle.modules.delta_fiber import DeltaFiberUpdate
from scripts.compare_delta_recall import make_batch


class DeltaFiberTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.module = DeltaFiberUpdate(4, 2, mem_dim=8, num_heads=2).double()
        self.inputs = torch.randn(2, 5, 3, 6, dtype=torch.float64)

    def test_causality_and_component_isolation(self):
        out = self.module(self.inputs)
        self.assertEqual(out.shape, (2, 5, 3, 2))
        torch.testing.assert_close(out[:, :3], self.module(self.inputs[:, :3]))
        for component in range(3):
            selected = slice(component, component + 1)
            torch.testing.assert_close(out[:, :, selected], self.module(self.inputs[:, :, selected]))
        self.assertGreater((out[:, -1:] - self.module(self.inputs[:, -1:])).abs().max(), 1e-6)

    def test_gates_preserve_dtype_and_detach_features(self):
        curvature = torch.tensor([-2.0, -0.5], requires_grad=True)
        entropy = torch.tensor(1.5, requires_grad=True)
        out = self.module(self.inputs, curvature, entropy)
        out.square().mean().backward()
        self.assertEqual(out.dtype, self.inputs.dtype)
        self.assertIsNone(curvature.grad)
        self.assertIsNone(entropy.grad)
        for layer in (self.module.curvature_mod, self.module.entropy_mod):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum(), 0)

    def test_empty_sequence(self):
        self.assertEqual(self.module(self.inputs[:, :0]).shape, (2, 0, 3, 2))

    def test_rejects_degenerate_one_coordinate_heads(self):
        with self.assertRaisesRegex(ValueError, "at least two coordinates"):
            DeltaFiberUpdate(4, 2, mem_dim=2, num_heads=2)

    def test_zero_gate_features_preserve_default(self):
        zeros = torch.zeros(2)
        torch.testing.assert_close(self.module(self.inputs, zeros, zeros), self.module(self.inputs))

    def test_recall_batch_query_hides_target_and_has_exactly_one_match(self):
        inputs, labels = make_batch(torch.Generator().manual_seed(23), 16)
        writes, query = inputs[:, :-1, 0], inputs[:, -1, 0]
        self.assertEqual(int(query[:, -2:].count_nonzero()), 0)
        matches = (writes[:, :, :4] * query[:, None, :4]).sum(dim=-1)
        torch.testing.assert_close(matches.sum(dim=1), torch.ones(16))
        values = (writes[:, :, -2:] * matches.unsqueeze(-1)).sum(dim=1)
        torch.testing.assert_close(values.argmax(dim=-1), labels)

    def test_erase_changes_repeated_key_reads_but_not_first_read(self):
        control = DeltaFiberUpdate(4, 2, mem_dim=8, num_heads=2, erase=False).double()
        control.load_state_dict(self.module.state_dict())
        repeated = self.inputs[:, :1].expand(-1, 2, -1, -1)
        delta_out = self.module(repeated)
        control_out = control(repeated)
        torch.testing.assert_close(delta_out[:, :1], control_out[:, :1])
        self.assertGreater((delta_out[:, 1:] - control_out[:, 1:]).abs().max(), 1e-6)

    def test_overwrite_labels_follow_most_recent_write(self):
        inputs, labels = make_batch(torch.Generator().manual_seed(23), 32, rounds=2)
        writes, query = inputs[:, :-1, 0], inputs[:, -1, 0]
        matches = (writes[:, :, :4] * query[:, None, :4]).sum(dim=-1).unflatten(1, (2, 4))
        torch.testing.assert_close(matches.sum(dim=2), torch.ones(32, 2))
        bits = writes[:, :, -2:].unflatten(1, (2, 4))
        retrieved = (bits * matches.unsqueeze(-1)).sum(dim=2).argmax(dim=-1)
        torch.testing.assert_close(retrieved[:, -1], labels)
        self.assertTrue((retrieved[:, 0] != retrieved[:, 1]).any())

    def test_additive_read_is_invariant_under_swapping_write_rounds(self):
        control = DeltaFiberUpdate(4, 2, mem_dim=8, num_heads=2, erase=False).double()
        inputs, _ = make_batch(torch.Generator().manual_seed(23), 32, rounds=2)
        inputs = inputs.double()
        swapped = torch.cat((inputs[:, 4:8], inputs[:, :4], inputs[:, -1:]), dim=1)
        torch.testing.assert_close(control(inputs)[:, -1], control(swapped)[:, -1])


if __name__ == "__main__":
    unittest.main()
