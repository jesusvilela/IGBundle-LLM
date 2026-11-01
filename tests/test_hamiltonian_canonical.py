"""Bounded canonical Poincare Hamiltonian checks; no trained model is loaded."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from igbundle.dynamics.hamiltonian import LeapfrogIntegrator, VectorField
from igbundle.geometry.poincare import PoincareBall

torch.set_num_threads(1)


class QuadraticPotential(nn.Module):
    def __init__(self, stiffness=2.0):
        super().__init__()
        self.stiffness = nn.Parameter(torch.tensor(float(stiffness)))

    def forward(self, q):
        return 0.5 * self.stiffness * q.square().sum(dim=-1, keepdim=True)


class CanonicalHamiltonianTests(unittest.TestCase):
    def test_hamiltonian_gradients_match_conformal_metric(self):
        c, stiffness = 0.7, 2.5
        q = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([[0.3, 0.15]], dtype=torch.float64, requires_grad=True)
        vf = VectorField(PoincareBall(2, c), QuadraticPotential(stiffness).double())
        grad_q, grad_p = torch.autograd.grad(vf.total_energy(q, p).sum(), (q, p))

        gap = 1 - c * q.square().sum(dim=-1, keepdim=True)
        expected_grad_p = gap.square() * p / 4
        expected_grad_q = stiffness * q - 0.5 * c * gap * q * p.square().sum(dim=-1, keepdim=True)
        torch.testing.assert_close(grad_p, expected_grad_p, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grad_q, expected_grad_q, rtol=1e-12, atol=1e-12)

    def test_exp_map_matches_radial_geodesic_and_origin_derivative(self):
        for c in (0.25, 1.0, 4.0):
            with self.subTest(curvature=c):
                ball = PoincareBall(2, c)
                origin = torch.zeros(2, dtype=torch.float64)
                tangent = torch.tensor([0.3, -0.4], dtype=torch.float64, requires_grad=True)
                mapped = ball.exp_map(origin, tangent)
                expected_norm = torch.tanh(torch.tensor(c**0.5 * 0.5, dtype=torch.float64)) / c**0.5
                torch.testing.assert_close(torch.linalg.vector_norm(mapped), expected_norm,
                                           rtol=1e-12, atol=1e-12)
                mapped.sum().backward()
                self.assertTrue(torch.isfinite(tangent.grad).all())

        ball = PoincareBall(2)
        origin = torch.zeros(2, dtype=torch.float64)
        tangent = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        ball.exp_map(origin, tangent).sum().backward()
        torch.testing.assert_close(tangent.grad, torch.ones_like(tangent))

    def test_exp_map_matches_independent_hyperboloid_flow(self):
        generator = torch.Generator().manual_seed(19)
        for c in (0.25, 1.0, 4.0):
            with self.subTest(curvature=c):
                ball = PoincareBall(3, c)
                x = torch.randn(4, 3, generator=generator, dtype=torch.float64)
                x = 0.3 * x / x.norm(dim=-1, keepdim=True) / c**0.5
                tangent = 0.05 * torch.randn(4, 3, generator=generator, dtype=torch.float64)

                gap = 1 - c * x.square().sum(dim=-1, keepdim=True)
                dot = (x * tangent).sum(dim=-1, keepdim=True)
                ambient_x = torch.cat((
                    (1 + c * x.square().sum(dim=-1, keepdim=True)) / (c**0.5 * gap),
                    2 * x / gap,
                ), dim=-1)
                ambient_tangent = torch.cat((
                    4 * c**0.5 * dot / gap.square(),
                    2 * tangent / gap + 4 * c * x * dot / gap.square(),
                ), dim=-1)
                speed = ball.conformal_factor(x) * tangent.norm(dim=-1, keepdim=True)
                z = c**0.5 * speed
                z_safe = z.clamp_min(torch.finfo(z.dtype).eps)
                ambient_end = torch.cosh(z) * ambient_x + (
                    torch.sinh(z) / z_safe
                ) * ambient_tangent
                expected = ambient_end[..., 1:] / (
                    1 + c**0.5 * ambient_end[..., :1]
                )
                torch.testing.assert_close(ball.exp_map(x, tangent), expected,
                                           rtol=1e-12, atol=1e-12)

    def test_free_geodesic_transports_covector_not_tangent(self):
        ball = PoincareBall(2)
        integrator = LeapfrogIntegrator(VectorField(ball), step_size=0.1, num_steps=1)
        q = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.2, -0.7]], dtype=torch.float64)

        with patch.object(ball, "parallel_transport",
                          side_effect=AssertionError("momentum is not a tangent")):
            q_actual, p_actual = integrator.step(q, p)

        velocity = p / ball.conformal_factor(q).square()
        q_expected = ball.exp_map(q, 0.1 * velocity)
        p_expected = ball.cotangent_transport(q, q_expected, p)
        torch.testing.assert_close(q_actual, q_expected, rtol=0, atol=1e-14)
        torch.testing.assert_close(p_actual, p_expected, rtol=0, atol=1e-14)

    def test_split_preserves_canonical_symplectic_form(self):
        ball = PoincareBall(2, c=0.7)
        potential = QuadraticPotential(1.5).double()
        integrator = LeapfrogIntegrator(
            VectorField(ball, potential), step_size=0.015, num_steps=1,
        )
        omega = torch.zeros((4, 4), dtype=torch.float64)
        omega[:2, 2:] = torch.eye(2, dtype=torch.float64)
        omega[2:, :2] = -torch.eye(2, dtype=torch.float64)

        for coordinates in (
            (0.2, -0.1, 0.3, 0.15),
            (-0.3, 0.15, -0.2, 0.1),
            (0.05, 0.25, 0.1, -0.4),
        ):
            with self.subTest(state=coordinates):
                phase_state = torch.tensor(
                    coordinates, dtype=torch.float64, requires_grad=True,
                )

                def flow(state):
                    q, p = state[:2].reshape(1, 2), state[2:].reshape(1, 2)
                    q_next, p_next = integrator.step(q, p)
                    return torch.cat((q_next.reshape(-1), p_next.reshape(-1)))

                jacobian = torch.autograd.functional.jacobian(flow, phase_state)
                torch.testing.assert_close(
                    jacobian.T @ omega @ jacobian, omega, rtol=1e-10, atol=1e-10,
                )

    def test_split_is_reversible_and_has_bounded_energy_error(self):
        ball = PoincareBall(2)
        vf = VectorField(ball, QuadraticPotential(1.5).double())
        forward = LeapfrogIntegrator(vf, step_size=0.005, num_steps=20)
        reverse = LeapfrogIntegrator(vf, step_size=-0.005, num_steps=20)
        q0 = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
        p0 = torch.tensor([[0.3, 0.15]], dtype=torch.float64)
        energy0 = vf.total_energy(q0, p0)

        q1, p1 = forward(q0, p0)
        q2, p2 = reverse(q1, p1)
        energy_error = (vf.total_energy(q1, p1) - energy0).abs().max().item()
        self.assertLess(energy_error, 2e-6)
        torch.testing.assert_close(q2, q0, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(p2, p0, rtol=1e-10, atol=1e-10)
        self.assertTrue(torch.isfinite(q1).all() and torch.isfinite(p1).all())

    def test_training_gradients_flow_through_split(self):
        potential = QuadraticPotential().double()
        vf = VectorField(PoincareBall(2), potential)
        integrator = LeapfrogIntegrator(vf, step_size=0.01, num_steps=2)
        q = torch.tensor([[0.1, -0.2]], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([[0.3, 0.05]], dtype=torch.float64, requires_grad=True)

        q_out, p_out = integrator(q, p)
        grad_q, grad_p, grad_stiffness = torch.autograd.grad(
            q_out.square().sum() + p_out.square().sum(),
            (q, p, potential.stiffness),
        )
        self.assertTrue(all(torch.isfinite(grad).all()
                            for grad in (grad_q, grad_p, grad_stiffness)))
        self.assertGreater(grad_q.abs().sum().item(), 0)
        self.assertGreater(grad_p.abs().sum().item(), 0)
        self.assertGreater(grad_stiffness.abs().sum().item(), 0)

    def test_no_grad_and_inference_mode_are_supported(self):
        vf = VectorField(PoincareBall(2), QuadraticPotential().double())
        integrator = LeapfrogIntegrator(vf, step_size=0.01, num_steps=1)
        q = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        p = torch.tensor([[0.3, 0.05]], dtype=torch.float64)

        with torch.no_grad():
            no_grad_state = integrator(q, p)
        with torch.inference_mode():
            inference_state = integrator(q, p)
        for state in (no_grad_state, inference_state):
            self.assertTrue(all(torch.isfinite(value).all() for value in state))

    def test_invalid_state_and_settings_fail_closed(self):
        ball = PoincareBall(2)
        vf = VectorField(ball)
        with self.assertRaises(ValueError):
            LeapfrogIntegrator(vf, step_size=float("nan"))
        with self.assertRaises(ValueError):
            LeapfrogIntegrator(vf, num_steps=0)

        integrator = LeapfrogIntegrator(vf)
        with self.assertRaisesRegex(ValueError, "inside"):
            integrator.step(torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2))
        with self.assertRaisesRegex(ValueError, "finite"):
            integrator.step(torch.zeros(1, 2), torch.tensor([[float("inf"), 0.0]]))
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            integrator.step(torch.zeros(1, 2, dtype=torch.bfloat16),
                            torch.zeros(1, 2, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
