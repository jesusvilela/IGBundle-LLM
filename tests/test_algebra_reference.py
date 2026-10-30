import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.algebra_reference import associator, certificate, conjugate, multiply, verify


class AlgebraReferenceTests(unittest.TestCase):
    def test_quaternion_product_against_independent_hamilton_formula(self):
        x, y = (2, -3, 5, 7), (11, 13, -17, 19)
        a, b, c, d = x
        e, f, g, h = y
        expected = (a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g,
                    a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e)
        self.assertEqual(multiply(x, y), expected)

    def test_exact_coefficients_and_polarized_identities(self):
        for n in (1, 2, 4, 8):
            result = certificate(n)
            self.assertEqual(result["norm_composition_residual_terms"], 0)
            self.assertTrue(result["alternativity_polarization"])
            self.assertEqual(result["basis_triples_checked"], n ** 3)
            self.assertEqual(result["associative"], n <= 4)

    def test_octonion_counterexample_is_explicit_and_nonzero(self):
        result = certificate(8)["nonassociativity_counterexample"]
        basis = [tuple(int(i == j) for j in range(8)) for i in range(8)]
        x, y, z = (basis[i] for i in result["basis_indices"])
        self.assertNotEqual(result["left"], result["right"])
        self.assertEqual(list(associator(x, y, z)), result["associator"])
        self.assertEqual(result["basis_indices"], [1, 2, 4])
        self.assertEqual(result["left"], [0] * 7 + [1])
        self.assertEqual(result["right"], [0] * 7 + [-1])

    def test_conjugate_norm_for_integer_vectors(self):
        for n in (1, 2, 4, 8):
            x = tuple(range(-2, n - 2))
            expected = (sum(a*a for a in x),) + (0,) * (n - 1)
            self.assertEqual(multiply(x, conjugate(x)), expected)
            self.assertEqual(multiply(conjugate(x), x), expected)

    def test_invalid_domains_are_explicit(self):
        for x, y in (((), ()), ((1, 2), (1,)), ((1,) * 16, (1,) * 16),
                     ((1.0,), (1,)), ((True,), (1,))):
            with self.assertRaises(ValueError):
                multiply(x, y)

    def test_published_certificate_reproduces_exactly(self):
        report = json.loads((ROOT / "docs" / "benchmark-results" / "algebra-reference.json").read_text())
        self.assertEqual(report, verify())


if __name__ == "__main__":
    unittest.main()
