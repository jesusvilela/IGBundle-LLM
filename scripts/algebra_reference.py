"""Exact Cayley-Dickson identity certificates, not a model/runtime integration."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path


DIMENSIONS = (1, 2, 4, 8)


def conjugate(x):
    return (x[0],) + tuple(-v for v in x[1:])


def _add(x, y):
    return tuple(a + b for a, b in zip(x, y))


def _subtract(x, y):
    return tuple(a - b for a, b in zip(x, y))


def _multiply(x, y):
    if len(x) == 1:
        return (x[0] * y[0],)
    half = len(x) // 2
    a, b = x[:half], x[half:]
    c, d = y[:half], y[half:]
    return (_subtract(_multiply(a, c), _multiply(conjugate(d), b))
            + _add(_multiply(d, a), _multiply(b, conjugate(c))))


def multiply(x, y):
    """Integer coefficients in dimensions 1, 2, 4, 8; no implicit broadcasting."""
    x, y = tuple(x), tuple(y)
    if len(x) not in DIMENSIONS or len(x) != len(y):
        raise ValueError("Equal dimensions in {1, 2, 4, 8} are required")
    if any(type(value) is not int for value in x + y):
        raise ValueError("Exact integer coefficients are required")
    return _multiply(x, y)


def associator(x, y, z):
    return _subtract(multiply(multiply(x, y), z), multiply(x, multiply(y, z)))


def _norm_coefficients(basis):
    """Residual coefficients of N(x*y) - N(x)*N(y), not sampled inputs."""
    n = len(basis)
    products = {(i, j): multiply(basis[i], basis[j]) for i, j in product(range(n), repeat=2)}
    residual = defaultdict(int)
    for i, j, k, ell in product(range(n), repeat=4):
        key = (*sorted((i, k)), *sorted((j, ell)))
        residual[key] += sum(a * b for a, b in zip(products[i, j], products[k, ell]))
    for i, j in product(range(n), repeat=2):
        residual[i, i, j, j] -= 1
    return {key: value for key, value in residual.items() if value}


def certificate(n):
    if n not in DIMENSIONS:
        raise ValueError("Reference dimension must be 1, 2, 4 or 8")
    basis = [tuple(int(i == j) for j in range(n)) for i in range(n)]
    zero = (0,) * n
    counterexample = None
    for x in basis:
        if multiply(x, basis[0]) != x or multiply(basis[0], x) != x:
            raise ValueError("Unit identity failed")
        for y in basis:
            if conjugate(multiply(x, y)) != multiply(conjugate(y), conjugate(x)):
                raise ValueError("Conjugation identity failed")
    for i, j, k in product(range(n), repeat=3):
        x, y, z = basis[i], basis[j], basis[k]
        a = associator(x, y, z)
        if a != zero and counterexample is None:
            counterexample = {
                "basis_indices": [i, j, k],
                "left": list(multiply(multiply(x, y), z)),
                "right": list(multiply(x, multiply(y, z))),
                "associator": list(a),
            }
        # Trilinearity extends antisymmetry on a basis to all real coefficients.
        if (_add(a, associator(y, x, z)) != zero
                or _add(a, associator(x, z, y)) != zero):
            raise ValueError("Alternativity polarization failed")
    if _norm_coefficients(basis):
        raise ValueError("Norm-composition polynomial has a nonzero coefficient")
    if (n <= 4 and counterexample is not None) or (n == 8 and counterexample is None):
        raise ValueError("Unexpected associativity result")
    return {
        "dimension": n,
        "basis_triples_checked": n ** 3,
        "conjugation_reverses_product": True,
        "alternativity_polarization": True,
        "norm_composition_residual_terms": 0,
        "associative": counterexample is None,
        "nonassociativity_counterexample": counterexample,
    }


def verify():
    return {
        "protocol": "exact-cayley-dickson-reference-v1",
        "convention": "(a,b)(c,d)=(ac-conj(d)b, da+b*conj(c)); conjugate reverses imaginary signs",
        "arithmetic": "Python integers, no floating point",
        "certificates": [certificate(n) for n in DIMENSIONS],
        "extension": "Bilinear product and linear conjugation; coefficient identities extend over real scalars.",
        "limits": [
            "Independent mathematical reference, not an IGBundle implementation.",
            "No checkpoint, activation, learned hyperbolicity or performance evidence.",
            "No quaternionic/octonionic hyperbolic metric or GGUF runtime is supplied.",
            "Dimensions above eight are deliberately unsupported.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(verify(), indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    print(rendered)
