"""Descriptive telemetry only; no curvature or performance estimator."""

import numpy as np


class DemoTelemetry:
    """Aggregate observed states without retaining generation-sized arrays.

    Every observed (layer, forward-call, batch, token, component) categorical
    row has equal weight. Repeated context during decoding is not deduplicated.
    """

    SAMPLE_LIMIT = 4096

    def __init__(self):
        self.reset()

    def reset(self):
        self.sigma_count = 0
        self.sigma_sum = 0.0
        self.sigma_in_interval = 0
        self.sigma_samples = np.empty(0, dtype=np.float64)
        self.probability_sum = None
        self.probability_count = 0
        self.entropy_sum = 0.0
        self.entropy_count = 0

    def add(self, sigma, probabilities):
        sigma = np.asarray(sigma, dtype=np.float64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if (sigma.ndim != 4 or probabilities.ndim != 4
                or sigma.shape[:3] != probabilities.shape[:3]
                or sigma.size == 0 or probabilities.size == 0):
            raise ValueError("Expected nonempty sigma[B,T,P,D] and probabilities[B,T,P,K]")
        if not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("Sigma must contain finite positive standard deviations")
        if (not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(-1), 1.0, rtol=0, atol=0.005)):
            raise ValueError("Expected finite categorical probabilities summing to one")
        if (self.probability_sum is not None
                and probabilities.shape[2:] != self.probability_sum.shape):
            raise ValueError("Categorical component/category shape changed during capture")

        # Normalize small low-precision softmax rounding error, not arbitrary scores.
        p = probabilities / probabilities.sum(-1, keepdims=True)
        logs = np.zeros_like(p)
        np.log(p, out=logs, where=p > 0)
        entropy = -(p * logs).sum(-1)
        matrix_sum = p.sum(axis=(0, 1))

        with np.errstate(over="raise", invalid="raise"):
            sigma_sum = self.sigma_sum + float(sigma.sum())
        if not np.isfinite(sigma_sum):
            raise ValueError("Sigma accumulator overflow")
        self.sigma_count += sigma.size
        self.sigma_sum = sigma_sum
        self.sigma_in_interval += int(((sigma > 0.5) & (sigma < 2.0)).sum())
        remaining = self.SAMPLE_LIMIT - self.sigma_samples.size
        if remaining:
            self.sigma_samples = np.concatenate((self.sigma_samples, sigma.ravel()[:remaining]))
        if self.probability_sum is None:
            self.probability_sum = matrix_sum
        else:
            self.probability_sum += matrix_sum
        self.probability_count += p.shape[0] * p.shape[1]
        self.entropy_sum += float(entropy.sum())
        self.entropy_count += entropy.size

    def summary(self):
        if not self.sigma_count:
            return None
        return {
            "mean_sigma": self.sigma_sum / self.sigma_count,
            "sigma_interval_percent": 100 * self.sigma_in_interval / self.sigma_count,
            "mean_categorical_entropy": self.entropy_sum / self.entropy_count,
            "sigma_count": self.sigma_count,
            "categorical_rows": self.entropy_count,
        }

    def mean_probabilities(self):
        if not self.probability_count:
            return None
        return self.probability_sum / self.probability_count


def format_metrics(summary):
    if summary is None:
        return (
            "### Telemetry unavailable\n"
            "No default-adapter states were captured. "
            "This is missing evidence, not a zero score."
        )
    return (
        "### Descriptive adapter telemetry\n\n"
        f"- **Mean Gaussian standard deviation (sigma):** {summary['mean_sigma']:.4f}\n"
        f"- **Sigma in the arbitrary interval (0.5, 2.0):** "
        f"{summary['sigma_interval_percent']:.1f}%\n"
        f"- **Mean categorical entropy (nats):** {summary['mean_categorical_entropy']:.4f}\n"
        f"- **Observed sigma values / categorical rows:** "
        f"{summary['sigma_count']} / {summary['categorical_rows']}\n\n"
        "These are not curvature, geometric faithfulness, or task accuracy. "
        "The interval is a descriptive counter, not a target or quality threshold. "
        "Observations are weighted across layers and forward calls; repeated context "
        "may be counted again during decoding."
    )
