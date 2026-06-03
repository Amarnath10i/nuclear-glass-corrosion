"""Domain-specific evaluation metrics for glass corrosion predictions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def relative_l2_error(y_pred: NDArray, y_true: NDArray) -> float:
    return float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + 1e-10))


def normalised_mass_loss_error(bnl_pred: NDArray, bnl_true: NDArray) -> float:
    """Mean absolute percentage error on Boron Normalised Loss [%]."""
    mask = np.abs(bnl_true) > 1e-8
    return float(np.mean(np.abs((bnl_pred[mask] - bnl_true[mask]) / bnl_true[mask])) * 100.0)


def peak_corrosion_rate_error(rate_pred: NDArray, rate_true: NDArray) -> float:
    return float(abs(rate_pred.max() - rate_true.max()) / (abs(rate_true.max()) + 1e-10) * 100.0)


def long_term_extrapolation_bias(bnl_pred: NDArray, bnl_true: NDArray, log_scale: bool = True) -> float:
    """Bias in long-term BNL predictions (log scale for 100 kyr accuracy)."""
    if log_scale:
        eps = 1e-10
        return float(np.mean(np.log10(bnl_pred + eps) - np.log10(bnl_true + eps)))
    return float(np.mean(bnl_pred - bnl_true))


def coverage_probability(y_true: NDArray, lower: NDArray, upper: NDArray) -> float:
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def interval_sharpness(lower: NDArray, upper: NDArray) -> float:
    return float((upper - lower).mean())


def compute_all_metrics(
    y_pred: NDArray,
    y_true: NDArray,
    lower: NDArray | None = None,
    upper: NDArray | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "relative_l2_error": relative_l2_error(y_pred, y_true),
        "bnl_mape_percent": normalised_mass_loss_error(y_pred, y_true),
        "peak_rate_error_percent": peak_corrosion_rate_error(y_pred, y_true),
        "long_term_log_bias": long_term_extrapolation_bias(y_pred, y_true),
    }
    if lower is not None and upper is not None:
        metrics["coverage_probability"] = coverage_probability(y_true, lower, upper)
        metrics["interval_sharpness"]   = interval_sharpness(lower, upper)
    return metrics
