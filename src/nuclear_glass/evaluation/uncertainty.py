"""Uncertainty quantification evaluation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def decompose_uncertainty(samples: NDArray) -> tuple[float, float, float]:
    """Decompose total predictive uncertainty into epistemic + aleatoric."""
    mean_per_sample = samples.mean(axis=-1)
    epistemic = float(mean_per_sample.var())
    aleatoric = float(samples.var(axis=-1).mean())
    return epistemic + aleatoric, epistemic, aleatoric


def calibration_error(
    y_true: NDArray,
    quantile_preds: dict[float, NDArray],
) -> dict[float, float]:
    """Calibration error per confidence level."""
    errors: dict[float, float] = {}
    for alpha, q_lo in quantile_preds.items():
        coverage = float(np.mean(y_true >= q_lo))
        errors[alpha] = abs((1 - alpha) - coverage)
    return errors


def sharpness_ratio(y_true: NDArray, lower: NDArray, upper: NDArray) -> float:
    data_range = float(y_true.max() - y_true.min()) + 1e-8
    return float((upper - lower).mean()) / data_range


def prediction_interval_plot_data(
    t: NDArray, y_pred: NDArray, lower: NDArray, upper: NDArray, y_true: NDArray | None = None
) -> dict[str, NDArray]:
    out: dict[str, NDArray] = {"t": t, "mean": y_pred, "lower": lower, "upper": upper}
    if y_true is not None:
        out["y_true"] = y_true
    return out
