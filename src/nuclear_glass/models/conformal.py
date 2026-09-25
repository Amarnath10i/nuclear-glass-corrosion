"""
Conformal prediction for nuclear waste form performance.

Provides distribution-free prediction intervals with guaranteed coverage.

Reference:
    Angelopoulos, A.N. and Bates, S. (2022). arXiv:2208.02814.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SplitConformalPredictor:
    """Split conformal prediction using residual nonconformity scores.

    Guarantees marginal coverage: P(y in C(x)) >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._q_hat: float | None = None

    def calibrate(self, y_cal: NDArray, y_pred_cal: NDArray) -> SplitConformalPredictor:
        scores = np.abs(y_cal - y_pred_cal)
        n = len(scores)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        self._q_hat = float(np.quantile(scores, level))
        return self

    def predict(self, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        if self._q_hat is None:
            raise RuntimeError("Call .calibrate() before .predict()")
        y = np.asarray(y_pred)
        return y - self._q_hat, y + self._q_hat

    @property
    def interval_width(self) -> float | None:
        return 2 * self._q_hat if self._q_hat is not None else None


class QuantileConformalPredictor:
    """Conformal prediction using quantile regression (asymmetric intervals)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._correction: float | None = None

    def calibrate(self, y_cal: NDArray, q_lo_cal: NDArray, q_hi_cal: NDArray) -> QuantileConformalPredictor:
        scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
        n = len(scores)
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        self._correction = float(np.quantile(scores, level))
        return self

    def predict(self, q_lo: NDArray, q_hi: NDArray) -> tuple[NDArray, NDArray]:
        if self._correction is None:
            raise RuntimeError("Call .calibrate() before .predict()")
        return q_lo - self._correction, q_hi + self._correction
