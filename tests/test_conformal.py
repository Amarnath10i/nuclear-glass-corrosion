"""Unit tests for conformal prediction."""

import numpy as np
import pytest
from nuclear_glass.models.conformal import SplitConformalPredictor


def test_split_conformal_coverage():
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, 1000)
    y_pred = y_true + rng.normal(0, 0.1, 1000)
    cp = SplitConformalPredictor(alpha=0.1)
    cp.calibrate(y_true[:500], y_pred[:500])
    lo, hi = cp.predict(y_pred[500:])
    coverage = np.mean((y_true[500:] >= lo) & (y_true[500:] <= hi))
    assert coverage >= 0.88  # 90% target with 2% slack


def test_predict_before_calibrate_raises():
    with pytest.raises(RuntimeError):
        SplitConformalPredictor(alpha=0.1).predict(np.array([1.0, 2.0]))
