"""Standard benchmark suite for nuclear glass corrosion models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nuclear_glass.evaluation.metrics import compute_all_metrics
from nuclear_glass.utils.logging import get_logger

logger = get_logger(__name__)

STANDARD_BENCHMARKS = ["srl165_pct", "isg_static", "vht_200c", "natural_analogue_roman_glass"]


@dataclass
class BenchmarkResult:
    name: str
    dataset: str
    metrics: dict[str, float]
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        return pd.Series({"name": self.name, "dataset": self.dataset, **self.metrics})


def run_benchmark(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    dataset: str,
    data_dir: Path | str = Path("data/benchmarks"),
    alpha: float = 0.10,
) -> BenchmarkResult:
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{dataset}.csv"

    if not csv_path.exists():
        logger.warning(f"Benchmark data not found: {csv_path} - generating synthetic reference")
        from nuclear_glass.data.loaders import synthetic_srl165
        ds = synthetic_srl165(n_samples=300)
        df = ds.df
    else:
        df = pd.read_csv(csv_path)

    X = df[["time_days", "temperature_C", "pH"]].values.astype(np.float32)
    y_true = df["BNL_g_m2"].values.astype(np.float32)
    y_pred = predict_fn(X)
    metrics = compute_all_metrics(y_pred, y_true)
    logger.info(f"[{dataset}] Metrics: {metrics}")
    return BenchmarkResult(name="model", dataset=dataset, metrics=metrics)


def compare_benchmarks(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_series() for r in results]).set_index("name")
