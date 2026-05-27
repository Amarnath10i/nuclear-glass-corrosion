"""Training callbacks for logging, early stopping, and W&B integration."""

from __future__ import annotations

from typing import Any

from nuclear_glass.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._counter  = 0
        self._best     = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best   = val_loss
            self._counter = 0
        else:
            self._counter += 1
        if self._counter >= self.patience:
            logger.info(f"Early stopping triggered after {self.patience} stagnant epochs.")
            return True
        return False


class WandbLogger:
    """Wrapper around Weights and Biases logging."""

    def __init__(self, project: str, name: str, config: dict[str, Any] | None = None) -> None:
        try:
            import wandb
            self._run = wandb.init(project=project, name=name, config=config)
            self.enabled = True
        except ImportError:
            logger.warning("wandb not installed - logging disabled.")
            self.enabled = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.enabled:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled:
            import wandb
            wandb.finish()


class MetricTracker:
    """Accumulate and summarise training metrics."""

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self._data.setdefault(k, []).append(v)

    def mean(self, key: str) -> float:
        vals = self._data.get(key, [])
        return sum(vals) / len(vals) if vals else float("nan")

    def last(self, key: str) -> float:
        vals = self._data.get(key, [])
        return vals[-1] if vals else float("nan")

    def reset(self) -> None:
        self._data.clear()
