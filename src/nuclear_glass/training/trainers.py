"""Custom training pipelines for physics-informed models."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nuclear_glass.utils.logging import get_logger

logger = get_logger(__name__)


class PINOTrainer:
    """Training loop for Physics-Informed Neural Operators."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 200,
        checkpoint_dir: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn     = loss_fn
        self.device      = device
        self.epochs      = epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            x, y = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = [b.to(self.device) for b in batch]
                y_pred = self.model(x)
                total_loss += self.loss_fn(y_pred, y).item()
        return total_loss / len(self.val_loader)

    def train(self) -> dict[str, list[float]]:
        logger.info(f"Starting training for {self.epochs} epochs on {self.device}")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            self.scheduler.step()
            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.checkpoint_dir:
                    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:4d}/{self.epochs} | train={train_loss:.4e} | val={val_loss:.4e} | {time.time()-t0:.1f}s")
        return self.history
