"""Domain-specific visualisation utilities for nuclear glass corrosion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "primary":   "#2C7BB6",
    "secondary": "#D7191C",
    "accent":    "#1A9641",
    "neutral":   "#636363",
    "highlight": "#FDAE61",
}


def plot_bnl_vs_time(
    time_days: np.ndarray, bnl_g_m2: np.ndarray, label: str = "Predicted",
    bnl_true: Optional[np.ndarray] = None, log_scale: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_days, bnl_g_m2, color=PALETTE["primary"], lw=2, label=label)
    if bnl_true is not None:
        ax.scatter(time_days, bnl_true, c=PALETTE["secondary"], s=20, zorder=5, label="Observed")
    if log_scale:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("BNL [g/m2]")
    ax.set_title("Boron Normalised Loss vs. Time", fontweight="bold")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_prediction_interval(
    time_days: np.ndarray, mean_pred: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    y_true: Optional[np.ndarray] = None, alpha: float = 0.10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(time_days, lower, upper, alpha=0.25, color=PALETTE["primary"], label=f"{int((1-alpha)*100)}% PI")
    ax.plot(time_days, mean_pred, color=PALETTE["primary"], lw=2, label="Mean prediction")
    if y_true is not None:
        ax.scatter(time_days, y_true, c=PALETTE["secondary"], s=18, zorder=6, label="Observed")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("BNL [g/m2]")
    ax.set_title("Corrosion Prediction with Uncertainty Bounds", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_arrhenius(
    temperatures_K: np.ndarray, rates_m_s: np.ndarray, save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    inv_T = 1000.0 / temperatures_K
    ax.scatter(inv_T, np.log10(rates_m_s), c=PALETTE["accent"], s=40, zorder=5)
    coeffs = np.polyfit(inv_T, np.log10(rates_m_s), 1)
    inv_T_fit = np.linspace(inv_T.min(), inv_T.max(), 100)
    ax.plot(inv_T_fit, np.poly1d(coeffs)(inv_T_fit), "--", color=PALETTE["neutral"], lw=1.5)
    Ea_kJ = -coeffs[0] * 2.303 * 8.314 / 1e3
    ax.set_xlabel("1000/T [K-1]"); ax.set_ylabel("log10(rate [m/s])")
    ax.set_title(f"Arrhenius Plot  -  Ea approx {Ea_kJ:.1f} kJ/mol", fontweight="bold")
    ax.grid(True, alpha=0.3); plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
