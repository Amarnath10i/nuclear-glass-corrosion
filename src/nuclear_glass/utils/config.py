"""Configuration management using Pydantic v2 + OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator


class PhysicsConfig(BaseModel):
    """Physical constants and model parameters."""

    temperature_K: float = Field(default=298.15, description="Ambient temperature [K]")
    pH: float = Field(default=7.0, ge=0.0, le=14.0, description="Solution pH")
    activation_energy_kJ_mol: float = Field(default=72.0, gt=0.0)
    pre_exponential_factor: float = Field(default=1.0e-10, gt=0.0)
    affinity_exponent: float = Field(default=1.0, gt=0.0)
    radiation_dose_Gy: float = Field(default=0.0, ge=0.0)


class ModelConfig(BaseModel):
    """ML model hyper-parameters."""

    architecture: str = Field(default="pino", pattern="^(pino|deeponet|bayesian_nn|ensemble)$")
    hidden_dim: int = Field(default=128, gt=0)
    num_layers: int = Field(default=6, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: str = "gelu"
    use_physics_loss: bool = True
    physics_loss_weight: float = Field(default=1.0, gt=0.0)


class TrainingConfig(BaseModel):
    """Training hyper-parameters."""

    epochs: int = Field(default=200, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1.0e-3, gt=0.0)
    weight_decay: float = Field(default=1.0e-5, ge=0.0)
    scheduler: str = "cosine"
    warmup_epochs: int = Field(default=10, ge=0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    seed: int = 42


class DataConfig(BaseModel):
    """Dataset configuration."""

    dataset: str = "srl165"
    data_dir: Path = Path("data/processed")
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    normalise: bool = True
    augment: bool = False

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Path:
        return Path(v).expanduser()


class ExperimentConfig(BaseModel):
    """Top-level experiment config."""

    name: str = "experiment"
    output_dir: Path = Path("outputs")
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    wandb_project: str | None = None
    seed: int = 42


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate a YAML experiment config."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return ExperimentConfig(**raw)


def config_from_omegaconf(cfg: DictConfig) -> ExperimentConfig:
    """Convert a Hydra DictConfig to ExperimentConfig."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    return ExperimentConfig(**raw)  # type: ignore[arg-type]
