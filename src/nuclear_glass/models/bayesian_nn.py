"""
Bayesian Neural Network with variational inference.

Uses reparameterisation trick for weight uncertainty (aleatoric + epistemic).

Reference:
    Blundell, C. et al. (2015). Weight uncertainty in neural networks.
    ICML 2015.

Changelog:
    - 2026-07-30: Fixed KL divergence closure bug in BayesianLinear
      (inner `kl` function now takes explicit w/b arguments).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIGMA_INIT = -3.0
PRIOR_SIGMA1 = 1.0
PRIOR_SIGMA2 = 0.001
PRIOR_PI     = 0.5

def log_gaussian(x: torch.Tensor, mu: float | torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    return -0.5 * math.log(2 * math.pi) - torch.log(torch.as_tensor(sigma, dtype=torch.float32)) - (x - mu) ** 2 / (2 * sigma ** 2)

def _kl_divergence(param: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL divergence from Gaussian posterior to scale-mixture prior."""
    log_post  = log_gaussian(param, 0.0, sigma).sum()
    log_prior = torch.log(
        PRIOR_PI * torch.exp(log_gaussian(param, 0.0, PRIOR_SIGMA1))
        + (1.0 - PRIOR_PI) * torch.exp(log_gaussian(param, 0.0, PRIOR_SIGMA2))
    ).sum()
    return log_post - log_prior

class BayesianLinear(nn.Module):
    """Linear layer with Gaussian weight posterior."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho   = nn.Parameter(torch.full((out_features, in_features), LOG_SIGMA_INIT))
        self.bias_mu      = nn.Parameter(torch.zeros(out_features))
        self.bias_rho     = nn.Parameter(torch.full((out_features,), LOG_SIGMA_INIT))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (output, kl_divergence)."""
        w_sigma = F.softplus(self.weight_rho)
        b_sigma = F.softplus(self.bias_rho)
        w = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)
        b = self.bias_mu   + b_sigma * torch.randn_like(self.bias_mu)

        out   = F.linear(x, w, b)
        kl_w  = _kl_divergence(w, w_sigma)
        kl_b  = _kl_divergence(b, b_sigma)
        return out, kl_w + kl_b

class BayesianMLP(nn.Module):
    """Variational Bayesian MLP for corrosion rate prediction."""

    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 1,
        hidden_dim: int = 128,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        dims = [in_features] + [hidden_dim] * (n_layers - 1) + [out_features]
        self.layers = nn.ModuleList(
            [BayesianLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_kl = torch.tensor(0.0, device=x.device)
        for i, layer in enumerate(self.layers):
            x, kl = layer(x)
            total_kl = total_kl + kl
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return x, total_kl

    def elbo_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_samples: int = 10,
        dataset_size: int = 1000,
    ) -> torch.Tensor:
        """Monte Carlo ELBO: E[log p(y|w)] - KL[q(w)||p(w)]."""
        log_likelihoods = []
        kls = []
        for _ in range(n_samples):
            y_pred, kl = self(x)
            ll = -F.mse_loss(y_pred.squeeze(), y)
            log_likelihoods.append(ll)
            kls.append(kl)
        return -(torch.stack(log_likelihoods).mean() - torch.stack(kls).mean() / dataset_size)
