"""
Loss functions for Variational Graph Auto-Encoders on Connectomics.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def vgae_loss(
    reconstructed_adj: torch.Tensor,
    target_adj: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    pos_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes joint Reconstruction Loss (MSE/BCE) + KL Divergence for VGAE.

    Args:
        reconstructed_adj: Reconstructed adjacency tensor (N, N)
        target_adj: Clean target adjacency tensor (N, N)
        mu: Mean tensor of latent distribution (N, latent_dim)
        logvar: Log-variance tensor of latent distribution (N, latent_dim)
        kl_weight: Beta hyperparameter scaling the KL divergence regularizer
        pos_weight: Weight for positive edges

    Returns:
        total_loss: Combined scalar loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # MSE Reconstruction Loss on non-diagonal edge weights
    recon_loss = F.mse_loss(reconstructed_adj, target_adj, reduction="mean")

    # Analytical KL-Divergence with standard Normal prior N(0, I)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + (kl_weight * kl_loss)
    return total_loss, recon_loss, kl_loss
