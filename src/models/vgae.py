"""
Variational Graph Auto-Encoder (VGAE) for Connectome Denoising.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gcn_encoder import GCNEncoder


class InnerProductDecoder(nn.Module):
    """
    Inner-product decoder reconstructing adjacency matrix:
    A_hat = sigmoid(Z * Z^T) or A_hat = ReLU(Z * Z^T)
    """

    def __init__(self, activation: str = "sigmoid"):
        super().__init__()
        self.activation = activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent node embeddings of shape (N, latent_dim)

        Returns:
            reconstructed_adj: Symmetric adjacency tensor of shape (N, N)
        """
        adj_logits = torch.mm(z, z.t())

        if self.activation == "sigmoid":
            adj_hat = torch.sigmoid(adj_logits)
        elif self.activation == "relu":
            adj_hat = F.relu(adj_logits)
        elif self.activation == "none":
            adj_hat = adj_logits
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Set diagonal to zero (no self-loops in standard functional connectomes)
        adj_hat = adj_hat - torch.diag(torch.diag(adj_hat))
        return adj_hat


class VariationalGraphAutoEncoder(nn.Module):
    """
    Full Variational Graph Auto-Encoder for fMRI Connectomics.
    Integrates GCN encoder, latent reparameterization, and inner-product decoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
        decoder_activation: str = "sigmoid",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = InnerProductDecoder(activation=decoder_activation)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * eps, where eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            # Deterministic inference: return mean representation
            return mu

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, adj)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (N_rois, input_dim)
            adj: Noisy input adjacency (N_rois, N_rois)

        Returns:
            adj_hat: Denoised reconstructed adjacency (N_rois, N_rois)
            z: Sampled latent embedding (N_rois, latent_dim)
            mu: Mean latent parameters (N_rois, latent_dim)
            logvar: Log-variance latent parameters (N_rois, latent_dim)
        """
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_hat = self.decode(z)
        return adj_hat, z, mu, logvar
