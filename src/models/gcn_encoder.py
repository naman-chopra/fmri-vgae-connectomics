"""
Graph Convolutional Network (GCN) Encoders for Connectomics.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class NativeGCNLayer(nn.Module):
    """
    Pure PyTorch Graph Convolution Layer supporting weighted adjacency matrices.
    Computes: Z = D^{-1/2} A D^{-1/2} X W
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix of shape (N, in_features)
            adj: Normalized adjacency matrix of shape (N, N)
        """
        # Graph propagation: A * X
        support = torch.mm(adj, x)
        # Linear transformation
        output = self.linear(support)
        return output


class GCNEncoder(nn.Module):
    """
    2-Layer Graph Convolutional Encoder for Variational Graph Auto-Encoders.
    Maps node features X and adjacency A to Gaussian latent parameters (mu, logvar).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        # First shared convolutional layer
        self.conv1 = NativeGCNLayer(input_dim, hidden_dim)

        # Dual heads for variational parameterization
        self.conv_mu = NativeGCNLayer(hidden_dim, latent_dim)
        self.conv_logvar = NativeGCNLayer(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (N_rois, input_dim)
            adj: Adjacency matrix (N_rois, N_rois)

        Returns:
            mu: Mean embeddings (N_rois, latent_dim)
            logvar: Log-variance embeddings (N_rois, latent_dim)
        """
        # Hidden layer with ReLU and Dropout
        h = F.relu(self.conv1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Compute latent distribution parameters
        mu = self.conv_mu(h, adj)
        logvar = self.conv_logvar(h, adj)

        # Clip logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar
