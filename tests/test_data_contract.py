import numpy as np
import torch

from src.data.preprocessing import compute_functional_connectivity, standardize_bold_signals
from src.models.vgae import VariationalGraphAutoEncoder


def test_bold_standardization_preserves_roi_time_orientation():
    bold = np.arange(24, dtype=float).reshape(4, 6)
    standardized = standardize_bold_signals(bold)
    assert standardized.shape == (4, 6)
    np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(standardized.std(axis=1), 1.0, atol=1e-7)


def test_connectivity_is_symmetric_and_has_zero_diagonal():
    rng = np.random.default_rng(42)
    bold = standardize_bold_signals(rng.normal(size=(8, 20)))
    adjacency = compute_functional_connectivity(bold)
    assert adjacency.shape == (8, 8)
    np.testing.assert_allclose(adjacency, adjacency.T, atol=1e-7)
    np.testing.assert_allclose(np.diag(adjacency), 0.0)
    assert np.all(adjacency >= 0.0)


def test_vgae_forward_returns_expected_shapes():
    model = VariationalGraphAutoEncoder(input_dim=12, hidden_dim=8, latent_dim=4)
    x = torch.randn(6, 12)
    adjacency = torch.rand(6, 6)
    adjacency.fill_diagonal_(0.0)
    reconstructed, latent, mean, logvar = model(x, adjacency)
    assert reconstructed.shape == (6, 6)
    assert latent.shape == (6, 4)
    assert mean.shape == (6, 4)
    assert logvar.shape == (6, 4)
    assert torch.allclose(torch.diag(reconstructed), torch.zeros(6))
