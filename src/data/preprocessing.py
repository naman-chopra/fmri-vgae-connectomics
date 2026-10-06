"""
Preprocessing and Graph Construction for BOLD fMRI Time-Series.
"""

from typing import Tuple, Optional
import numpy as np
import torch


def standardize_bold_signals(timeseries: np.ndarray) -> np.ndarray:
    """
    Standardize BOLD time-series per ROI to have zero mean and unit variance.

    Args:
        timeseries: Array of shape (N_rois, T_timepoints) or (T_timepoints, N_rois)

    Returns:
        Standardized array of shape (N_rois, T_timepoints)
    """
    if timeseries.shape[0] > timeseries.shape[1]:
        # Transpose if timepoints is first dimension
        timeseries = timeseries.T

    mean = np.mean(timeseries, axis=1, keepdims=True)
    std = np.std(timeseries, axis=1, keepdims=True)
    std[std == 0] = 1.0  # Prevent division by zero
    return (timeseries - mean) / std


def compute_functional_connectivity(
    bold_matrix: np.ndarray,
    positive_only: bool = True,
    zero_diagonal: bool = True,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Compute pairwise Pearson correlation matrix from standardized BOLD signals.

    Args:
        bold_matrix: Standardized BOLD signals of shape (N_rois, T_timepoints)
        positive_only: If True, clamp negative correlation coefficients to 0
        zero_diagonal: If True, set self-connections (diagonal) to 0
        threshold: Optional correlation threshold below which edges are zeroed

    Returns:
        Functional connectivity adjacency matrix of shape (N_rois, N_rois)
    """
    # Pearson correlation matrix
    corr_matrix = np.corrcoef(bold_matrix)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    if positive_only:
        corr_matrix = np.maximum(corr_matrix, 0.0)

    if threshold is not None:
        corr_matrix[corr_matrix < threshold] = 0.0

    if zero_diagonal:
        np.fill_diagonal(corr_matrix, 0.0)

    return corr_matrix


def matrix_to_edge_index_and_weight(
    adj_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dense adjacency matrix to PyTorch Geometric edge_index and edge_weight.

    Args:
        adj_matrix: Dense tensor of shape (N, N)

    Returns:
        edge_index: Tensor of shape (2, E)
        edge_weight: Tensor of shape (E,)
    """
    edge_index = torch.stack(torch.where(adj_matrix > 0))
    edge_weight = adj_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_weight
