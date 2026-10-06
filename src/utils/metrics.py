"""
Evaluation Metrics for Connectome Denoising and Modularity Stability.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_connectome_reconstruction(
    predicted_adj: np.ndarray,
    target_adj: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates reconstruction quality between predicted denoised connectome and ground truth clean target.

    Args:
        predicted_adj: Reconstructed adjacency matrix (N, N)
        target_adj: Target ground-truth adjacency matrix (N, N)

    Returns:
        metrics: Dictionary containing MSE, MAE, Pearson r correlation, and Relative Frobenius Error
    """
    # Exclude diagonal
    n = predicted_adj.shape[0]
    mask = ~np.eye(n, dtype=bool)

    pred_flat = predicted_adj[mask]
    target_flat = target_adj[mask]

    mse = float(np.mean((pred_flat - target_flat) ** 2))
    mae = float(np.mean(np.abs(pred_flat - target_flat)))

    # Pearson correlation across edge weights
    if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
        pearson_r = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    else:
        pearson_r = 0.0

    # Relative Frobenius error: ||A_pred - A_target||_F / ||A_target||_F
    frob_target = np.linalg.norm(target_adj, "fro")
    if frob_target > 0:
        rel_frob = float(np.linalg.norm(predicted_adj - target_adj, "fro") / frob_target)
    else:
        rel_frob = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "pearson_r": pearson_r,
        "relative_frobenius_error": rel_frob,
    }


def evaluate_link_prediction_auc(
    predicted_adj: np.ndarray,
    target_adj: np.ndarray,
    threshold: float = 0.2,
) -> Tuple[float, float]:
    """
    Evaluates edge link prediction capability via AUC-ROC and Average Precision (AP).

    Args:
        predicted_adj: Continuous predicted edge weights (N, N)
        target_adj: Continuous or binary target adjacency (N, N)
        threshold: Binarization cutoff for positive edge existence

    Returns:
        auc: ROC-AUC score
        ap: Average Precision score
    """
    n = predicted_adj.shape[0]
    mask = ~np.eye(n, dtype=bool)

    y_score = predicted_adj[mask]
    y_true = (target_adj[mask] > threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        return 1.0, 1.0

    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    return auc, ap
