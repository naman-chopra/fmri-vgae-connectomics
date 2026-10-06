"""
Graph Laplacian Decomposition and KNEED Elbow Detection for Inter-Subject Subgrouping.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False


def compute_normalized_laplacian(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Computes symmetric normalized Graph Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}

    Args:
        adj_matrix: Adjacency matrix of shape (N, N)

    Returns:
        Normalized Laplacian matrix of shape (N, N)
    """
    degree = np.sum(adj_matrix, axis=1)
    d_inv_sqrt = np.zeros_like(degree, dtype=float)
    np.power(degree, -0.5, out=d_inv_sqrt, where=degree > 0)

    D_mat = np.diag(d_inv_sqrt)
    L_norm = np.eye(adj_matrix.shape[0]) - D_mat @ adj_matrix @ D_mat
    return L_norm


def eigen_decomposition_kneed(
    adj_matrix: np.ndarray,
    max_k: int = 10,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Decomposes normalized Laplacian and determines optimal number of clusters k via KNEED.

    Args:
        adj_matrix: Subject similarity matrix of shape (num_subjects, num_subjects)
        max_k: Maximum number of clusters to evaluate

    Returns:
        optimal_k: Detected optimal number of clusters
        eigenvalues: Sorted eigenvalues of Laplacian
        eigenvectors: Corresponding eigenvectors
    """
    L_norm = compute_normalized_laplacian(adj_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Sort eigenvalues in ascending order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Evaluate knee/elbow on the first max_k eigenvalues
    x = np.arange(1, min(max_k + 1, len(eigenvalues) + 1))
    y = eigenvalues[: len(x)]

    if HAS_KNEED:
        kneedle = KneeLocator(x, y, curve="concave", direction="increasing")
        optimal_k = kneedle.elbow if kneedle.elbow is not None else 4
    else:
        # Fallback heuristic: find maximum eigengap
        eigengaps = np.diff(y)
        optimal_k = int(np.argmax(eigengaps) + 1)

    return optimal_k, eigenvalues, eigenvectors


def cluster_subjects(
    similarity_matrix: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Clusters subjects into subgroups using Spectral Clustering / KMeans on Laplacian eigenvectors.

    Args:
        similarity_matrix: Pairwise subject distance/similarity of shape (S, S)
        n_clusters: Target number of subject cohorts (default k=4)
        random_state: Random seed

    Returns:
        labels: Cluster assignment array of shape (S,)
        metrics: Dictionary containing silhouette, Calinski-Harabasz, and Davies-Bouldin scores
    """
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    similarity_matrix = np.maximum(similarity_matrix, 0.0)
    np.fill_diagonal(similarity_matrix, 0.0)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    labels = clustering.fit_predict(similarity_matrix)

    # Convert affinity to a zero-diagonal distance matrix for validation.
    distance_matrix = 1.0 - np.clip(similarity_matrix, 0.0, 1.0)
    np.fill_diagonal(distance_matrix, 0.0)
    sil = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
    cal = float(calinski_harabasz_score(distance_matrix, labels))
    dav = float(davies_bouldin_score(distance_matrix, labels))

    metrics = {
        "silhouette_score": sil,
        "calinski_harabasz": cal,
        "davies_bouldin": dav,
    }
    return labels, metrics
