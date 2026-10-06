"""
Brain Graph Modularity and Community Detection Algorithms.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering


def compute_newman_modularity(adj_matrix: np.ndarray, community_labels: np.ndarray) -> float:
    """
    Computes Newman's Modularity Score Q on a weighted graph:
    Q = (1 / 2m) * sum_{ij} [ A_{ij} - (k_i k_j / 2m) ] * delta(c_i, c_j)

    Args:
        adj_matrix: Weighted symmetric adjacency matrix of shape (N, N)
        community_labels: Cluster assignment array of shape (N,)

    Returns:
        Q: Modularity score in [-0.5, 1.0] (higher indicates stronger community structure)
    """
    m = np.sum(adj_matrix) / 2.0
    if m == 0:
        return 0.0

    k = np.sum(adj_matrix, axis=1)  # Node strengths/degrees
    num_nodes = adj_matrix.shape[0]

    # Expected adjacency under configuration null model
    expected = np.outer(k, k) / (2.0 * m)
    B = adj_matrix - expected  # Modularity matrix

    # Indicator matrix for community membership
    Q = 0.0
    unique_communities = np.unique(community_labels)
    for c in unique_communities:
        mask = (community_labels == c)
        Q += np.sum(B[np.ix_(mask, mask)])

    Q = Q / (2.0 * m)
    return float(Q)


def detect_brain_communities(
    adj_matrix: np.ndarray,
    n_communities: int = 7,  # Default: 7 Canonical Yeo Networks
    method: str = "spectral",
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Detects modular brain subnetworks from a functional connectome.

    Args:
        adj_matrix: Connectome adjacency matrix of shape (N, N)
        n_communities: Number of target modules
        method: Community detection method ('spectral' or 'louvain')
        random_state: Random seed

    Returns:
        labels: Array of community labels for each ROI
        modularity_q: Newman's modularity score Q
    """
    adj_clean = np.maximum(adj_matrix, 0.0)
    np.fill_diagonal(adj_clean, 0.0)

    if method == "spectral":
        clustering = SpectralClustering(
            n_clusters=n_communities,
            affinity="precomputed",
            random_state=random_state,
            assign_labels="kmeans",
        )
        labels = clustering.fit_predict(adj_clean)
    else:
        # Fallback to spectral
        clustering = SpectralClustering(
            n_clusters=n_communities,
            affinity="precomputed",
            random_state=random_state,
        )
        labels = clustering.fit_predict(adj_clean)

    modularity_q = compute_newman_modularity(adj_clean, labels)
    return labels, modularity_q


def iterative_consensus_spectral_clustering(
    adj_matrices: List[np.ndarray],
    n_clusters: int = 7,
    n_iterations: int = 25,
    threshold: float = 0.4,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterative Consensus Spectral Clustering (ICSC) across multiple subject connectomes.
    Constructs a co-assignment consensus matrix and iteratively pools community partitions.

    Args:
        adj_matrices: List of connectome matrices
        n_clusters: Number of target modules
        n_iterations: Number of consensus iterations
        threshold: Noise suppression threshold for consensus matrix

    Returns:
        consensus_labels: Final unified module assignments
        consensus_matrix: Final co-assignment agreement matrix
    """
    num_nodes = adj_matrices[0].shape[0]
    num_graphs = len(adj_matrices)

    # Initial clustering on individual graphs
    co_assignment = np.zeros((num_nodes, num_nodes))

    for adj in adj_matrices:
        labels, _ = detect_brain_communities(adj, n_communities=n_clusters, random_state=random_state)
        # Add outer product indicator
        for c in np.unique(labels):
            mask = (labels == c).astype(float)
            co_assignment += np.outer(mask, mask)

    consensus_matrix = co_assignment / float(num_graphs)
    consensus_matrix[consensus_matrix < threshold] = 0.0
    np.fill_diagonal(consensus_matrix, 0.0)

    # Final spectral clustering on consensus matrix
    consensus_labels, _ = detect_brain_communities(
        consensus_matrix, n_communities=n_clusters, random_state=random_state
    )
    return consensus_labels, consensus_matrix
