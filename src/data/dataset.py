"""
Dataset Loaders and Synthetic Connectome Generator for fMRI Graphs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    standardize_bold_signals,
    compute_functional_connectivity,
    matrix_to_edge_index_and_weight,
)
from src.data.hcp_archive import (
    load_matlab_correlation,
    load_matlab_timeseries,
    validated_matlab_subjects,
)


class HCPMatlabConnectomeDataset(Dataset):
    """Lazy dataset for complete subjects in ``HCP_rs_power.zip``.

    The archive is treated as the source of truth.  A subject is admitted only
    when all four time-series and all four correlation files are present.  Data
    is loaded on demand, so the 6 GB archive is never extracted wholesale.
    """

    def __init__(
        self,
        archive_path: str | Path,
        subjects: Optional[List[str]] = None,
        positive_only: bool = False,
    ):
        self.archive_path = Path(archive_path)
        complete, self.rejected_subjects = validated_matlab_subjects(self.archive_path)
        complete = set(complete)
        self.subjects = sorted(complete if subjects is None else complete & set(subjects))
        self.positive_only = positive_only

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        subject = self.subjects[index]
        runs = [load_matlab_timeseries(self.archive_path, subject, run) for run in range(1, 5)]
        correlations = [load_matlab_correlation(self.archive_path, subject, run) for run in range(1, 5)]
        concatenated = np.concatenate(runs, axis=1)
        features = standardize_bold_signals(concatenated)
        averaged = np.mean(correlations, axis=0)
        averaged = (averaged + averaged.T) / 2.0
        if self.positive_only:
            averaged = np.maximum(averaged, 0.0)
        np.fill_diagonal(averaged, 0.0)
        adjacency = torch.from_numpy(averaged.astype(np.float32))
        edge_index, edge_weight = matrix_to_edge_index_and_weight(adjacency)
        return {
            "subject": subject,
            "x": torch.from_numpy(features.astype(np.float32)),
            "run_timeseries": torch.from_numpy(np.stack(runs).astype(np.float32)),
            "run_adjs": torch.from_numpy(np.stack(correlations).astype(np.float32)),
            "adj": adjacency,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }


class ConnectomeGraphDataset(Dataset):
    """
    PyTorch Dataset for paired noisy and target (clean) functional connectomes.
    """

    def __init__(
        self,
        features: List[torch.Tensor],
        noisy_adjs: List[torch.Tensor],
        clean_adjs: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            features: List of normalized BOLD feature tensors, each (N_rois, T_timepoints)
            noisy_adjs: List of single-session noisy adjacency tensors, each (N_rois, N_rois)
            clean_adjs: Optional list of cross-session averaged target adjacency tensors
        """
        self.features = features
        self.noisy_adjs = noisy_adjs
        self.clean_adjs = clean_adjs if clean_adjs is not None else noisy_adjs

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat = self.features[idx]
        noisy_adj = self.noisy_adjs[idx]
        clean_adj = self.clean_adjs[idx]

        noisy_edge_index, noisy_edge_weight = matrix_to_edge_index_and_weight(noisy_adj)
        clean_edge_index, clean_edge_weight = matrix_to_edge_index_and_weight(clean_adj)

        return {
            "x": feat,
            "noisy_adj": noisy_adj,
            "clean_adj": clean_adj,
            "noisy_edge_index": noisy_edge_index,
            "noisy_edge_weight": noisy_edge_weight,
            "clean_edge_index": clean_edge_index,
            "clean_edge_weight": clean_edge_weight,
        }


class SyntheticConnectomeGenerator:
    """
    Generates synthetic multi-subject, multi-session resting-state fMRI connectomes
    with controllable modular structure, inter-subject clustering, and physiological noise.
    """

    def __init__(
        self,
        num_subjects: int = 50,
        num_rois: int = 264,
        timepoints: int = 1200,
        num_modules: int = 7,  # Standard Yeo-7 canonical brain networks
        num_subgroups: int = 4,
        noise_level: float = 0.35,
        random_seed: int = 42,
    ):
        self.num_subjects = num_subjects
        self.num_rois = num_rois
        self.timepoints = timepoints
        self.num_modules = num_modules
        self.num_subgroups = num_subgroups
        self.noise_level = noise_level
        self.rng = np.random.RandomState(random_seed)

    def generate(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], np.ndarray]:
        """
        Generates paired BOLD features, noisy single-session connectomes, and clean ground-truth targets.

        Returns:
            features: List of (N_rois, T_timepoints) tensors
            noisy_adjs: List of (N_rois, N_rois) tensors
            clean_adjs: List of (N_rois, N_rois) tensors
            subgroup_labels: Array of shape (num_subjects,) with true subgroup assignments
        """
        features = []
        noisy_adjs = []
        clean_adjs = []
        subgroup_labels = []

        # ROI to canonical module assignment
        module_assignments = np.array_split(np.arange(self.num_rois), self.num_modules)

        for i in range(self.num_subjects):
            # Assign subject to 1 of 4 natural cohorts
            subgroup = i % self.num_subgroups
            subgroup_labels.append(subgroup)

            # Generate latent modular neural activation signals
            module_signals = self.rng.randn(self.num_modules, self.timepoints)
            bold_signals = np.zeros((self.num_rois, self.timepoints))

            # Subgroup-specific modulation factor
            subgroup_mod = 1.0 + 0.2 * (subgroup - (self.num_subgroups / 2.0))

            for m_idx, rois in enumerate(module_assignments):
                for roi in rois:
                    # ROI signal = module ground truth signal + local neural variability
                    local_noise = self.rng.randn(self.timepoints) * 0.4
                    bold_signals[roi] = module_signals[m_idx] * subgroup_mod + local_noise

            # Standardize signals
            std_bold = standardize_bold_signals(bold_signals)
            feat_tensor = torch.from_numpy(std_bold).float()

            # Clean ground truth connectivity matrix
            clean_adj = compute_functional_connectivity(std_bold, positive_only=True, zero_diagonal=True)
            clean_tensor = torch.from_numpy(clean_adj).float()

            # Generate single-session noisy connectivity with scanner drift & motion artifacts
            noise_matrix = self.rng.randn(self.num_rois, self.num_rois) * self.noise_level
            noise_matrix = (noise_matrix + noise_matrix.T) / 2.0  # Keep symmetric
            np.fill_diagonal(noise_matrix, 0.0)

            noisy_adj = np.clip(clean_adj + noise_matrix, 0.0, 1.0)
            np.fill_diagonal(noisy_adj, 0.0)
            noisy_tensor = torch.from_numpy(noisy_adj).float()

            features.append(feat_tensor)
            clean_adjs.append(clean_tensor)
            noisy_adjs.append(noisy_tensor)

        return features, noisy_adjs, clean_adjs, np.array(subgroup_labels)
