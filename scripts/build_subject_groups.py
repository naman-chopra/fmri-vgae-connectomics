"""Build a reproducible subject-similarity matrix and spectral cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from kneed import KneeLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering.laplacian_spectral import cluster_subjects, eigen_decomposition_kneed
from src.data.dataset import HCPMatlabConnectomeDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("original-data/HCP_rs_power.zip"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/subject-groups"))
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--similarity", choices=("cosine", "pearson"), default="cosine")
    parser.add_argument("--legacy-kneed", action="store_true")
    args = parser.parse_args()

    if args.cache:
        vectors_full = np.load(args.cache / "averaged_positive_adjacencies.npy")
        subjects = json.loads((args.cache / "subjects.json").read_text())
        matrices = vectors_full[: args.max_subjects] if args.max_subjects else vectors_full
        subjects = subjects[: len(matrices)]
        vectors = matrices[:, np.triu_indices(264, k=1)[0], np.triu_indices(264, k=1)[1]].astype(np.float32)
    else:
        dataset = HCPMatlabConnectomeDataset(args.archive)
        subjects = dataset.subjects[: args.max_subjects] if args.max_subjects else dataset.subjects
        dataset = HCPMatlabConnectomeDataset(args.archive, subjects=subjects)
        vectors = []
        for index in range(len(dataset)):
            sample = dataset[index]
            matrix = sample["adj"].numpy()
            vectors.append(matrix[np.triu_indices(matrix.shape[0], k=1)])
        vectors = np.asarray(vectors, dtype=np.float32)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    # This matmul is the GPU-accelerated stage. Cosine is the original
    # internship calculation; Pearson is retained as a comparison mode.
    tensor = torch.from_numpy(vectors).to(device)
    if args.similarity == "pearson":
        tensor = tensor - tensor.mean(dim=1, keepdim=True)
    tensor = tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-8)
    similarity = (tensor @ tensor.T).clamp(-1.0, 1.0)
    similarity = similarity.cpu().numpy()
    if args.similarity == "pearson":
        similarity = (similarity + 1.0) / 2.0
    del tensor
    if device == "cuda":
        torch.cuda.synchronize()
    np.fill_diagonal(similarity, 0.0)

    if args.legacy_kneed:
        from scipy.sparse import csgraph
        laplacian = csgraph.laplacian(similarity, normed=True)
        eigenvalues, _ = np.linalg.eig(laplacian)
        detected = KneeLocator(
            np.arange(len(eigenvalues)), eigenvalues,
            curve="concave", direction="increasing",
        ).knee
        detected_k = int(detected) if detected is not None else -1
    else:
        detected_k, eigenvalues, _ = eigen_decomposition_kneed(
            similarity, max_k=min(10, len(subjects) - 1)
        )
    labels, metrics = cluster_subjects(similarity, n_clusters=args.clusters, random_state=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "subject_similarity.npy", similarity)
    np.save(args.output_dir / "subject_connectome_vectors.npy", vectors)
    np.save(args.output_dir / "subject_labels.npy", labels)
    (args.output_dir / "subject_ids.json").write_text(json.dumps(subjects, indent=2))
    report = {
        "archive": str(args.archive),
        "similarity_device": device,
        "similarity_method": args.similarity,
        "legacy_kneed": args.legacy_kneed,
        "subjects": len(subjects),
        "rois": 264,
        "connectome_features": int(vectors.shape[1]),
        "requested_clusters": args.clusters,
        "kneed_detected_k": int(detected_k),
        "cluster_sizes": {str(int(label)): int(np.sum(labels == label)) for label in np.unique(labels)},
        "metrics": metrics,
        "eigenvalues_first_10": np.real_if_close(eigenvalues[:10]).real.tolist(),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
