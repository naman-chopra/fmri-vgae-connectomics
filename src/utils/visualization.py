"""
Visualization Utilities for Brain Connectomes and Modularity Partitions.
"""

from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_tri_panel_connectome_comparison(
    noisy_adj: np.ndarray,
    denoised_adj: np.ndarray,
    clean_adj: np.ndarray,
    save_path: Optional[str] = None,
    cmap: str = "hot",
):
    """
    Plots side-by-side comparison of Noisy Input, VGAE Denoised, and Clean Target connectomes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax = max(np.max(noisy_adj), np.max(clean_adj), np.max(denoised_adj))

    im0 = axes[0].imshow(noisy_adj, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    axes[0].set_title("Single-Run Noisy Connectome ($A_{noisy}$)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("ROIs (Power-264)")
    axes[0].set_ylabel("ROIs (Power-264)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(denoised_adj, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    axes[1].set_title(r"VGAE Reconstructed ($\hat{A}$)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("ROIs (Power-264)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(clean_adj, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    axes[2].set_title("Target Clean Connectome ($A_{clean}$)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("ROIs (Power-264)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    losses: List[float],
    recon_losses: List[float],
    kl_losses: List[float],
    save_path: Optional[str] = None,
):
    """
    Plots training loss convergence curves (Total Loss, Reconstruction MSE, KL Divergence).
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(losses) + 1)

    ax1.plot(epochs, losses, label="Total Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, recon_losses, label="Recon Loss (MSE)", color="#2ca02c", linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(epochs, kl_losses, label="KL Divergence", color="#d62728", linestyle=":")
    ax2.set_ylabel("KL Divergence", color="#d62728", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("VGAE Connectome Denoising Loss Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
