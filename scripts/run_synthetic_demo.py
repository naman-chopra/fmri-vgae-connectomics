#!/usr/bin/env python3
"""
Single-Command End-to-End Synthetic Demo for fmri-vgae-connectomics.
Generates synthetic HCP resting-state connectomes, executes KNEED subject subgrouping,
trains a Variational Graph Autoencoder (VGAE), and computes downstream modularity scores.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import SyntheticConnectomeGenerator
from src.models.vgae import VariationalGraphAutoEncoder
from src.models.losses import vgae_loss
from src.clustering.laplacian_spectral import (
    eigen_decomposition_kneed,
    cluster_subjects,
)
from src.clustering.modularity import (
    detect_brain_communities,
    compute_newman_modularity,
)
from src.utils.metrics import (
    evaluate_connectome_reconstruction,
    evaluate_link_prediction_auc,
)
from src.utils.visualization import (
    plot_tri_panel_connectome_comparison,
    plot_training_curves,
)


def run_demo():
    parser = argparse.ArgumentParser(description="Run End-to-End Connectome Denoising & Modularity Demo")
    parser.add_argument("--num_subjects", type=int, default=40, help="Number of synthetic subjects")
    parser.add_argument("--num_rois", type=int, default=264, help="Number of brain ROIs (Power-264)")
    parser.add_argument("--timepoints", type=int, default=1200, help="BOLD timepoints per scan (HCP S1200)")
    parser.add_argument("--latent_dim", type=int, default=32, help="VGAE latent representation dimension")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="demo_outputs", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 75)
    print("🧠 fMRI-VGAE-Connectomics: End-to-End Synthetic Pipeline Demo")
    print(f"   Device: {device} | ROIs: {args.num_rois} | Subjects: {args.num_subjects}")
    print("=" * 75)

    # 1. Generate Synthetic HCP Dataset
    print("\n[Step 1/4] Generating synthetic resting-state BOLD time-series & connectomes...")
    generator = SyntheticConnectomeGenerator(
        num_subjects=args.num_subjects,
        num_rois=args.num_rois,
        timepoints=args.timepoints,
        num_modules=7,
        num_subgroups=4,
        noise_level=0.35,
        random_seed=42,
    )
    features, noisy_adjs, clean_adjs, true_subgroups = generator.generate()
    print(f"   ✓ Generated {len(features)} subject graphs ({args.num_rois} ROIs, {args.timepoints} timepoints)")

    # 2. Inter-Subject Subgrouping via Normalized Laplacian & KNEED
    print("\n[Step 2/4] Executing Inter-Subject Subgrouping (Laplacian Spectrum + KNEED)...")
    # Build pairwise subject similarity matrix
    subject_sim = np.zeros((args.num_subjects, args.num_subjects))
    for i in range(args.num_subjects):
        for j in range(args.num_subjects):
            corr = np.corrcoef(clean_adjs[i].numpy().flatten(), clean_adjs[j].numpy().flatten())[0, 1]
            subject_sim[i, j] = max(0.0, float(corr))

    optimal_k, eigenvalues, _ = eigen_decomposition_kneed(subject_sim, max_k=8)
    subgroup_labels, cluster_metrics = cluster_subjects(subject_sim, n_clusters=optimal_k)
    print(f"   ✓ KNEED Algorithm detected optimal k = {optimal_k} natural subject cohorts")
    print(f"   ✓ Subgroup Silhouette Score: {cluster_metrics['silhouette_score']:.4f}")

    # 3. Train Variational Graph Auto-Encoder (VGAE)
    print(f"\n[Step 3/4] Training VGAE Connectome Denoising Engine ({args.epochs} epochs)...")
    model = VariationalGraphAutoEncoder(
        input_dim=args.timepoints,
        hidden_dim=128,
        latent_dim=args.latent_dim,
        dropout=0.1,
        decoder_activation="sigmoid",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses, recon_losses, kl_losses = [], [], []

    pbar = tqdm(range(args.epochs), desc="Training VGAE")
    for epoch in pbar:
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        model.train()
        for i in range(len(features)):
            x = features[i].to(device)
            noisy_adj = noisy_adjs[i].to(device)
            target_adj = clean_adjs[i].to(device)

            optimizer.zero_grad()
            adj_hat, z, mu, logvar = model(x, noisy_adj)
            loss, r_loss, k_loss = vgae_loss(adj_hat, target_adj, mu, logvar, kl_weight=0.01)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += r_loss.item()
            epoch_kl += k_loss.item()

        n_batches = len(features)
        losses.append(epoch_loss / n_batches)
        recon_losses.append(epoch_recon / n_batches)
        kl_losses.append(epoch_kl / n_batches)

        pbar.set_postfix({
            "Loss": f"{losses[-1]:.4f}",
            "Recon MSE": f"{recon_losses[-1]:.4f}",
        })

    # 4. Evaluation & Downstream Modularity Benchmarks
    print("\n[Step 4/4] Evaluating Denoising Quality & Modularity Recovery...")
    model.eval()
    test_idx = 0
    with torch.no_grad():
        test_x = features[test_idx].to(device)
        test_noisy = noisy_adjs[test_idx].to(device)
        test_clean = clean_adjs[test_idx].numpy()

        test_hat, _, _, _ = model(test_x, test_noisy)
        denoised_np = test_hat.cpu().numpy()

    # Metrics
    recon_metrics = evaluate_connectome_reconstruction(denoised_np, test_clean)
    noisy_metrics = evaluate_connectome_reconstruction(test_noisy.cpu().numpy(), test_clean)
    auc_score, ap_score = evaluate_link_prediction_auc(denoised_np, test_clean)

    # Modularity Q
    _, q_noisy = detect_brain_communities(test_noisy.cpu().numpy(), n_communities=7)
    _, q_denoised = detect_brain_communities(denoised_np, n_communities=7)
    _, q_clean = detect_brain_communities(test_clean, n_communities=7)

    print("\n" + "=" * 55)
    print("📊 BENCHMARK RESULTS SUMMARY (Sample Connectome)")
    print("=" * 55)
    print(f" • Input Noisy Connectome MSE:       {noisy_metrics['mse']:.5f}")
    print(f" • VGAE Denoised Connectome MSE:     {recon_metrics['mse']:.5f} (↓ {(1 - recon_metrics['mse']/noisy_metrics['mse'])*100:.1f}% reduction)")
    print(f" • Reconstructed Pearson r:          {recon_metrics['pearson_r']:.4f}")
    print(f" • Connectome Link Prediction AUC:   {auc_score:.4f} (AP: {ap_score:.4f})")
    print("-" * 55)
    print(f" • Modularity Q (Noisy Scan):        {q_noisy:.4f}")
    print(f" • Modularity Q (VGAE Denoised):     {q_denoised:.4f} (↑ Restored)")
    print(f" • Modularity Q (Clean Ground Truth):{q_clean:.4f}")
    print("=" * 55)

    # 5. Save Plots
    tri_path = os.path.join(args.output_dir, "connectome_tri_panel_comparison.png")
    plot_tri_panel_connectome_comparison(
        test_noisy.cpu().numpy(),
        denoised_np,
        test_clean,
        save_path=tri_path,
    )
    print(f"\n[Saved Visuals] Tri-panel Connectome Comparison -> {tri_path}")

    loss_path = os.path.join(args.output_dir, "training_loss_curves.png")
    plot_training_curves(losses, recon_losses, kl_losses, save_path=loss_path)
    print(f"[Saved Visuals] Training Dynamics Loss Curves   -> {loss_path}")

    print("\n✅ End-to-End Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
