# fMRI Connectomics Pipeline Flow

## Aim

The project aims to reproduce and extend the internship research pipeline for HCP resting-state fMRI: construct subject-level Power-264 functional connectomes, learn a denoised graph representation with a variational graph autoencoder, compare subject-level structure, and evaluate downstream clustering/module stability.

## Provenance and source data

The reference source is the local `HCP_rs_power` MATLAB archive recovered from the original project data. It contains four runs per subject, with `TS` arrays of shape `(264, 1200)` and `Corre` matrices of shape `(264, 264)`. Shape-aware validation produced 818 usable subjects, matching the historical `patients.txt` overlap.

The NumPy `ts_power_s1200` archive is a separate 213-subject processed subset and is not silently merged with the MATLAB cohort. Raw archives and extracted/cache data are excluded from Git.

## Intended pipeline

```text
HCP MATLAB runs
    -> validate four-run subject completeness and array shapes
    -> standardize BOLD signals and average four run correlations
    -> construct positive or signed weighted connectomes
    -> inject controlled synthetic edge noise for held-out evaluation
    -> VGAE: GCN encoder -> latent node representation -> graph decoder
    -> compare denoised graph with clean held-out connectome
    -> subject similarity and spectral clustering/KNEED
    -> ICSC/module/community stability analysis
```

## What was improvised

- ZIP-backed lazy loading was used initially to preserve the source archive, then replaced for experiments by an external-SSD extraction and caches because repeated MATLAB decompression/parsing left the GPU idle.
- A dense matrix autoencoder was added as a diagnostic baseline, not as the proposed research model.
- Positive-only adjacency was used for the denoising experiments to avoid signed-noise ambiguity; the historical clustering path used signed correlations. These are separate analyses and must not be compared as identical protocols.
- A connectome-feature VGAE was added as a diagnostic variant because the compact adjacency cache does not contain BOLD features. The primary VGAE path now uses cached standardized four-run BOLD features with `264 x 4800` node features.
- GPU acceleration is applied to model training and subject-similarity matrix multiplication; MATLAB parsing remains CPU/I/O-bound.

## Completed

- Archive inventory, shape validation, source/cohort reconciliation, and `.gitignore` protection for data.
- Extracted source archive on external SSD and created caches for 818 subjects:
  - averaged positive connectomes: `(818, 264, 264)`;
  - standardized four-run BOLD features: `(818, 264, 4800)`.
- Full-cohort dense denoiser at 30% noise: test MSE `0.00718` versus noisy baseline `0.04490`.
- Full-cohort connectome-feature VGAE at 30% noise: test MSE `0.01036`.
- Initial BOLD-feature VGAE at 30% noise: test MSE `0.48302`; this failed the baseline gate and is being stabilized before interpretation.
- Original-style four-cluster spectral analysis: KNEED detected `k=4` on the cached positive-connectome protocol.
- GPU monitoring, reproducible subject splits, reports, and regression tests (`5 passed`).

## Current decision gates

1. Stabilize the BOLD-feature VGAE and require held-out MSE to beat the noisy baseline before downstream interpretation.
2. Run the 5%, 15%, and 30% noise sweep under the same split and report identity, dense, connectome-feature VGAE, and BOLD-feature VGAE baselines.
3. Only then cluster denoised outputs and compare cluster stability with the historical signed-connectome clustering.
4. Complete module/ICSC analysis and figures after the denoiser passes the reconstruction and stability gates.

## Reproducibility rule

Every result must record source path/cache, cohort size, feature source, signed/positive adjacency choice, noise fraction, split seed, model configuration, GPU status, and output report path. No downstream biological claim is made from a model that fails the held-out baseline gate.
