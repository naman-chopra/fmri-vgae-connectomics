# Original internship pipeline

This document records the implementation path found in the archived IIT Ropar material. It is a reconstruction aid, not a claim that every notebook belongs to one final experiment.

## Evidence reviewed

- `IIT_RPR/code/load_data.py`
- `IIT_RPR/code/vgae_2.ipynb`, `vgae_22.ipynb`, `vgae_23.ipynb`, and `vgae_working.ipynb`
- `IIT_RPR/code/Timeseries_concatenation_200.ipynb` and `Timeseries_concatenation_800.ipynb`
- `IIT_RPR/code/Averaging_corr_200.ipynb`, `add_noise.ipynb`, and grouping notebooks

## Reconstructed data flow

```text
HCP subject/session time series
        |
        v
Concatenate selected sessions per subject
        |
        +--> ROI-by-time-series arrays --> VGAE node features
        |
        +--> correlation matrices --> averaged clean targets
                              \--> synthetic noisy matrices
                                           |
                                           v
                                  GCN/VGAE denoising
                                           |
                                           v
                                  community analysis
```

The repeated references to `timeseries_818`, `ts_1000`, and `timeseries_final` indicate that subject filtering and subset size changed across experiments. The notebooks also use both 4,800-feature and 1,200-feature configurations. The 4,800-feature configuration is consistent with concatenating four 1,200-timepoint sessions, but this must be verified from the actual arrays before being treated as canonical.

## Directly supported by the archive

- The graphs use 264 ROIs.
- The original loader pairs clean adjacency, noisy adjacency, and feature matrices by directory iteration.
- Noisy adjacency diagonals are explicitly set to zero.
- Node features are normalized column-wise in the original loader.
- The model family is a graph autoencoder/VGAE with an inner-product adjacency decoder.
- Clean targets are averaged or otherwise prepared adjacency matrices; noise is added in separate preprocessing notebooks.
- Subject-level similarity and elbow/KNEED analysis are used downstream for grouping.

## Ambiguities to resolve

1. Which subject set is canonical: 200, 818, or 1,000 subjects?
2. Which feature width is canonical: 1,200 or 4,800?
3. How are files aligned across the three directories?
4. Is the clean matrix a cross-session average, a group average, or another target?
5. Which noise branch is intended: signed Gaussian, positive-only, or repeated positive noise?
6. Which VGAE notebook represents the final internship result?
7. Are the saved arrays and metadata shareable, or only usable for local reproduction?

## Reproduction policy

The implementation in `src/` should first reproduce the original data contract and baseline behavior. It should not silently correct methodological choices. Improvements such as explicit file matching, held-out splits, alternative losses, and stronger baselines belong in separate experiments.
