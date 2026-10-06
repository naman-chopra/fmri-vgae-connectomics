# Functional Connectome Denoising with Variational Graph Autoencoders

This repository develops a graph-learning pipeline for denoising resting-state functional MRI connectomes and evaluating the stability of downstream community structure.

The project is motivated by an undergraduate research internship on group-level modular structure in functional MRI. It is currently a research engineering scaffold with a fully runnable synthetic benchmark. The real-data reproduction path is intentionally kept separate until data provenance, preprocessing, and institutional sharing constraints are documented.

## Research question

Can a variational graph autoencoder recover a more stable functional-connectivity structure from a noisy single-session connectome than direct thresholding or an unprocessed graph?

The working pipeline is:

```text
BOLD time series
        |
        v
ROI-level standardization and Pearson connectivity
        |
        v
Noisy weighted brain graph + node features
        |
        v
GCN encoder -> variational latent representation -> inner-product decoder
        |
        v
Denoised connectome -> community detection and modularity analysis
```

## Current status

Implemented:

- ROI-wise BOLD standardization and functional-connectivity construction
- Weighted graph conversion utilities
- Native PyTorch GCN encoder
- Variational graph autoencoder with an inner-product decoder
- Reconstruction, link-prediction, and modularity metrics
- Spectral clustering, KNEED-based subgroup exploration, and consensus clustering utilities
- Synthetic multi-subject connectome generator with controllable modules and noise
- End-to-end synthetic experiment with saved training and connectome visualizations

Not yet claimed:

- A validated result on the original internship dataset
- Generalization to unseen subjects or sessions
- A clinical or neuroscientific finding
- Publication-level benchmarking

## Quick start

Create an environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-deps
pip install torch-geometric numpy scipy scikit-learn kneed networkx matplotlib seaborn pandas pyyaml tqdm pytest
```

Run the synthetic end-to-end experiment:

```bash
python scripts/run_synthetic_demo.py --epochs 50 --output_dir demo_outputs
```

The command generates:

- a synthetic multi-subject connectome collection;
- a VGAE denoising run;
- reconstruction and link-prediction metrics;
- modularity comparisons for noisy, denoised, and target graphs;
- connectome comparison and training-curve figures in `demo_outputs/`.

## Repository layout

```text
configs/       Experiment configuration
scripts/       Runnable experiments
src/data/      BOLD preprocessing and dataset abstractions
src/models/    GCN and VGAE implementations
src/clustering/Community detection and consensus utilities
src/utils/     Metrics and visualization
tests/         Regression and scientific sanity checks
```

The historical internship material is not part of the public source distribution. It is retained locally under `raw_source/` for provenance and review, and is excluded from Git by default.

## Methodological notes

The synthetic benchmark creates a clean connectome from controlled modular BOLD signals and then adds symmetric edge noise. This is useful for testing implementation behavior, but it is not a substitute for held-out real fMRI evaluation.

Before a real-data experiment is presented as a result, the project should add:

1. a documented dataset and atlas acquisition procedure;
2. subject/session-level train, validation, and test separation;
3. baselines such as thresholding, shrinkage, and an ordinary autoencoder;
4. repeated-seed confidence intervals;
5. downstream community-stability measures in addition to reconstruction error.

## Reproducibility and scope

Experiments should record the random seed, data split, preprocessing choices, model configuration, and software environment. Results from the synthetic demo should be described as implementation validation, not as evidence that the method improves real neuroimaging analyses.

## License

The source code is intended to be released under the MIT License. Dataset copyrights, source papers, institutional documents, and third-party assets remain subject to their original licenses and are not relicensed by this repository.
