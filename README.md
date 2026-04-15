# OmniGate: Omics-Integrated Gating for Multi-Cancer Subtype Classification

<p align="center">
  <a href="#dataset-description">
    <img src="https://img.shields.io/badge/Dataset-MLOmics-blue?style=for-the-badge" />
  </a>
  <a href="#repository-layout">
    <img src="https://img.shields.io/badge/Project-Structure-green?style=for-the-badge" />
  </a>
  <a href="#how-to-train">
    <img src="https://img.shields.io/badge/Train-Pipeline-orange?style=for-the-badge" />
  </a>
  <a href="#outputs">
    <img src="https://img.shields.io/badge/Results-Outputs-purple?style=for-the-badge" />
  </a>
  <a href="#configuration">
    <img src="https://img.shields.io/badge/Config-Settings-red?style=for-the-badge" />
  </a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/XGBoost-Ablation-success?style=flat-square" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Multi--Omics-Cancer%20Classification-purple?style=flat-square" />
</p>

> OMNIGATE is a deep learning framework designed for robust multi-modal cancer subtype classification. Unlike traditional fusion methods that simply concatenate features, OMNIGATE utilizes a dynamic context gating mechanism that learns to weigh the importance of specific omics layers (mRNA, miRNA, CNV, Methylation) on a per-sample basis.


## Dataset Description
The dataset used in this study is extracted from the [MLOmics dataset](https://github.com/chenzRG/Cancer-Multi-Omics-Benchmark), a publicly available multi-omics benchmark designed for cancer subtype classification tasks. It integrates heterogeneous molecular data collected from large-scale cancer genomics projects.
## Overview

The core model learns a latent representation for each omics modality and then applies a context-aware gate to each latent block before final classification. This lets the network dynamically emphasize the most informative modality for each sample instead of relying on static concatenation alone.

The current `src` pipeline supports:

- Multi-cancer training across `GS-BRCA`, `GS-LGG`, `GS-OV`, `GS-COAD`, and `GS-GBM`
- Gated multi-omics neural fusion with focal loss and regularization terms
- Dynamic fold selection based on the minimum class count
- Classifier-head ablation with `Base_MLP`, `SVM`, `XGBoost`, and `Deeper_MLP`
- Aggregated gate-importance plots and Top-20 feature sensitivity plots
- Fold-wise and global CSV export for downstream analysis
  
![Pipeline overview](docs/assets/omnigate_pipeline.png)



## Repository Layout

```text
OMNIGATE/
├── preprocessing/
│   └── processed_multicancer/
│       └── GS-*/                      # Per-cancer processed arrays and feature-name files
├── results_aggregated/               # Generated outputs after training
├── src/
│   ├── config.py                     # Global settings, paths, runtime configuration
│   ├── data.py                       # Dataset and feature-name loading
│   ├── models.py                     # Losses and gated fusion network
│   ├── training.py                   # Fold training and classifier ablation
│   ├── reporting.py                  # Plotting and CSV export
│   └── main.py                       # End-to-end training entrypoint
├── docs/
│   └── assets/                       # README figures
├── final_ablation_summary_all_cancers.csv
└── requirements.txt

```

## Data Format

Each cancer directory under `preprocessing/processed_multicancer/` is expected to contain:

- `mRNA_processed.npy`
- `miRNA_processed.npy`
- `CNV_processed.npy`
- `Methy_processed.npy`
- `labels.npy`
- `mRNA_features.json`
- `miRNA_features.json`
- `CNV_features.json`
- `Methy_features.json`

Example:

```text
preprocessing/processed_multicancer/GS-BRCA/
├── mRNA_processed.npy
├── miRNA_processed.npy
├── CNV_processed.npy
├── Methy_processed.npy
├── labels.npy
├── mRNA_features.json
├── miRNA_features.json
├── CNV_features.json
└── Methy_features.json
```

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The training pipeline expects these main packages:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `captum`

## How To Train

Run the full multi-cancer pipeline from the repository root:

```bash
python -m src.main
```

This command will:

1. Load each cancer dataset from `preprocessing/processed_multicancer/`
2. Train the gated fusion model with stratified cross-validation
3. Evaluate alternative classifier heads on the learned fused representation
4. Compute sensitivity-based Top-20 feature rankings for each omics modality
5. Save all aggregated figures and CSV summaries to `results_aggregated/`

## Outputs

After training, the pipeline writes outputs such as:

- `results_aggregated/final_ablation_summary_all_cancers.csv`
- `results_aggregated/<CANCER>/detailed_ablation_results.csv`
- `results_aggregated/<CANCER>/aggregated_gate_importance.png`
- `results_aggregated/<CANCER>/aggregated_top20_mRNA.csv`
- `results_aggregated/<CANCER>/aggregated_top20_mRNA.png`

Equivalent Top-20 feature files are also produced for `miRNA`, `CNV`, and `Methy`.

## Configuration

Main runtime settings are defined in `src/config.py`, including:

- `MAX_EPOCHS`
- `MIN_EPOCHS`
- `PATIENCE`
- `LR`
- `WEIGHT_DECAY`
- `ALIGN_W`
- `ORTHO_W`
- `GATE_ENT_W`
- `SPARSITY_W`
- `OMICS_DROPOUT_P`
- `LATENT_DIM`

If you want to adapt the pipeline for new experiments, this is the first file to modify.

## Method Summary

The model trains one encoder per modality, concatenates latent vectors to build global context, and then predicts modality-specific gates from that context. The gated latent vectors are fused and passed into a classifier head. Training combines focal loss with alignment, orthogonality, gate-entropy, and sparsity terms to improve robustness and reduce redundant modality usage.

The ablation workflow reuses learned fused embeddings and compares:

- Neural baseline head
- SVM head
- XGBoost head
- Deeper MLP head

This design makes it easier to test whether performance gains come from the representation itself, the classifier head, or both.

## Reproducibility

- Random seeds are fixed in `src/config.py`
- Fold generation uses `StratifiedKFold`
- CUDA is used automatically when available
- Output directories are created automatically on startup

## Intended Use

This codebase is structured for research experimentation, internal benchmarking, and figure generation around multi-omics cancer subtype classification. For production or clinical deployment, additional dataset validation, calibration, uncertainty estimation, and external evaluation would be required.
