from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
MAX_EPOCHS = 500
MIN_EPOCHS = 150
PATIENCE = 45
LR = 1e-3
WEIGHT_DECAY = 5e-4

# Logic & Loss Weights
ALIGN_W = 0.2
ORTHO_W = 0.1
GATE_ENT_W = 0.05
SPARSITY_W = 0.01

OMICS_DROPOUT_P = 0.15
TOP_K_GENES = 20
DEFAULT_N_SPLITS = 5
LATENT_DIM = 128

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "preprocessing", "processed_multicancer")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results_aggregated")

GS_TYPES = ["GS-BRCA", "GS-LGG", "GS-OV", "GS-COAD", "GS-GBM"]
OMICS = ["mRNA", "miRNA", "CNV", "Methy"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_environment() -> None:
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    warnings.filterwarnings("ignore")
