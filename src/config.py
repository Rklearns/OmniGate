import torch
import numpy as np

# Hardware & Reproducibility
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

# Paths
BASE_DIR = "./preprocessing/processed_multicancer"
RESULTS_ROOT = "./results_all_cancers"

# Data
GS_TYPES = ["GS-BRCA", "GS-LGG", "GS-OV", "GS-COAD", "GS-GBM"]
OMICS = ["mRNA", "miRNA", "CNV", "Methy"]
N_SPLITS = 5

# Hyperparameters
MAX_EPOCHS = 400
MIN_EPOCHS = 100
PATIENCE = 45
LR = 1e-3
WEIGHT_DECAY = 5e-4
OMICS_DROPOUT_P = 0.2

# Loss Weights
ALIGN_W = 0.2
GATE_ENT_W = 0.05

# Analysis
TOP_K_GENES = 20
