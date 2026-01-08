import torch
import os
import numpy as np

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Paths
DATA_ROOT = "./preprocessing/processed_multicancer"
RESULTS_ROOT = "./research_results"

# Data Config
GS_TYPES = ["GS-BRCA", "GS-GBM", "GS-COAD", "GS-LGG", "GS-OV"]
OMICS = ["mRNA", "miRNA", "CNV", "Methy"]
TOP_K_GENES = 20

# Training Hyperparameters
N_SPLITS = 5
MAX_EPOCHS = 300
MIN_EPOCHS = 80
PATIENCE = 40
LR = 1e-3
WEIGHT_DECAY = 5e-4

def setup_directories():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
