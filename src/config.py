import torch
import os

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
MAX_EPOCHS = 500
MIN_EPOCHS = 100
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

# Go up one level from the current script directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.join(PROJECT_ROOT, "preprocessing", "processed_multicancer")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_analysis")
DIAGRAMS_DIR = os.path.join(PROJECT_ROOT, "paper_diagrams")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DIAGRAMS_DIR, exist_ok=True)


GS_TYPES = ["GS-BRCA", "GS-LGG", "GS-OV", "GS-COAD", "GS-GBM"]
OMICS = ["mRNA", "miRNA", "CNV", "Methy"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

