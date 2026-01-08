import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from src.config import DEVICE

def load_cancer_data(cancer_type, base_dir, omics_list):
    """Loads raw numpy files for a specific cancer type."""
    data_dir = os.path.join(base_dir, cancer_type)
    omics_data = {}
    try:
        for m in omics_list:
            path = os.path.join(data_dir, f"{m}_processed.npy")
            omics_data[m] = np.load(path)
        y = np.load(os.path.join(data_dir, "labels.npy"))
        return omics_data, y
    except FileNotFoundError:
        return None, None

def prepare_split(omics_data, y, tr_idx, te_idx):
    """Scales data and converts to Tensors for a specific fold."""
    Xtr, Xte = {}, {}
    
    # Scale each omic modality independently
    for m, X in omics_data.items():
        sc = StandardScaler()
        Xtr[m] = sc.fit_transform(X[tr_idx])
        Xte[m] = sc.transform(X[te_idx])

    # Convert to Tensors
    Xtr_t = {m: torch.tensor(v, dtype=torch.float32).to(DEVICE) for m, v in Xtr.items()}
    Xte_t = {m: torch.tensor(v, dtype=torch.float32).to(DEVICE) for m, v in Xte.items()}
    
    ytr = torch.tensor(y[tr_idx], dtype=torch.long).to(DEVICE)
    
    return Xtr_t, Xte_t, ytr, Xtr, Xte
