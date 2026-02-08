import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_class_weight
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import from our modules
from config import *
from models import GatedMultiOmicsClassifier, FocalLoss, alignment_loss, orthogonality_loss, gate_entropy
from utils import generate_aggregated_reports

# ENSURE REPRODUCIBILITY
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def train_and_collect(omics_data, y, tr, te, accumulators):
    # 1. Data Prep
    Xtr, Xte = {}, {}
    for m, X in omics_data.items():
        sc = StandardScaler()
        Xtr[m] = sc.fit_transform(X[tr])
        Xte[m] = sc.transform(X[te])

    Xtr_t = {m: torch.tensor(v, dtype=torch.float32).to(DEVICE) for m, v in Xtr.items()}
    Xte_t = {m: torch.tensor(v, dtype=torch.float32).to(DEVICE) for m, v in Xte.items()}
    ytr_t = torch.tensor(y[tr], dtype=torch.long).to(DEVICE)

    # 2. Model Setup
    w = compute_class_weight("balanced", classes=np.unique(y[tr]), y=y[tr])
    w = torch.tensor(w, dtype=torch.float32).to(DEVICE)
    loss_fn = FocalLoss(alpha=w)
    
    in_dims = {m: Xtr[m].shape[1] for m in Xtr}
    model = GatedMultiOmicsClassifier(in_dims, len(np.unique(y))).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 3. Training Loop
    best_prec = 0
    best_state = None
    wait = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        
        train_inputs = {k: v.clone() for k,v in Xtr_t.items()}
        if torch.rand(1).item() < OMICS_DROPOUT_P:
            drop_m = np.random.choice(list(train_inputs.keys()))
            train_inputs[drop_m] = torch.zeros_like(train_inputs[drop_m])

        logits, zs, gates = model(train_inputs)
        
        loss = (loss_fn(logits, ytr_t) + 
                ALIGN_W * alignment_loss(zs) + 
                ORTHO_W * orthogonality_loss(zs) +
                GATE_ENT_W * sum(gate_entropy(g) for g in gates.values()) +
                SPARSITY_W * sum(g.mean() for g in gates.values()))
        
        loss.backward()
        opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            preds = model(Xte_t)[0].argmax(1).cpu().numpy()
            prec = precision_score(y[te], preds, average="weighted", zero_division=0)
            
        if prec > best_prec:
            best_prec = prec
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if epoch >= MIN_EPOCHS and wait >= PATIENCE: break

    # 4. Collection for Aggregation
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    
    # A. Predictions
    with torch.no_grad():
        logits, zs_list, gates_map = model(Xte_t)
        preds = logits.argmax(1).cpu().numpy()
        accumulators['y_true'].extend(y[te])
        accumulators['y_pred'].extend(preds)
        
        # B. Latent Vectors (Concat for t-SNE)
        z_global = torch.cat(zs_list, dim=1).cpu().numpy()
        accumulators['latent'].append(z_global)
        
        # C. Gate Values
        for m in OMICS:
            accumulators['gates'][m].append(gates_map[m].cpu().numpy().mean())

    # D. Gene Sensitivity (Summed Gradients)
    for m in OMICS:
        inputs = Xte_t[m].clone().requires_grad_(True)
        full_inputs = {k: Xte_t[k] for k in OMICS}
        full_inputs[m] = inputs
        
        logits, _, _ = model(full_inputs)
        score = logits.max(dim=1)[0].mean()
        score.backward()
        
        sens = inputs.grad.abs().mean(dim=0).cpu().numpy()
        if accumulators['sensitivity'][m] is None:
             accumulators['sensitivity'][m] = sens
        else:
             accumulators['sensitivity'][m] += sens

    return best_prec

if __name__ == "__main__":
    print(f"Starting Multi-Cancer Analysis (Seed={SEED})...")
    print(f"Diagrams will be saved to: {DIAGRAMS_DIR}")
    
    performance_records = []

    for CANCER in GS_TYPES:
        print(f"\n==================== {CANCER} ====================")
        DATA_DIR = os.path.join(BASE_DIR, CANCER)
        
        try:
            omics_data = {
                "mRNA": np.load(os.path.join(DATA_DIR, "mRNA_processed.npy")),
                "miRNA": np.load(os.path.join(DATA_DIR, "miRNA_processed.npy")),
                "CNV": np.load(os.path.join(DATA_DIR, "CNV_processed.npy")),
                "Methy": np.load(os.path.join(DATA_DIR, "Methy_processed.npy")),
            }
            y = np.load(os.path.join(DATA_DIR, "labels.npy"))
        except FileNotFoundError:
            print(f"Skipping {CANCER} (Files not found)")
            continue

        accumulators = {
            'y_true': [],
            'y_pred': [],
            'gates': {m: [] for m in OMICS},
            'latent': [],
            'sensitivity': {m: None for m in OMICS}
        }

        # Dynamic Splits
        min_class_size = np.min(np.unique(y, return_counts=True)[1])
        n_splits = max(2, min_class_size) if min_class_size < DEFAULT_N_SPLITS else DEFAULT_N_SPLITS
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        
        scores = []
        for i, (tr, te) in enumerate(skf.split(omics_data["mRNA"], y)):
            print(f"  > Processing Fold {i}...")
            prec = train_and_collect(omics_data, y, tr, te, accumulators)
            scores.append(prec)

        avg_prec = np.mean(scores)
        print(f"  > {CANCER} Avg Precision: {avg_prec:.4f}")
        
        # Log Performance
        performance_records.append({
            "Cancer_Subtype": CANCER,
            "Mean_Precision": avg_prec,
            "Num_Folds": n_splits,
            "Min_Class_Size": min_class_size
        })
        
        # --- Generate Final "Legit" Diagrams ---
        generate_aggregated_reports(CANCER, accumulators)

    # Save Summary CSV
    perf_df = pd.DataFrame(performance_records)
    perf_df.to_csv(os.path.join(RESULTS_DIR, "performance_summary.csv"), index=False)
    print("\nALL PROCESSES COMPLETED.")
    print(f"Performance summary saved to {os.path.join(RESULTS_DIR, 'performance_summary.csv')}")
