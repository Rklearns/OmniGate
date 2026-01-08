import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

# ============================================================
# IMPORTS FROM LOCAL MODULES
# ============================================================
from src.config import (
    BASE_DIR, RESULTS_ROOT, GS_TYPES, OMICS, N_SPLITS, TOP_K_GENES, 
    set_seed, DEVICE
)
from src.dataset import load_cancer_data, prepare_split
from src.utils import train_single_fold

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    #  Setup
    set_seed()
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    #  Iterate over Cancer Types
    for CANCER in GS_TYPES:
        print(f"\n==================== {CANCER} ====================")
        OUT_DIR = os.path.join(RESULTS_ROOT, CANCER)
        os.makedirs(OUT_DIR, exist_ok=True)

        #  Load Data
        omics_data, y = load_cancer_data(CANCER, BASE_DIR, OMICS)
        if omics_data is None:
            print(f"Skipping {CANCER} (missing files in {BASE_DIR})")
            continue

        #  Cross-Validation Initialization
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        
        fold_scores = []
        gate_means = defaultdict(list)
        latent_collect = defaultdict(list)
        
        # We need a reference model/data from the last fold for feature importance
        last_model = None
        last_Xte_t = None

        # 5. Training Loop
        for fold_idx, (tr, te) in enumerate(skf.split(omics_data["mRNA"], y)):
            # Prepare data for this fold
            Xtr_t, Xte_t, ytr_t, Xtr_numpy, _ = prepare_split(omics_data, y, tr, te)
            input_dims = {m: Xtr_numpy[m].shape[1] for m in Xtr_numpy}

            # Train the model
            prec, model = train_single_fold(
                omics_data, y, tr, te, Xtr_t, Xte_t, ytr_t, input_dims
            )
            fold_scores.append(prec)
            
            # Store for post-hoc analysis
            last_model = model
            last_Xte_t = Xte_t

            # Collect Gate & Latent stats for this fold
            model.eval()
            with torch.no_grad():
                # Forward pass on test set
                _, zs, gates = model(Xte_t)
                
                # Store Gate values
                for m in OMICS:
                    gate_means[m].append(gates[m].mean().item())
                
                # Store Latent representations (for CORI)
                # zs is a list aligned with OMICS list order in model
                # We assume model.encoders iterates in same order as OMICS list if dict is ordered
                # Safer way: model returns 'gates' dict, but 'zs' list. 
                # Let's map zs back to omics names based on list index if OMICS is fixed.
                for i, m in enumerate(OMICS): 
                    # Note: We rely on OMICS order being consistent.
                    # GatedMultiOmicsClassifier.forward iterates self.encoders.
                    # As long as Python 3.7+ is used, dict insertion order is preserved.
                    latent_collect[m].append(torch.abs(zs[i]).mean(dim=0).cpu().numpy())
            
            print(f"  Fold {fold_idx+1}/{N_SPLITS} Precision: {prec:.4f}")

        # ========================================================
        #  SAVE METRICS & PLOTS
        # ========================================================

        # --- A. Performance ---
        perf_df = pd.DataFrame({
            "Mean_Precision": [np.mean(fold_scores)],
            "Std_Precision": [np.std(fold_scores)]
        })
        perf_df.to_csv(os.path.join(OUT_DIR, "performance.csv"), index=False)
        print(f"  > Mean Precision: {np.mean(fold_scores):.4f}")

        # ---  Gate Importance ---
        mean_gate = {m: np.mean(gate_means[m]) for m in OMICS}
        pd.DataFrame(mean_gate.items(), columns=["Omics", "Mean_Gate"]).to_csv(
            os.path.join(OUT_DIR, "gate_importance.csv"), index=False
        )

        plt.figure(figsize=(6,4))
        plt.bar(mean_gate.keys(), mean_gate.values(), color='skyblue', edgecolor='black')
        plt.title(f"{CANCER}: Gate Importance (Modality Relevance)")
        plt.ylabel("Avg Gate Value (0-1)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "gate_importance.png"), dpi=300)
        plt.close()

        # ---  FAEC (Gate Stability) ---
        # FAEC here is defined as the standard deviation of gate values across folds (lower is better/more stable)
        faec = {m: np.std(gate_means[m]) for m in OMICS}
        
        plt.figure(figsize=(6,4))
        plt.bar(faec.keys(), faec.values(), color='salmon', edgecolor='black')
        plt.title(f"{CANCER}: FAEC (Gate Stability - Std Dev)")
        plt.ylabel("Standard Deviation")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "faec_gate_stability.png"), dpi=300)
        plt.close()

        # ---  CORI (Cross-Omics Redundancy Index) ---
        # Correlation between the mean latent representations of omics pairs
        mean_latent = {m: np.mean(latent_collect[m], axis=0) for m in OMICS}
        cori_rows = []

        for i in range(len(OMICS)):
            for j in range(i + 1, len(OMICS)):
                m1, m2 = OMICS[i], OMICS[j]
                # Sort latent activations to compare distribution shapes
                v1 = np.sort(mean_latent[m1])[::-1]
                v2 = np.sort(mean_latent[m2])[::-1]
                
                # Ensure lengths match (should be 128 based on model, but safe check)
                L = min(len(v1), len(v2))
                corr = np.corrcoef(v1[:L], v2[:L])[0, 1]
                
                cori_rows.append({
                    "Pair": f"{m1}-{m2}",
                    "CORI": abs(corr)
                })

        cori_df = pd.DataFrame(cori_rows)
        cori_df.to_csv(os.path.join(OUT_DIR, "cori.csv"), index=False)

        if not cori_df.empty:
            plt.figure(figsize=(8, 4))
            plt.bar(cori_df["Pair"], cori_df["CORI"], color='lightgreen', edgecolor='black')
            plt.xticks(rotation=30)
            plt.title(f"{CANCER}: CORI (Latent Redundancy)")
            plt.ylabel("Correlation Coefficient")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "cori_plot.png"), dpi=300)
            plt.close()

        # --- Top-20 Genes (Sensitivity Analysis) ---
        if last_model:
            print("  > Calculating Top-20 Genes via Gradient Sensitivity...")
            for m in OMICS:
                encoder = last_model.encoders[m]
                
                # Clone test data and enable gradients
                X = last_Xte_t[m].clone().detach().requires_grad_(True)
                
                # Forward pass through just the encoder
                z = encoder(X)
                
                # Objective: Maximize norm of latent embedding
                score = torch.norm(z, dim=1).mean()
                score.backward()

                # Calculate sensitivity (mean absolute gradient)
                sens = X.grad.abs().mean(dim=0).cpu().numpy()
                
                # Get Top K indices
                idx = np.argsort(sens)[::-1][:TOP_K_GENES]
                
                pd.DataFrame({
                    "Feature_Index": idx,
                    "Sensitivity": sens[idx]
                }).to_csv(
                    os.path.join(OUT_DIR, f"top_{TOP_K_GENES}_{m}_genes.csv"),
                    index=False
                )

    print("\n========================================================")
    print("ALL CANCERS PROCESSED SUCCESSFULLY. RESULTS SAVED.")
    print("========================================================")

if __name__ == "__main__":
    main()
