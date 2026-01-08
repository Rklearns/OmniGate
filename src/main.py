import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
from collections import defaultdict

# Import from our modules
from src.config import (
    DATA_ROOT, RESULTS_ROOT, GS_TYPES, OMICS, 
    N_SPLITS, TOP_K_GENES, SEED, setup_directories
)
from src.utils import train_on_split

def main():
    setup_directories()
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for cancer in GS_TYPES:
        print(f"\n==================== {cancer} ====================")

        DATA_DIR = f"{DATA_ROOT}/{cancer}"
        OUT_DIR = f"{RESULTS_ROOT}/{cancer}"
        os.makedirs(OUT_DIR, exist_ok=True)

        # --- LOAD DATA ---
        try:
            X = {m: np.load(f"{DATA_DIR}/{m}_processed.npy") for m in OMICS}
            y = np.load(f"{DATA_DIR}/labels.npy")
            feature_names = {
                m: json.load(open(f"{DATA_DIR}/{m}_features.json"))
                for m in OMICS
            }
        except FileNotFoundError:
            print(f"Skipping {cancer} (missing files)")
            continue

        X_full = np.concatenate([X[m] for m in OMICS], axis=1)

        # --- LOOP ---
        full_scores = []
        ablation_scores = {m: [] for m in OMICS}
        weight_importance_per_fold = []

        for tr, te in skf.split(X_full, y):
            prec, model = train_on_split(X_full, y, tr, te)
            full_scores.append(prec)

            # Weight-based importance
            W = model.fc1.weight.detach().cpu().numpy()
            weight_importance_per_fold.append(np.linalg.norm(W, axis=0))

            # Ablation
            offset = 0
            for m in OMICS:
                d = X[m].shape[1]
                X_drop = X_full.copy()
                X_drop[:, offset:offset+d] = 0.0
                p_drop, _ = train_on_split(X_drop, y, tr, te)
                ablation_scores[m].append(p_drop)
                offset += d

        # --- ANALYSIS (Feature Importance) ---
        mean_importance = np.mean(weight_importance_per_fold, axis=0)
        offset = 0
        gene_tables = {}

        for m in OMICS:
            d = X[m].shape[1]
            scores = mean_importance[offset:offset+d]
            genes = feature_names[m]
            
            # Safety check for lengths
            min_len = min(len(scores), len(genes))
            df = pd.DataFrame({
                "Gene": genes[:min_len],
                "Importance": scores[:min_len]
            }).sort_values("Importance", ascending=False)

            df.head(TOP_K_GENES).to_csv(f"{OUT_DIR}/top_{TOP_K_GENES}_{m}_genes.csv", index=False)
            gene_tables[m] = df
            offset += d

        # --- FAEC (Stability) ---
        faec_counter = defaultdict(int)
        for imp in weight_importance_per_fold:
            offset = 0
            for m in OMICS:
                d = X[m].shape[1]
                scores = imp[offset:offset+d]
                genes = feature_names[m]
                
                min_len = min(len(scores), len(genes))
                top_idx = np.argsort(scores[:min_len])[-TOP_K_GENES:]
                for idx in top_idx:
                    faec_counter[(m, genes[idx])] += 1
                offset += d

        pd.DataFrame([
            {"Omics": m, "Gene": g, "FAEC": c / N_SPLITS} for (m, g), c in faec_counter.items()
        ]).sort_values("FAEC", ascending=False).to_csv(f"{OUT_DIR}/faec_gene_consistency.csv", index=False)

        # --- CORI (Redundancy) ---
        cori_records = []
        for m1, m2 in combinations(OMICS, 2):
            s1 = gene_tables[m1]["Importance"].values
            s2 = gene_tables[m2]["Importance"].values
            min_len = min(len(s1), len(s2))
            corr = np.corrcoef(s1[:min_len], s2[:min_len])[0, 1]
            cori_records.append({"Omics_A": m1, "Omics_B": m2, "CORI": abs(corr)})
        
        pd.DataFrame(cori_records).to_csv(f"{OUT_DIR}/cori_table.csv", index=False)

        # --- SUMMARY ---
        summary = {"Mean_Precision": float(np.mean(full_scores)), "Std_Precision": float(np.std(full_scores))}
        pd.DataFrame([summary]).to_csv(f"{OUT_DIR}/performance_summary.csv", index=False)
        print(f"{cancer} PRECISION: {summary['Mean_Precision']:.4f} ± {summary['Std_Precision']:.4f}")

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
