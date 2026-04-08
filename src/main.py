from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from .config import DEFAULT_N_SPLITS, GS_TYPES, RESULTS_ROOT, SEED, configure_environment
    from .data import load_cancer_dataset, load_feature_names
    from .reporting import create_accumulators, generate_aggregated_plots
    from .training import train_and_collect
except ImportError:
    from config import DEFAULT_N_SPLITS, GS_TYPES, RESULTS_ROOT, SEED, configure_environment
    from data import load_cancer_dataset, load_feature_names
    from reporting import create_accumulators, generate_aggregated_plots
    from training import train_and_collect


def run_pipeline() -> None:
    configure_environment()
    global_ablation_summary = []

    for cancer_name in GS_TYPES:
        print(f"\n==================== {cancer_name} ====================")

        try:
            omics_data, labels = load_cancer_dataset(cancer_name)
        except FileNotFoundError:
            print(f"Skipping {cancer_name} (Files not found)")
            continue

        feature_names = load_feature_names(cancer_name, omics_data)
        accumulators = create_accumulators()

        min_class_size = np.min(np.unique(labels, return_counts=True)[1])
        n_splits = max(2, min_class_size) if min_class_size < DEFAULT_N_SPLITS else DEFAULT_N_SPLITS
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(omics_data["mRNA"], labels)):
            print(f"  > Fold {fold_idx}...")
            train_and_collect(omics_data, labels, train_idx, test_idx, cancer_name, accumulators)
            plt.close("all")

        print(f"\n{cancer_name} Classifier Ablation Results (Averaged across folds):")
        for classifier_name, metrics in accumulators["ablation"].items():
            mean_precision = np.mean(metrics["prec"])
            mean_recall = np.mean(metrics["rec"])
            mean_f1 = np.mean(metrics["f1"])
            print(
                f"  > {classifier_name:12s} | Prec: {mean_precision:.4f} | "
                f"Rec: {mean_recall:.4f} | F1: {mean_f1:.4f}"
            )
            global_ablation_summary.append(
                {
                    "Cancer": cancer_name,
                    "Classifier": classifier_name,
                    "Mean_Precision": mean_precision,
                    "Mean_Recall": mean_recall,
                    "Mean_F1": mean_f1,
                }
            )

        print(f"  > Generating aggregated plots and saving CSVs for {cancer_name}...")
        generate_aggregated_plots(cancer_name, accumulators, n_splits, feature_names)

    if global_ablation_summary:
        summary_df = pd.DataFrame(global_ablation_summary)
        summary_path = os.path.join(RESULTS_ROOT, "final_ablation_summary_all_cancers.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nALL PROCESSES COMPLETED. Master summary saved to: {summary_path}")
    else:
        print("\nALL PROCESSES COMPLETED.")


if __name__ == "__main__":
    run_pipeline()
