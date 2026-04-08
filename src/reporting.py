from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

try:
    from .config import OMICS, RESULTS_ROOT, TOP_K_GENES
except ImportError:
    from config import OMICS, RESULTS_ROOT, TOP_K_GENES


def create_accumulators() -> dict:
    return {
        "y_true": [],
        "y_pred": [],
        "gates": defaultdict(list),
        "sensitivity": {omic: None for omic in OMICS},
        "ablation": {
            "Base_MLP": {"prec": [], "rec": [], "f1": []},
            "SVM": {"prec": [], "rec": [], "f1": []},
            "XGBoost": {"prec": [], "rec": [], "f1": []},
            "Deeper_MLP": {"prec": [], "rec": [], "f1": []},
        },
    }


def generate_aggregated_plots(
    cancer_name: str,
    accumulators: dict,
    n_folds: int,
    feature_names: dict[str, list[str]],
) -> None:
    save_dir = os.path.join(RESULTS_ROOT, cancer_name)
    os.makedirs(save_dir, exist_ok=True)

    detailed_ablation_records = []
    for classifier_name, metrics in accumulators["ablation"].items():
        row = {"Classifier": classifier_name}
        for fold_idx in range(n_folds):
            row[f"Fold_{fold_idx}_Prec"] = metrics["prec"][fold_idx]
            row[f"Fold_{fold_idx}_Rec"] = metrics["rec"][fold_idx]
            row[f"Fold_{fold_idx}_F1"] = metrics["f1"][fold_idx]
        row["Mean_Prec"] = np.mean(metrics["prec"])
        row["Mean_Rec"] = np.mean(metrics["rec"])
        row["Mean_F1"] = np.mean(metrics["f1"])
        detailed_ablation_records.append(row)

    pd.DataFrame(detailed_ablation_records).to_csv(
        os.path.join(save_dir, "detailed_ablation_results.csv"),
        index=False,
    )

    gate_means = {omic: np.mean(values) for omic, values in accumulators["gates"].items()}
    gate_stds = {omic: np.std(values) for omic, values in accumulators["gates"].items()}

    plt.figure(figsize=(6, 5))
    x_vals = list(gate_means.keys())
    y_vals = list(gate_means.values())
    y_err = list(gate_stds.values())
    sns.barplot(x=x_vals, y=y_vals, hue=x_vals, palette="viridis", legend=False)
    plt.errorbar(x=range(len(x_vals)), y=y_vals, yerr=y_err, fmt="none", c="black", capsize=5)
    plt.title(f"{cancer_name}: Aggregated Gate Importance", fontweight="bold")
    plt.ylabel("Mean Gate Activation")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "aggregated_gate_importance.png"))
    plt.close()

    for omic in OMICS:
        if accumulators["sensitivity"][omic] is None:
            continue

        avg_sensitivity = accumulators["sensitivity"][omic] / n_folds
        top_indices = np.argsort(avg_sensitivity)[::-1][:TOP_K_GENES]
        top_scores = avg_sensitivity[top_indices]
        top_names = [feature_names[omic][idx] for idx in top_indices]

        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        sns.barplot(x=top_scores, y=top_names, hue=top_names, palette="magma", legend=False, orient="h", ax=ax)
        ax.set_title(f"{cancer_name} ({omic}): Top 20 Features", fontweight="bold")
        ax.set_xlabel("Average Sensitivity")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)

        # Force scientific notation for very small sensitivity values so miRNA plots remain readable.
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"aggregated_top20_{omic}.png"))
        plt.close(fig)

        pd.DataFrame(
            {
                "Feature_Name": top_names,
                "Feature_Index": top_indices,
                "Avg_Sensitivity": top_scores,
            }
        ).to_csv(os.path.join(save_dir, f"aggregated_top20_{omic}.csv"), index=False)
