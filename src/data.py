from __future__ import annotations

import os
import re

import numpy as np

try:
    from .config import BASE_DIR, OMICS
except ImportError:
    from config import BASE_DIR, OMICS


def load_cancer_dataset(cancer_name: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data_dir = os.path.join(BASE_DIR, cancer_name)
    omics_data = {
        "mRNA": np.load(os.path.join(data_dir, "mRNA_processed.npy")),
        "miRNA": np.load(os.path.join(data_dir, "miRNA_processed.npy")),
        "CNV": np.load(os.path.join(data_dir, "CNV_processed.npy")),
        "Methy": np.load(os.path.join(data_dir, "Methy_processed.npy")),
    }
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    return omics_data, labels


def load_feature_names(cancer_name: str, omics_data: dict[str, np.ndarray]) -> dict[str, list[str]]:
    data_dir = os.path.join(BASE_DIR, cancer_name)
    feature_names: dict[str, list[str]] = {}

    for omic in OMICS:
        num_features = omics_data[omic].shape[1]
        json_path = os.path.join(data_dir, f"{omic}_features.json")

        extracted_names: list[str] = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        extracted_names.append(match.group(1))

        if len(extracted_names) >= num_features:
            feature_names[omic] = extracted_names[:num_features]
        elif extracted_names:
            padding = [f"{omic}_Feat_{idx}" for idx in range(len(extracted_names), num_features)]
            feature_names[omic] = extracted_names + padding
        else:
            feature_names[omic] = [f"{omic}_Feat_{idx}" for idx in range(num_features)]

    return feature_names
