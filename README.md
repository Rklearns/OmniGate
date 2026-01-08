# Gated Multi-Modal Fusion for Pan-Cancer Subtyping 🧬🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A class-aware, interpretable deep learning framework for integrating multi-omics data (mRNA, miRNA, CNV, Methylation) to classify cancer subtypes. This project introduces a **Context-Aware Gating Mechanism** to dynamically weight modalities and a novel explainability suite (**FAEC** & **CORI**) to validate model stability.

---

##  Key Innovations

1.  **Context-Aware Gating:** Instead of static averaging, our **GateNet** learns to assign a "Trust Score" (0-1) to each omic modality specifically for each patient.
2.  **Imbalance-Resistant Training:** Utilizes **Focal Loss** combined with **Alignment** and **Entropy Regularization** to handle rare cancer subtypes effectively.
3.  **Robust Explainability:**
    *   **Gate Analysis:** Identifies which omics technology is most valuable.
    *   **FAEC (Fold-Aware Effect Consistency):** Validates the stability of feature importance across cross-validation folds.
    *   **CORI (Cross-Omics Redundancy Index):** Quantifies redundancy between data types to guide cost-effective clinical testing.
    *   **Sensitivity Analysis:** Discovers Top-20 driver genes using gradient-based backpropagation.

---

