# Gated Multi-Modal Fusion for Pan-Cancer Subtyping 🧬🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A class-aware, interpretable deep learning framework for integrating multi-omics data (mRNA, miRNA, CNV, Methylation) to classify cancer subtypes. This project introduces a **Context-Aware Gating Mechanism** to dynamically weight modalities and a novel explainability suite (**FAEC** & **CORI**) to validate model stability.

---
![Alt text](https://github.com/Rklearns/IPD-MultiOmics_Research/blob/main/Explainability.jpg)


##  Key Innovations

1.  **Context-Aware Gating:** Instead of static averaging, our **GateNet** learns to assign a "Trust Score" (0-1) to each omic modality specifically for each patient.
2.  **Imbalance-Resistant Training:** Utilizes **Focal Loss** combined with **Alignment** and **Entropy Regularization** to handle rare cancer subtypes effectively.
3.  **Robust Explainability:**
    *   **Gate Analysis:** Identifies which omics technology is most valuable.
    *   **FAEC (Fold-Aware Effect Consistency):** Validates the stability of feature importance across cross-validation folds.
    *   **CORI (Cross-Omics Redundancy Index):** Quantifies redundancy between data types to guide cost-effective clinical testing.
    *   **Sensitivity Analysis:** Discovers Top-20 driver genes using gradient-based backpropagation.

---

## 🏆 Benchmark Comparison

We compared our **Gated Multi-Modal Framework** against the standard **MLOmics Baseline**. Our model successfully outperformed the baseline, achieving superior precision on key MLOmics datasets.

| Cancer Type | Dataset | MLOmics Baseline (Precision) | **Ours (Gated Fusion)** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Breast Carcinoma** | GS-BRCA | 87.00% | **87.52%** | +0.52% |
| **Brain Glioma** | GS-LGG | 94.00% | **97.78%** |  **+3.78%** |

> **Key Observation:** The significant **3.78% boost** in Low-Grade Glioma (LGG) classification demonstrates the effectiveness of our **Gating Mechanism**. By dynamically up-weighting high-quality molecular signals and suppressing noise, our model captures subtle subtype differences that standard fusion methods often miss.

![Classify Architecture](https://github.com/Rklearns/IPD-MultiOmics_Research/blob/main/Classify.jpg)
