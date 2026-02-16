# OMNIGATE: Omics-Integrated Gating for Explainable Multi-Cancer Subtype Classification 🧬🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)


#### OMNIGATE is a deep learning framework designed for robust multi-modal cancer subtype classification. Unlike traditional fusion methods that simply concatenate features, OMNIGATE utilizes a dynamic context gating mechanism that learns to weigh the importance of specific omics layers (mRNA, miRNA, CNV, Methylation) on a per-sample basis.

This repository contains the official implementation of the paper, including the preprocessing logic, model architecture, and the explainability pipeline that generates publication-ready visualizations.
##  Key Contributions

1. **Dynamic Context Gating**: The model does not treat all omics equally. It constructs a "Global Context" vector to dynamically suppress noise and amplify signal from relevant modalities (e.g., prioritizing Methylation over mRNA for specific subtypes).

2. **Multi-Objective Loss Landscape**: The training optimizes a compound loss function:

3. **Focal Loss**: To handle extreme class imbalance in cancer subtypes.

4. **Orthogonality & Alignment Loss**: Ensures latent representations are distinct yet semantically aligned.

5. **Sparsity & Entropy Regularization**: Forces the gates to make decisive choices rather than averaging inputs.

6. **Built-in Explainability**: The pipeline automatically extracts:

7. **Top-K Biomarkers**: Gradient-based sensitivity analysis identifies the top 20 genes driving predictions.

8. **Modality Importance**: Visualizes which omic layer contributes most to the decision.

Robust Reproducibility: Uses Stratified K-Fold Cross-Validation with fixed random seeds (42) and aggregated reporting to ensure results are statistically significant, not just a "lucky fold."


## Installation 

Clone the repository:

```bash
git clone https://github.com/yourusername/OMNIGATE.git
cd OMNIGATE
```
Install dependencies:
It is recommended to use a virtual environment (Conda or venv)
```bash
pip install -r requirements.txt
```
Usage
To run the full pipeline (Training, Validation, and Explanation Generation):
```bash
cd src
python main.py
```




## 🏆 Benchmark Comparison

We compared our **Gated Multi-Modal Framework** against the standard **MLOmics Baseline**. Our model successfully outperformed the baseline, achieving superior precision on key MLOmics datasets.

| Cancer Type | Dataset | MLOmics Best Baseline(Precision)| **OMNIGATE**(ours) | 
| :--- | :--- | :--- | :--- |
| **Breast Carcinoma** | GS-BRCA | 87.00% | **88.35%** | 
| **Brain Glioma** | GS-LGG | 94.00% | **98.47%** | 
| **Adenocarcinoma** | GS-OV | **95%**| 89% | 
| **Glioblastoma Multiforme** | GS-GBM | 95.00% | 84%|
| **Colon Adenocarcinoma** | GS-COAD| 93.00% | **94.36**|

![Classify Architecture](https://github.com/Rklearns/IPD-MultiOmics_Research/blob/main/omnigate_pipeline.png)
