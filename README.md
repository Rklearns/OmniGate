# IPD-MultiOmics_Research
For Our Research We only require, 12 Cancer Types since if we take all 32 Cancer Types->That can lead to data imbalance
We are taking top features of each omic layer of each Cancer Type
```bash
MLOmics_βVAE_Dataset/
├── Classification_datasets/                  # Cancers with known molecular subtype labels
│   ├── GS-BRCA/
│   │   └── Top/
│   │       ├── BRCA_mRNA_top.csv             # Top mRNA expression features (genes × samples)
│   │       ├── BRCA_miRNA_top.csv            # Top miRNA expression features
│   │       ├── BRCA_CNV_top.csv              # Top CNV (copy number variation) features
│   │       ├── BRCA_Methy_top.csv            # Top methylation features
│   │       └── BRCA_label_num.csv            # Molecular subtype labels (used for supervised validation, optional)
│   ├── GS-COAD/
│   ├── GS-GBM/
│   ├── GS-LGG/
│   └── GS-OV/
├── Clustering_datasets/                      # Cancers without pre-defined subtypes (unsupervised discovery)
│   ├── ACC/
│   │   └── Top/
│   │       ├── ACC_mRNA_top.csv
│   │       ├── ACC_miRNA_top.csv
│   │       ├── ACC_CNV_top.csv
│   │       ├── ACC_Methy_top.csv
│   │       └── survival_ACC.csv              # Survival/clinical outcome data for each patient
│   ├── KIRC/
│   ├── LIHC/
│   ├── LUAD/
│   ├── LUSC/
│   ├── PRAD/
│   ├── THCA/
│   └── others as selected
└── README.md                                 # This readme file describing dataset structure and usage
```            
To Access the modified Dataset please Access the dataset from this google drive link since original is of no use for us and is of almost 8GB
[https://drive.google.com/drive/folders/1Cu2MCzuEzo-ulUBe0DVJz2GFTfC_nNRS?usp=sharing](url)
