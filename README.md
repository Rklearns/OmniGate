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
The Code to get that type of data from the huge dataset is as follows:
```python
import shutil
import os
from google.colab import files


# Create a focused dataset with ONLY what you need
target_path = '/content/MLOmics_βVAE_Focused'
os.makedirs(target_path, exist_ok=True)

# Select ONLY your target 12 cancer types (not all 32!)
target_cancers = {
    'Classification_datasets': ['GS-BRCA', 'GS-COAD', 'GS-GBM', 'GS-LGG', 'GS-OV'],
    'Clustering_datasets': ['ACC', 'KIRC', 'LIHC', 'LUAD', 'LUSC', 'PRAD', 'THCA']
}

source_main = f"{ipd_path}/Main_Dataset"

print(" Creating focused dataset for β-VAE (12 cancer types only)...")

for dataset_type, cancers in target_cancers.items():
    for cancer in cancers:
        source_top = f"{source_main}/{dataset_type}/{cancer}/Top"
        
        if os.path.exists(source_top):
            target_dir = f"{target_path}/{dataset_type}/{cancer}/Top"
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy ONLY the essential files
            essential_files = []
            for file in os.listdir(source_top):
                if file.endswith('.csv') and any(x in file.lower() for x in 
                    ['mrna', 'mirna', 'cnv', 'methy', 'survival', 'label']):
                    essential_files.append(file)
                    shutil.copy2(f"{source_top}/{file}", f"{target_dir}/{file}")
            
            print(f" {cancer}: {len(essential_files)} files copied")

# Add README
shutil.copy2(f"{ipd_path}/README.md", f"{target_path}/README.md")

# Create FOCUSED zip (much smaller!)
print("\n Creating focused β-VAE dataset zip...")
shutil.make_archive('/content/MLOmics_βVAE_Selected', 'zip', target_path)

# Check the f
