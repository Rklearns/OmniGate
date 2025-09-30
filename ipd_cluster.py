################################################################################
#
#                                --- STAGE 0 ---
#                           SETUP AND ENVIRONMENT
#
################################################################################


# --- Import libraries ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import shap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from google.colab import drive


try:
    drive.mount('/content/drive', force_remount=True)
    print("\nGoogle Drive mounted successfully.")
except Exception as e:
    print(f"\nERROR: Google Drive could not be mounted. {e}")

print("Environment setup complete.")


################################################################################
#
#                                --- STAGE 1 ---
#                  DATA LOADING AND PREPROCESSING (CLUSTERING COHORT)
#
################################################################################

print("\n--- STAGE 1: Loading and Preprocessing Clustering Cohort Data ---")


DATA_ROOT = '/content/drive/MyDrive/MLOmics_βVAE_Focused'
CLUSTERING_CANCERS = ['ACC','KIRC','LIHC','LUAD','LUSC','PRAD','THCA']

# --- [MODIFIED] Helper functions for data loading ---
def top_rows(df, k):
    """Selects top k rows with the highest variance."""
    return df.loc[df.var(axis=1).nlargest(k).index]

def load_cancer(cancer):
    """
    Loads all omics data for a single cancer type, preserving original sample IDs.
    """
    top_dir = f'{DATA_ROOT}/Clustering_datasets/{cancer}/Top'
    if not os.path.isdir(top_dir):
        print(f"  - Directory not found for {cancer}, skipping.")
        return None

    # Use one file to establish the primary list of sample IDs
    base_path = f"{top_dir}/{cancer}_mRNA_top.csv"
    if not os.path.isfile(base_path):
        print(f"  - Base mRNA file not found for {cancer}, skipping.")
        return None

    sample_ids = pd.read_csv(base_path, index_col=0).columns
    frames = []

    for omic in ['mRNA', 'miRNA', 'CNV', 'Methy']:
        csv_path = f"{top_dir}/{cancer}_{omic}_top.csv"
        if not os.path.isfile(csv_path): continue

        df = pd.read_csv(csv_path, index_col=0)
        k = 200 if omic == 'miRNA' else 1000
        df_top = top_rows(df, k).T # Transpose to get samples as rows

        # Align columns to ensure patient consistency before appending
        df_top = df_top.reindex(sample_ids).fillna(0)
        frames.append(df_top)

    if not frames: return None

    # Concatenate features horizontally, keeping the sample IDs as the index
    merged = pd.concat(frames, axis=1)
    merged.columns = [f'{cancer}_feat_{i}' for i in range(merged.shape[1])]

    # CRUCIAL: Prefix the original TCGA sample ID with the cancer type
    merged.index = f'{cancer}_' + merged.index.astype(str)

    print(f'  ✔ Loaded {cancer} | Shape: {merged.shape}')
    return merged.astype('float32')

# --- Load, concatenate, and process the data ---
clustering_dfs = {cancer: load_cancer(cancer) for cancer in CLUSTERING_CANCERS}
clustering_dfs = {k: v for k, v in clustering_dfs.items() if v is not None}
clustering_data = pd.concat(clustering_dfs.values(), axis=0)

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
imputed_clust = imputer.fit_transform(clustering_data)
scaled_clust_df = pd.DataFrame(scaler.fit_transform(imputed_clust), index=clustering_data.index, columns=clustering_data.columns)

print(f"\nClustering Cohort ready for analysis. Final shape: {scaled_clust_df.shape}")

# --- [NEW] Load, combine, and align ACTUAL survival data ---

print("\nLoading and Combining Real Survival Data...")
all_survival_dfs = []
for cancer in CLUSTERING_CANCERS:
    survival_path = f'{DATA_ROOT}/Clustering_datasets/{cancer}/Top/survival_{cancer}.csv'
    if os.path.isfile(survival_path):
        surv_df = pd.read_csv(survival_path)
        # Rename columns for lifelines compatibility
        surv_df = surv_df.rename(columns={
            'sample_name': 'Sample',
            'survival_times': 'Time',
            'event_observed': 'Event'
        })
        # Prefix sample IDs with the cancer type to match the main dataframe
        surv_df['Sample'] = f'{cancer}_' + surv_df['Sample'].astype(str)
        all_survival_dfs.append(surv_df)
        print(f"  ✔ Loaded survival data for {cancer}")
    else:
        print(f"  - WARNING: Survival file not found for {cancer} at {survival_path}")

# Combine all individual survival dataframes into one
if all_survival_dfs:
    combined_survival_df = pd.concat(all_survival_dfs, ignore_index=True).set_index('Sample')
    # CRITICAL FINAL STEP: Align survival data with the clustering data index
    clinical_data_clust = combined_survival_df.reindex(scaled_clust_df.index)

    # Check for any samples that didn't have a survival entry
    missing_survival_count = clinical_data_clust['Time'].isnull().sum()
    if missing_survival_count > 0:
        print(f"\nWARNING: {missing_survival_count} samples are missing survival data. They will be ignored in the survival plot.")

    print(f"\nReal clinical data loaded and aligned. Final shape: {clinical_data_clust.shape}")
    print("Preview of loaded clinical data:")
    print(clinical_data_clust.head())
else:
    print("\nFATAL ERROR: No survival data files were loaded. The program cannot continue with survival analysis.")
    # Create an empty df to prevent errors, though the plots will be empty
    clinical_data_clust = pd.DataFrame(index=scaled_clust_df.index, columns=['Time', 'Event'])


# --- Universal Parameters ---
n_components = 16
n_clusters = 4 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_clust_tensor = torch.tensor(scaled_clust_df.values, dtype=torch.float32).to(device)


################################################################################
#
#                                --- STAGE 2 ---
#                      COMPREHENSIVE CLUSTERING BENCHMARK
#
################################################################################

# This stage remains unchanged. It will run as before.
print(f"\n--- STAGE 2: Benchmarking Clustering Methods on Unlabeled Cohort (k={n_clusters}) ---")
embeddings = {}

# --- Define Models (Autoencoder, BetaVAE, DEC) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoded_dim=16):
        super().__init__(); self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, encoded_dim)); self.decoder = nn.Sequential(nn.Linear(encoded_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))
    def forward(self, x): return self.encoder(x)

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, beta=1.5):
        super().__init__(); self.beta = beta; self.encoder_fc1 = nn.Linear(input_dim, 128); self.encoder_fc2 = nn.Linear(128, 64); self.mu_layer = nn.Linear(64, latent_dim); self.logvar_layer = nn.Linear(64, latent_dim); self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))
    def encode(self, x): h = F.relu(self.encoder_fc1(x)); h = F.relu(self.encoder_fc2(h)); return self.mu_layer(h), self.logvar_layer(h)
    def reparameterize(self, mu, logvar): std = torch.exp(0.5*logvar); eps = torch.randn_like(std); return mu + eps*std
    def forward(self, x): mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); return self.decoder(z), mu, logvar

def beta_vae_loss(recon_x, x, mu, logvar, beta): return F.mse_loss(recon_x, x, reduction='sum') - 0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class DECAutoencoder(nn.Module):
    def __init__(self, input_dim=16, encoded_dim=10, n_clusters=4):
        super().__init__(); self.encoder = nn.Sequential(nn.Linear(input_dim, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, encoded_dim)); self.decoder = nn.Sequential(nn.Linear(encoded_dim, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, input_dim)); self.clustering_layer = nn.Parameter(torch.Tensor(n_clusters, encoded_dim)); torch.nn.init.xavier_normal_(self.clustering_layer.data)
    def forward(self, x): latent_z = self.encoder(x); q = 1.0 / (1.0 + torch.sum(torch.square(latent_z.unsqueeze(1) - self.clustering_layer), dim=2)); q = q.pow(1.0); q = (q.t() / torch.sum(q, dim=1)).t(); return self.decoder(latent_z), q, latent_z

# --- Generate Embeddings for All Models ---
print("Generating PCA embeddings..."); pca = PCA(n_components=n_components, random_state=42); embeddings['PCA'] = pca.fit_transform(scaled_clust_df.values)

print("Generating NMF embeddings..."); nmf_scaler = MinMaxScaler(); data_non_negative = nmf_scaler.fit_transform(scaled_clust_df.values); nmf = NMF(n_components=n_components, random_state=42, init='random', max_iter=400); embeddings['NMF'] = nmf.fit_transform(data_non_negative)

dataloader = DataLoader(TensorDataset(X_clust_tensor), batch_size=256, shuffle=True)

print("Training Regular Autoencoder..."); ae = Autoencoder(scaled_clust_df.shape[1]).to(device); optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

for epoch in range(50):
    for (batch,) in dataloader: encoded = ae(batch); decoded = ae.decoder(encoded); loss = F.mse_loss(decoded, batch); optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad(): embeddings['AE'] = ae(X_clust_tensor).cpu().numpy()
print("Training β-VAE..."); bvae = BetaVAE(scaled_clust_df.shape[1]).to(device); optimizer = torch.optim.Adam(bvae.parameters(), lr=1e-3)

for epoch in range(50):
    for (batch,) in dataloader: recon, mu, logvar = bvae(batch); loss = beta_vae_loss(recon, batch, mu, logvar, 1.5); optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad(): mu, _ = bvae.encode(X_clust_tensor); embeddings['bVAE'] = mu.cpu().numpy()
print("Training PCA-DEC..."); X_pca_tensor = torch.tensor(embeddings['PCA'], dtype=torch.float32).to(device)
dec_ae = DECAutoencoder(n_clusters=n_clusters).to(device); optimizer = torch.optim.Adam(dec_ae.parameters(), lr=1e-3)

for epoch in range(100):
    for (batch,) in DataLoader(TensorDataset(X_pca_tensor), batch_size=256, shuffle=True): recon, _, _ = dec_ae(batch); loss = F.mse_loss(recon, batch); optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad(): _, _, latent_features = dec_ae(X_pca_tensor); latent_features = latent_features.cpu().numpy()

kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(latent_features); dec_ae.clustering_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

def target_distribution(q): weight = q**2 / q.sum(0); return (weight.t() / weight.sum(1)).T
optimizer = torch.optim.Adam(dec_ae.parameters(), lr=1e-4)
for epoch in range(100):
    if epoch % 10 == 0:
        with torch.no_grad(): _, q_full, _ = dec_ae(X_pca_tensor); p_full = target_distribution(q_full)
    for (batch,) in DataLoader(TensorDataset(X_pca_tensor), batch_size=256, shuffle=True):
        recon_batch, q_batch, _ = dec_ae(batch); loss = F.kl_div(q_batch.log(), p_full[torch.randperm(p_full.size(0))[:batch.size(0)]], reduction='batchmean') + 0.1 * F.mse_loss(recon_batch, batch); optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad(): _, _, Z_dec_latent = dec_ae(X_pca_tensor); embeddings['PCA-DEC'] = Z_dec_latent.cpu().numpy()

# --- Run Benchmark and Create Results Table ---
benchmark_results = []
for name, embedding in embeddings.items():
    for cluster_method in ['KMeans', 'GMM']:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42) if cluster_method == 'KMeans' else GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(embedding)
        if len(np.unique(labels)) > 1:
            benchmark_results.append({
                'Method': f'{name} + {cluster_method}',
                'Silhouette Score (↑)': silhouette_score(embedding, labels),
                'Calinski-Harabasz (↑)': calinski_harabasz_score(embedding, labels),
                'Davies-Bouldin (↓)': davies_bouldin_score(embedding, labels)
            })

benchmark_df = pd.DataFrame(benchmark_results).set_index('Method')
pd.set_option('display.float_format', '{:.4f}'.format)
print("\n" + "="*80)
print(f"       CLUSTERING BENCHMARK RESULTS (CLUSTERING COHORT, k={n_clusters})")
print("="*80)
print(benchmark_df.sort_values(by='Silhouette Score (↑)', ascending=False))
print("="*80 + "\n")


################################################################################
#
#                                --- STAGE 3 ---
#                  POST-HOC ANALYSIS OF THE WINNING METHOD
#
################################################################################

print("\n--- STAGE 3: Validation and Interpretation of Winning Clusters ---")

# --- Identify the winning method and get its labels ---
winning_method_name = benchmark_df['Silhouette Score (↑)'].idxmax()
print(f"Winning Method Identified: {winning_method_name}")
winning_embedding_name, winning_cluster_algo = winning_method_name.split(' + ')
winning_embedding = embeddings[winning_embedding_name]
model = KMeans(n_clusters, n_init=10, random_state=42) if winning_cluster_algo == 'KMeans' else GaussianMixture(n_components=n_clusters, random_state=42)
winning_labels = model.fit_predict(winning_embedding)
clinical_data_clust['Subtype'] = winning_labels

# SUBTYPE COMPOSITION ---
print("\nGenerating Subtype Composition Plot...")
original_cancer_types = [idx.split('_')[0] for idx in scaled_clust_df.index]
composition_df = pd.DataFrame({'Original_Cancer_Type': original_cancer_types, 'Pan_Cancer_Subtype': winning_labels})
composition_table = pd.crosstab(composition_df['Pan_Cancer_Subtype'], composition_df['Original_Cancer_Type'])
composition_table_percent = composition_table.div(composition_table.sum(axis=0), axis=1) * 100
composition_table_percent.T.plot(kind='bar', stacked=True, figsize=(14, 8), cmap='viridis', width=0.8)
plt.title('Composition of Original Cancer Types within Discovered Subtypes', fontsize=18)
plt.xlabel('Original Cancer Type', fontsize=14); plt.ylabel('Percentage of Patients (%)', fontsize=14); plt.xticks(rotation=45)
plt.legend(title='Discovered Subtype', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.tight_layout(); plt.show()



print("\nGenerating Kaplan-Meier Survival Plot...")

# Only proceed if there is at least some valid data to plot after cleaning
if not clinical_data_clust.dropna(subset=['Time', 'Event']).empty:
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)

    for subtype_id in sorted(clinical_data_clust['Subtype'].unique()):
        # Select the subtype from the main clinical dataframe
        subset_with_nans = clinical_data_clust[clinical_data_clust['Subtype'] == subtype_id]

        #  Drop rows with missing Time or Event data ---
        subset = subset_with_nans.dropna(subset=['Time', 'Event'])

        if not subset.empty:
            # Fit the model on the cleaned data for this specific subtype
            kmf.fit(subset['Time'], event_observed=subset['Event'], label=f'Subtype {subtype_id} (n={len(subset)})')
            kmf.plot_survival_function(ax=ax)


    cleaned_clinical_data = clinical_data_clust.dropna(subset=['Time', 'Event'])
    if cleaned_clinical_data['Subtype'].nunique() > 1:
        p_value = multivariate_logrank_test(cleaned_clinical_data['Time'], cleaned_clinical_data['Subtype'], cleaned_clinical_data['Event']).p_value
        plt.title(f'Survival by Discovered Subtypes (Log-rank p-value: {p_value:.4f})', fontsize=18)
    else:
        plt.title('Survival by Discovered Subtypes', fontsize=18)

    plt.xlabel('Time (Days)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("Skipping survival plot: No valid survival data was available after cleaning.")



# --- ANALYSIS 3: BIOLOGICAL INTERPRETATION (SHAP) ---
print("\nGenerating SHAP Plot for Driver Gene Identification...")
clf_explain = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(scaled_clust_df.values, winning_labels)
explainer = shap.TreeExplainer(clf_explain)
shap_sample = scaled_clust_df.sample(min(200, scaled_clust_df.shape[0]), random_state=42)
shap_values = explainer.shap_values(shap_sample)
shap.summary_plot(shap_values, shap_sample, plot_type="bar", class_names=[f"Subtype {i}" for i in sorted(np.unique(winning_labels))], show=False)
plt.title('Top Molecular Drivers of Discovered Subtypes', fontsize=16); plt.tight_layout(); plt.show()


print("\n--- CLUSTERING COHORT ANALYSIS COMPLETE ---")
print("NEXT STEP: Take the top genes from the SHAP plot for each subtype and perform Gene Set Enrichment Analysis (GSEA).")

# TO be done apna is gsea method 
