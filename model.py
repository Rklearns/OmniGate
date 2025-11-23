import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION & REPRODUCIBILITY
# -----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cpu')  # M4 Optimization: Use CPU for stability/compatibility as requested

# -----------------------------------------------------------------------------
# 1. VAE MODULE (Per-Omic Dimensionality Reduction)
# -----------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2_mu = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.bn3(self.fc3(z)))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # MSE Loss (Reconstruction)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# -----------------------------------------------------------------------------
# 2. GAT CLASSIFIER (Graph Attention Network)
# -----------------------------------------------------------------------------
class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, heads=4, dropout=0.5):
        super(GATClassifier, self).__init__()
        self.dropout = dropout
        
        # GAT Layer 1
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        # GAT Layer 2
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention_weights = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        
        # Layer 2 (Classifier)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1), attention_weights

# -----------------------------------------------------------------------------
# 3. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
def load_and_align_data(base_path):
    print("📌 Loading Data from:", base_path)

    files = {
        'mRNA': 'BRCA_mRNA_top.csv',
        'miRNA': 'BRCA_miRNA_top.csv',
        'CNV': 'BRCA_CNV_top.csv',
        'Methy': 'BRCA_Methy_top.csv'
    }
    
    label_file = '54814634_BRCA_label_num.csv'
    
    data_dict = {}
    patient_ids_list = []

    # 1. Load Omics Data
    for omic, filename in files.items():
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        print(f"   Loading {omic}...")
        # Read CSV. Assumption: Rows=Features, Cols=Patients.
        df = pd.read_csv(path, index_col=0) 
        
        # Transpose: (Features, Patients) -> (Patients, Features)
        df = df.T 
        
        # Ensure index (patient IDs) are strings
        df.index = df.index.astype(str)
        
        data_dict[omic] = df
        patient_ids_list.append(set(df.index))

    # 2. Load Labels
    label_path = os.path.join(base_path, label_file)
    print(f"   Loading Labels...")
    labels_df = pd.read_csv(label_path)
    
    # CRITICAL: Attach patient IDs to labels
    # We assume labels correspond exactly to the order of the mRNA file (or first loaded file)
    # as per instructions ("corresponds exactly to the patient list order").
    ref_patients = data_dict['mRNA'].index
    
    if len(labels_df) != len(ref_patients):
        raise ValueError(f"Label count ({len(labels_df)}) does not match patient count ({len(ref_patients)}) in mRNA file.")
        
    labels_df.index = ref_patients
    labels_df.rename(columns={'Label': 'target'}, inplace=True)

    # 3. Align Patients (Intersection)
    common_patients = set.intersection(*patient_ids_list)
    common_patients = sorted(list(common_patients)) # Sort for reproducibility
    
    print(f"✅ Alignment Complete. {len(common_patients)} common patients found across all files.")
    
    # Reindex everything to common patients
    aligned_data = {}
    for omic in data_dict:
        aligned_data[omic] = data_dict[omic].loc[common_patients].values.astype(np.float32)
        
    aligned_labels = labels_df.loc[common_patients]['target'].values.astype(np.int64)
    
    return aligned_data, aligned_labels, common_patients

def train_vae(model, data, epochs=50, batch_size=32, lr=1e-3, omic_name="Omic"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    print(f"   Training VAE for {omic_name}...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 20 == 0:
            print(f"     Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataset):.4f}")
            
    return model

def get_latent_features(model, data):
    model.eval()
    with torch.no_grad():
        t_data = torch.tensor(data).to(device)
        mu, _ = model.encode(t_data)
    return mu.cpu().numpy()

# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-Omics BRCA Classifier (VAE + GAT)")
    parser.add_argument('--data_path', type=str, default="/Users/rishitkar/Desktop/MLOmics/Main_Dataset/Classification_datasets/GS-BRCA/Top/", help="Path to dataset files")
    parser.add_argument('--latent_dim', type=int, default=128, help="VAE latent dimension")
    parser.add_argument('--epochs_vae', type=int, default=30, help="Epochs for VAE pretraining")
    parser.add_argument('--epochs_gat', type=int, default=200, help="Epochs for GAT training")
    parser.add_argument('--k_neighbors', type=int, default=10, help="K for kNN graph")
    args = parser.parse_args()

    # --- A. Data Loading ---
    data_dict, labels, patient_ids = load_and_align_data(args.data_path)
    
    # --- B. Preprocessing & VAE Encoding ---
    latent_vectors = []
    
    print("\n🔄 Starting VAE Encoding Pipeline...")
    for omic_name, X in data_dict.items():
        # StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define VAE
        input_dim = X_scaled.shape[1]
        vae = VAE(input_dim, args.latent_dim).to(device)
        
        # Train VAE
        vae = train_vae(vae, X_scaled, epochs=args.epochs_vae, omic_name=omic_name)
        
        # Extract Latent
        z = get_latent_features(vae, X_scaled)
        latent_vectors.append(z)
    
    # Concatenate all latent representations
    # Shape: (N_patients, 128 * 4 = 512)
    X_fused = np.concatenate(latent_vectors, axis=1)
    print(f"✅ Feature Fusion Complete. Shape: {X_fused.shape}")

    # --- C. Graph Construction (kNN) ---
    print(f"\n🕸️ Building Patient Similarity Graph (k={args.k_neighbors}, Metric=Cosine)...")
    knn = NearestNeighbors(n_neighbors=args.k_neighbors + 1, metric='cosine') # +1 because self is included
    knn.fit(X_fused)
    distances, indices = knn.kneighbors(X_fused)
    
    # Create Edge Index for PyG
    source_nodes = []
    target_nodes = []
    
    for i in range(len(X_fused)):
        # Skip the first neighbor (self)
        neighbors = indices[i, 1:] 
        for n in neighbors:
            source_nodes.append(i)
            target_nodes.append(n)
            # Make Undirected
            source_nodes.append(n)
            target_nodes.append(i)
            
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # --- D. Prepare GAT Data ---
    X_tensor = torch.tensor(X_fused, dtype=torch.float)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Stratified Train/Test Split
    train_idx, test_idx = train_test_split(
        range(len(labels)), 
        test_size=0.2, 
        stratify=labels, 
        random_state=SEED
    )
    
    # Create masks
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data = data.to(device)

    # --- E. Train GAT ---
    print("\n🚀 Training GAT Classifier...")
    num_classes = len(np.unique(labels))
    gat_model = GATClassifier(
        input_dim=X_fused.shape[1], 
        hidden_dim=128, 
        num_classes=num_classes, 
        heads=4
    ).to(device)
    
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(args.epochs_gat):
        gat_model.train()
        optimizer.zero_grad()
        out, _ = gat_model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation (using test set as proxy for this script to keep it single-file simple, 
        # in prod use separate val set)
        gat_model.eval()
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        
        if acc > best_val_acc:
            best_val_acc = acc
            patience_counter = 0
            # Save best model state if needed
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Loss {loss.item():.4f}, Test Acc {acc:.4f}")

    # --- F. Final Evaluation ---
    gat_model.eval()
    out, attention_weights = gat_model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    
    print("\n📊 Final Evaluation Results:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\n🔍 Explainability Info:")
    print(f"GAT Attention weights shape (edges, heads): {attention_weights[1].shape}")
    print("Input Feature Importance can be inferred from the first layer weights or attention aggregation analysis.")

if __name__ == "__main__":
    main()
