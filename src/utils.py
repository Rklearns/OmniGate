import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from config import TOP_K_GENES, DIAGRAMS_DIR, RESULTS_DIR, OMICS

# ============================================================
# PUBLICATION QUALITY STYLE SETTINGS
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.unicode_minus'] = False

def save_paper_plot(fig, cancer_name, filename):
    """Saves plot to the dedicated diagrams folder in PDF (vector) and PNG"""
    pdf_path = os.path.join(DIAGRAMS_DIR, cancer_name)
    os.makedirs(pdf_path, exist_ok=True)
    fig.savefig(os.path.join(pdf_path, f"{filename}.pdf"), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(pdf_path, f"{filename}.png"), dpi=300, bbox_inches='tight')

def generate_aggregated_reports(cancer_name, accumulators):
    print(f"  > Generating Publication Diagrams for {cancer_name}...")
    
    # ----------------------------------------------------
    # 1. AGGREGATED CONFUSION MATRIX
    # ----------------------------------------------------
    cm = confusion_matrix(accumulators['y_true'], accumulators['y_pred'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14}, ax=ax)
    ax.set_title(f"Confusion Matrix: {cancer_name}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    save_paper_plot(fig, cancer_name, "Fig_Confusion_Matrix")
    plt.close()

    # ----------------------------------------------------
    # 2. GLOBAL GATE IMPORTANCE (Mean across all folds)
    # ----------------------------------------------------
    means = []
    stds = []
    labels = []
    
    for m in OMICS:
        fold_values = accumulators['gates'][m]
        means.append(np.mean(fold_values))
        stds.append(np.std(fold_values))
        labels.append(m)
        
    fig, ax = plt.subplots(figsize=(6, 5))
    x_pos = np.arange(len(labels))
    
    # Standard Bar Plot
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, ecolor='black', capsize=10, 
           color=sns.color_palette("viridis", len(labels)))
    ax.set_ylabel('Gate Activation Weight (0-1)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(f"Global Modality Importance: {cancer_name}", fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    save_paper_plot(fig, cancer_name, "Fig_Gate_Importance")
    plt.close()

    # ----------------------------------------------------
    # 3. ROBUST LATENT SPACE (t-SNE)
    # ----------------------------------------------------
    z_all = np.concatenate(accumulators['latent'], axis=0)
    y_all = np.array(accumulators['y_true'])
    
    if len(y_all) > 30:
        # Fixed random state for t-SNE reproducibility
        tsne = TSNE(n_components=2, perplexity=min(30, len(y_all)-1), random_state=42)
        z_tsne = tsne.fit_transform(z_all)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        # Hue assigned to y_all (classes)
        sns.scatterplot(x=z_tsne[:,0], y=z_tsne[:,1], hue=y_all, palette="tab10", s=80, alpha=0.8, ax=ax)
        ax.set_title(f"Latent Manifold ({cancer_name})", fontweight='bold')
        ax.legend(title="Class")
        sns.move_legend(ax, "upper right")
        
        save_paper_plot(fig, cancer_name, "Fig_Latent_TSNE")
        plt.close()

    # ----------------------------------------------------
    # 4. TOP 20 GENES (Averaged Gradient)
    # ----------------------------------------------------
    raw_results_dir = os.path.join(RESULTS_DIR, cancer_name)
    os.makedirs(raw_results_dir, exist_ok=True)

    for m in OMICS:
        if accumulators['sensitivity'][m] is not None:
            # Approx average sensitivity
            avg_sens = accumulators['sensitivity'][m] / 5.0 
            
            idx = np.argsort(avg_sens)[::-1][:TOP_K_GENES]
            top_genes_scores = avg_sens[idx]
            feature_names = [f"Feat {i}" for i in idx]
            
            # Save CSV
            df = pd.DataFrame({"Feature_Index": idx, "Importance_Score": top_genes_scores})
            df.to_csv(os.path.join(raw_results_dir, f"top20_{m}_genes.csv"), index=False)
            
            # Plot - FIXED WARNING HERE
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(
                x=top_genes_scores, 
                y=feature_names, 
                hue=feature_names,      # Assign y to hue
                palette="magma", 
                legend=False,           # Disable legend
                orient='h', 
                ax=ax
            )
            ax.set_title(f"Top 20 {m} Features ({cancer_name})", fontweight='bold')
            ax.set_xlabel("Mean Gradient Sensitivity")

            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1e}"))
            
            save_paper_plot(fig, cancer_name, f"Fig_Top20_{m}")
            plt.close()
