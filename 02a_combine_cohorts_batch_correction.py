import os
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS
# ============================================================
BASE_DIR    = r"D:\MultiModal_Classification"
PREPROC_DIR = os.path.join(BASE_DIR, "output", "01_preprocessing")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output", "02_batch_correction_deg_pca")
os.makedirs(OUTPUT_DIR, exist_ok=True)

META_COLS = ["PATIENT_ID", "SUBTYPE", "AGE", "SEX", "TUMOR_STAGE", "OS_STATUS", "OS_MONTHS"]

# ============================================================
# STEP 1 — Load the three cohorts, restrict to common genes
# ============================================================
print("[1/3] Loading per-cohort matrices and identifying common genes...")

df2012 = pd.read_csv(os.path.join(PREPROC_DIR, "tcga2012", "merged_df_2012.csv"), index_col=0)
df2016 = pd.read_csv(os.path.join(PREPROC_DIR, "metabric2016", "merged_df_2016.csv"), index_col=0)
df2018 = pd.read_csv(os.path.join(PREPROC_DIR, "tcga2018", "merged_df_2018.csv"), index_col=0)

common_genes = list(set(df2012.columns) & set(df2016.columns) & set(df2018.columns))
common_genes = [g for g in common_genes if g not in META_COLS]
print(f"    Common genes across all 3 cohorts: {len(common_genes)}")

df2012 = df2012[META_COLS + common_genes].copy()
df2016 = df2016[META_COLS + common_genes].copy()
df2018 = df2018[META_COLS + common_genes].copy()

df2012["BATCH"] = "Batch_2012"
df2016["BATCH"] = "Batch_2016"
df2018["BATCH"] = "Batch_2018"

combined_df = pd.concat([df2012, df2016, df2018], axis=0, ignore_index=True)
print(f"    Combined dataset (before correction): {combined_df.shape}")

combined_df.to_csv(os.path.join(OUTPUT_DIR, "combined_before_batch_correction.csv"), index=False)

# ============================================================
# STEP 2 — ComBat batch correction
# ============================================================
print("[2/3] Running ComBat batch correction (scanpy.pp.combat)...")

gene_data = combined_df[common_genes].values
batch_labels = combined_df["BATCH"].values

adata = sc.AnnData(X=gene_data)
adata.obs["batch"] = batch_labels
sc.pp.combat(adata, key="batch")

combat_corrected = pd.DataFrame(adata.X, columns=common_genes)
corrected_df = pd.concat([combined_df[META_COLS + ["BATCH"]], combat_corrected], axis=1)

corrected_df.to_csv(os.path.join(OUTPUT_DIR, "combined_after_batch_correction.csv"), index=False)
print(f"    Batch-corrected matrix saved: {corrected_df.shape}")

# ============================================================
# STEP 3 — Before/after PCA visualization (Figure 1)
# ============================================================
print("[3/3] Generating before/after batch-correction PCA plots...")

def plot_pca(data, batches, title, save_path):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame({"PC1": pca_result[:, 0], "PC2": pca_result[:, 1], "Batch": batches})
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Batch", palette="Set2", s=60, alpha=0.8)
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    plt.legend(title="Batch", loc="best", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_pca(combined_df[common_genes].values, batch_labels, "PCA Before Batch Correction",
          os.path.join(OUTPUT_DIR, "PCA_Before_Batch_Correction.png"))
plot_pca(corrected_df[common_genes].values, batch_labels, "PCA After Batch Correction",
          os.path.join(OUTPUT_DIR, "PCA_After_Batch_Correction.png"))

print(f"\nDone. Cohort integration & batch correction complete.")
print(f"Final pooled+corrected matrix: {corrected_df.shape[0]} samples x {len(common_genes)} genes")
print(f"Subtype distribution:\n{corrected_df['SUBTYPE'].value_counts().sort_index()}")
