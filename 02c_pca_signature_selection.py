import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# ============================================================
# PATHS & PARAMETERS
# ============================================================
BASE_DIR   = r"D:\MultiModal_Classification"
STAGE_DIR  = os.path.join(BASE_DIR, "output", "02_batch_correction_deg_pca")
DEG_MATRIX_PATH = os.path.join(STAGE_DIR, "SUBTYPE_DEG_expression_matrix.csv")
BATCH_CORRECTED_PATH = os.path.join(STAGE_DIR, "combined_after_batch_correction.csv")
RESULTS_DIR = os.path.join(STAGE_DIR, "PCA_signature")

VARIANCE_THRESHOLD = 0.90     
LABEL_MAPPING = {0: "Normal", 1: "Luminal A", 2: "Luminal B", 3: "HER2", 4: "Basal"}
TOP_N_LOADING_GENES = 50      # genes considered per relevant PC before top-10 signature cut
COLORS = {"Normal": "blue", "Luminal A": "green", "Luminal B": "orange", "HER2": "red", "Basal": "purple"}


# ============================================================
# STEP 1 — Load, standardize, PCA
# ============================================================
def load_and_standardize(data_path):
    data = pd.read_csv(data_path)
    sample_ids = data["PATIENT_ID"]
    encoded_labels = data["SUBTYPE"]
    labels = encoded_labels.map(LABEL_MAPPING)
    gene_columns = [c for c in data.columns if c not in ["PATIENT_ID", "SUBTYPE"]]
    expression_matrix = data[gene_columns]

    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_matrix)
    print(f"Loaded {expression_matrix.shape[0]} samples x {expression_matrix.shape[1]} genes")
    print(f"Subtype distribution:\n{labels.value_counts().sort_index()}")
    return sample_ids, encoded_labels, labels, gene_columns, expression_scaled


def run_pca(expression_scaled, variance_threshold):
    pca_full = PCA()
    pca_full.fit(expression_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"PCs needed for {variance_threshold*100:.0f}% variance: {n_components}")

    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(expression_scaled)
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    return pca, pca_components, n_components, variance_explained, cumulative_variance


# ============================================================
# STEP 2 — Visualizations (Figure 3)
# ============================================================
def make_dirs(results_dir):
    for sub in ["plots", "csv_files", "biomarkers", "validation_sets", "reports", "configuration"]:
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)


def plot_variance(variance_explained, cumulative_variance, n_components, variance_threshold, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    pcs = range(1, len(variance_explained) + 1)
    ax1.bar(pcs, variance_explained * 100, alpha=0.7, color="skyblue")
    ax1.set_xlabel("Principal Component"); ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Individual Variance Explained by PCs"); ax1.grid(True, alpha=0.3)

    ax2.plot(pcs, cumulative_variance * 100, "b-", marker="o", linewidth=2, markersize=4)
    ax2.axhline(y=variance_threshold * 100, color="r", linestyle="--", label=f"{variance_threshold*100:.0f}% Threshold")
    ax2.axvline(x=n_components, color="g", linestyle="--", label=f"PC{n_components}")
    ax2.set_xlabel("Number of PCs"); ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance Explained"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "variance_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_2d_pcs(pca_df, labels, variance_explained, results_dir):
    for pc_x, pc_y in [(0, 1), (0, 2), (1, 2)]:
        fig, ax = plt.subplots(figsize=(12, 8))
        for label in labels.unique():
            mask = pca_df["LABEL"] == label
            ax.scatter(pca_df.loc[mask, f"PC{pc_x+1}"], pca_df.loc[mask, f"PC{pc_y+1}"],
                       c=COLORS.get(label, "gray"), label=label, alpha=0.7, s=50)
        ax.set_xlabel(f"PC{pc_x+1} ({variance_explained[pc_x]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y+1} ({variance_explained[pc_y]*100:.1f}%)")
        ax.set_title(f"PCA: PC{pc_x+1} vs PC{pc_y+1} - Subtype Separation")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "plots", f"pc{pc_x+1}_pc{pc_y+1}_subtype.png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_3d(pca_df, labels, variance_explained, n_components, results_dir):
    if n_components < 3:
        return
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    for label in labels.unique():
        mask = pca_df["LABEL"] == label
        ax.scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"], pca_df.loc[mask, "PC3"],
                   c=COLORS.get(label, "gray"), label=label, alpha=0.7, s=50)
    ax.set_xlabel(f"PC1 ({variance_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance_explained[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({variance_explained[2]*100:.1f}%)")
    ax.set_title("3D PCA - Subtype Separation")
    ax.legend()
    plt.savefig(os.path.join(results_dir, "plots", "pca_3d_subtype.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# STEP 3 — Biomarker / signature extraction (Table 3 / ST columns)
# ============================================================
def identify_label_pcs(pca_df, labels, n_components):
    """For each subtype, find the 3 PCs whose mean score differs most from the other subtypes."""
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    label_means = pca_df.groupby("LABEL")[pc_columns].mean()
    label_pcs = {}
    for label in labels.unique():
        label_mean = label_means.loc[label]
        other_means = label_means.drop(label)
        deviations = np.abs(label_mean - other_means.mean())
        label_pcs[label] = deviations.nlargest(3).index.tolist()
        print(f"  {label}: most differentiating PCs = {label_pcs[label]}")
    return label_pcs


def extract_label_biomarkers(loadings_df, pca_df, label, relevant_pcs, top_n_genes):
    """Top-loading genes (direction-aware) for one subtype, across its most differentiating PCs."""
    biomarkers = []
    for pc in relevant_pcs:
        label_mean = pca_df[pca_df["LABEL"] == label][pc].mean()
        overall_mean = pca_df[pc].mean()
        direction = 1 if label_mean > overall_mean else -1
        pc_biomarkers = (loadings_df.nlargest(top_n_genes, pc)[[pc]] if direction > 0
                          else loadings_df.nsmallest(top_n_genes, pc)[[pc]])
        pc_biomarkers["PC_Contribution"] = pc
        pc_biomarkers["Direction"] = "Overexpressed" if direction > 0 else "Underexpressed"
        pc_biomarkers["Label"] = label
        biomarkers.append(pc_biomarkers)
    if not biomarkers:
        return pd.DataFrame()
    combined = pd.concat(biomarkers)
    return combined[~combined.index.duplicated(keep="first")].head(top_n_genes)


def save_minimal_signature(biomarkers_df, label, results_dir):
    """Top-10-gene signature per subtype (feeds the final 50-gene panel)."""
    if biomarkers_df.empty:
        return
    safe_label = label.replace(" ", "_").replace("/", "_")
    top10 = biomarkers_df.head(10)
    loading_col = top10.columns[0]
    signature_df = pd.DataFrame({
        "Gene": top10.index,
        "Loading_Value": top10[loading_col].values,
        "Direction": top10["Direction"].values,
        "PC_Contribution": top10["PC_Contribution"].values,
        "Rank": range(1, len(top10) + 1),
    })
    signature_df.to_csv(os.path.join(results_dir, "validation_sets", f"{safe_label.lower()}_10_gene_signature.csv"), index=False)


# ============================================================
# STEP 4 — Combine per-subtype signatures -> final 50-gene panel + matrix
# ============================================================
def combine_top_signatures(validation_sets_dir, output_file):
    all_files = glob.glob(os.path.join(validation_sets_dir, "*_signature.csv"))
    combined = pd.concat([pd.read_csv(f)[["Gene"]] for f in all_files], axis=0)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    print(f"Deduplicated subtype signature panel: {len(combined)} genes -> {output_file}")
    return combined


def build_final_signature_matrix(deg_list_path, expression_matrix_path, output_path, gene_col="Gene"):
    deg_genes = pd.read_csv(deg_list_path)[gene_col].astype(str).tolist()
    expr_df = pd.read_csv(expression_matrix_path, index_col=0)
    meta_cols = ["PATIENT_ID", "AGE", "SEX", "OS_STATUS", "OS_MONTHS", "SUBTYPE", "TUMOR_STAGE"]
    existing_meta = [c for c in meta_cols if c in expr_df.columns]
    meta_df = expr_df[existing_meta]
    common_genes = [g for g in deg_genes if g in expr_df.columns]
    final_matrix = pd.concat([meta_df, expr_df[common_genes]], axis=1)
    final_matrix.to_csv(output_path, index=True)
    print(f"Final signature matrix saved: {output_path}  (shape={final_matrix.shape}, "
          f"{len(common_genes)}/{len(deg_genes)} genes found)")
    return final_matrix


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    make_dirs(RESULTS_DIR)

    print("=" * 70)
    print("STEP 1: Load 426-gene DEG matrix, standardize, run PCA")
    print("=" * 70)
    sample_ids, encoded_labels, labels, gene_columns, expression_scaled = load_and_standardize(DEG_MATRIX_PATH)
    pca, pca_components, n_components, variance_explained, cumulative_variance = run_pca(expression_scaled, VARIANCE_THRESHOLD)

    pca_df = pd.DataFrame(pca_components, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df["PATIENT_ID"] = sample_ids.values
    pca_df["ENCODED_LABEL"] = encoded_labels.values
    pca_df["LABEL"] = labels.values

    print("\n" + "=" * 70)
    print("STEP 2: Generate PCA visualizations (Figure 3)")
    print("=" * 70)
    plot_variance(variance_explained, cumulative_variance, n_components, VARIANCE_THRESHOLD, RESULTS_DIR)
    plot_2d_pcs(pca_df, labels, variance_explained, RESULTS_DIR)
    plot_3d(pca_df, labels, variance_explained, n_components, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("STEP 3: Extract subtype-specific biomarkers and top-10 signatures")
    print("=" * 70)
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=gene_columns)
    label_pcs = identify_label_pcs(pca_df, labels, n_components)

    for label, relevant_pcs in label_pcs.items():
        biomarkers_df = extract_label_biomarkers(loadings_df, pca_df, label, relevant_pcs, TOP_N_LOADING_GENES)
        safe_label = label.replace(" ", "_").replace("/", "_")
        biomarkers_df.reset_index().rename(columns={"index": "Gene"}).to_csv(
            os.path.join(RESULTS_DIR, "biomarkers", f"biomarkers_{safe_label.lower()}.csv"), index=False)
        save_minimal_signature(biomarkers_df, label, RESULTS_DIR)
        print(f"  {label}: {len(biomarkers_df)} candidate biomarkers -> top-10 signature saved")

    variance_summary = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(variance_explained))],
        "Variance_Explained": variance_explained, "Cumulative_Variance": cumulative_variance,
    })
    variance_summary.to_csv(os.path.join(RESULTS_DIR, "csv_files", "variance_summary.csv"), index=False)
    pca_df.to_csv(os.path.join(RESULTS_DIR, "csv_files", "sample_pca_coordinates.csv"), index=False)
    loadings_df.reset_index().rename(columns={"index": "Gene"}).to_csv(
        os.path.join(RESULTS_DIR, "csv_files", "gene_loadings_all_pcs.csv"), index=False)

    config = {
        "data_path": DEG_MATRIX_PATH, "label_column": "SUBTYPE", "label_mapping": LABEL_MAPPING,
        "variance_threshold": VARIANCE_THRESHOLD, "n_samples": int(len(sample_ids)),
        "n_genes": int(len(gene_columns)), "n_components": int(n_components),
        "timestamp": str(pd.Timestamp.now()),
    }
    with open(os.path.join(RESULTS_DIR, "configuration", "pca_parameters.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("STEP 4: Combine subtype signatures -> final ~50-gene panel + matrix")
    print("=" * 70)
    dedup_path = os.path.join(STAGE_DIR, "final_combined_top50_PCA_DEGs_subtype.csv")
    combine_top_signatures(os.path.join(RESULTS_DIR, "validation_sets"), dedup_path)

    build_final_signature_matrix(
        deg_list_path=dedup_path,
        expression_matrix_path=BATCH_CORRECTED_PATH,
        output_path=os.path.join(STAGE_DIR, "SUBTYPE_PCA_DEG_expression_matrix.csv"),
        gene_col="Gene",
    )

    print("\nDone. This SUBTYPE_PCA_DEG_expression_matrix.csv is the FINAL signature matrix")
    print("used as input by 03a/03b/03c (classifiers), 04 (prognosis), and 05a/05b (SHAP).")
