import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"D:\MultiModal_Classification"
DATA_FILE = os.path.join(BASE_DIR, "output", "02_batch_correction_deg_pca", "SUBTYPE_PCA_DEG_expression_matrix.csv")
MODEL_FILE = os.path.join(BASE_DIR, "output", "03_classification", "ml5", "XGB", "best_model_updated_95CI.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "05_shap")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NON_FEATURE_COLS = ["SUBTYPE", "PATIENT_ID", "AGE", "SEX", "OS_STATUS", "OS_MONTHS", "TUMOR_STAGE"]
CLASS_NAMES = ["Normal", "LuminalA", "LuminalB", "Her2", "Basal"]
TOP_N = 30


def main():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors="ignore")
    y = df["SUBTYPE"]
    feature_names = X.columns.tolist()
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features (genes).")

    model = joblib.load(MODEL_FILE)
    print(f"Model loaded: {len(model.classes_)} classes -> {model.classes_}")

    print("\nComputing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X)

    if isinstance(raw_shap, list):
        shap_values = raw_shap
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_values = ([raw_shap[:, :, i] for i in range(raw_shap.shape[2])]
                        if raw_shap.shape[2] == len(model.classes_)
                        else [raw_shap[i] for i in range(raw_shap.shape[0])])
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
        shap_values = [raw_shap]
    else:
        raise ValueError(f"Unexpected SHAP output type/shape: {type(raw_shap)}")

    n_classes = len(shap_values)
    assert n_classes == len(CLASS_NAMES), f"Expected {len(CLASS_NAMES)} classes, got {n_classes}"

    # Global importance
    shap.summary_plot(shap_values, X, plot_type="bar", class_names=CLASS_NAMES, show=False)
    plt.title("Global Feature Importance (Mean |SHAP| across all classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_global_importance_bar.png"), dpi=600, bbox_inches="tight")
    plt.close()

    # Per-class beeswarm + bar
    for i, class_name in enumerate(CLASS_NAMES):
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values[i], X, show=False)
        plt.title(f"SHAP Summary Plot (Beeswarm) - {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_beeswarm_{class_name}.png"), dpi=600, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values[i], X, plot_type="bar", show=False)
        plt.title(f"Feature Importance (Mean |SHAP|) - {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bar_{class_name}.png"), dpi=600, bbox_inches="tight")
        plt.close()

    # Top-N genes per class
    all_top = []
    for i, class_name in enumerate(CLASS_NAMES):
        mean_abs = np.abs(shap_values[i]).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:TOP_N]
        top = [(feature_names[j], mean_abs[j]) for j in idx]
        pd.DataFrame(top, columns=["Gene", "Mean_Abs_SHAP"]).to_csv(
            os.path.join(OUTPUT_DIR, f"top_{TOP_N}_genes_{class_name}.csv"), index=False)
        all_top += [{"Class": class_name, "Gene": g, "Mean_Abs_SHAP": s} for g, s in top]
    pd.DataFrame(all_top).to_csv(os.path.join(OUTPUT_DIR, "top_genes_all_classes.csv"), index=False)

    # Per-gene-per-class mean SHAP matrix (feeds 05b)
    mean_matrix = np.array([np.mean(shap_values[i], axis=0) for i in range(n_classes)]).T
    df_mean = pd.DataFrame(mean_matrix, index=feature_names, columns=CLASS_NAMES)
    df_mean.to_csv(os.path.join(OUTPUT_DIR, "mean_shap_per_gene_per_class.csv"))

    # Top-50 heatmap
    overall = np.mean([np.abs(shap_values[i]).mean(axis=0) for i in range(n_classes)], axis=0)
    top_genes = [feature_names[j] for j in np.argsort(overall)[::-1][:50]]
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_mean.loc[top_genes], cmap="vlag", center=0, linewidths=0.5)
    plt.title("Mean SHAP Values: Top 50 Genes Across Subtypes\n(Red = pushes toward subtype, Blue = pushes away)")
    plt.xlabel("Subtype")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_shap_heatmap_top50.png"), dpi=600, bbox_inches="tight")
    plt.close()

    print(f"\nDone. SHAP analysis complete. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
