import os
import platform
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels
import xgboost
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# ============================================================================
# CONFIG paths
# ============================================================================
BASE_DIR = r"D:\MultiModal_Classification"
COMBINED_EXPR_CSV = os.path.join(BASE_DIR, "output", "02_batch_correction_deg_pca", "combined_after_batch_correction.csv")
OUTPUT_DIR        = os.path.join(BASE_DIR, "output", "07_leakage_pca_variance_threshold")

META_COLS = ['PATIENT_ID', 'SUBTYPE', 'AGE', 'SEX', 'TUMOR_STAGE',
             'OS_STATUS', 'OS_MONTHS', 'BATCH']
LABEL_COL = 'SUBTYPE'
BATCH_LABEL_MAP = {
    "batch_2012": "TCGA-BRCA-2012",
    "batch_2016": "METABRIC-BRCA-2016",
    "batch_2018": "TCGA-BRCA-2018",
}

N_FOLDS = 5
RANDOM_STATE = 42
DEG_TOP_N_PER_SUBTYPE_POOL = 200
TOP_N_SIGNATURE_PER_CLASS = 10
ORIGINAL_PCA_VARIANCE = 0.90     # the manuscript's stated cutoff
VARIANCE_THRESHOLDS_TO_COMPARE = [0.80, 0.90, 0.95]
EFFECT_SIZE_THRESH = 0.3
PVAL_THRESHOLD = 0.05


# ============================================================================
# Reproducibility
# ============================================================================

def reproducibility_table():
    return pd.DataFrame([
        {"Item": "Python", "Value": platform.python_version()},
        {"Item": "numpy", "Value": np.__version__},
        {"Item": "pandas", "Value": pd.__version__},
        {"Item": "scipy", "Value": scipy.__version__},
        {"Item": "statsmodels", "Value": statsmodels.__version__},
        {"Item": "scikit-learn", "Value": sklearn.__version__},
        {"Item": "xgboost", "Value": xgboost.__version__},
        {"Item": "Random seed", "Value": str(RANDOM_STATE)},
        {"Item": "N folds", "Value": str(N_FOLDS)},
        {"Item": "Variance thresholds compared", "Value": str(VARIANCE_THRESHOLDS_TO_COMPARE)},
    ])


# ============================================================================
# Shared building blocks
# ============================================================================

def safe_ttest(a, b):
    if len(a) < 2 or len(b) < 2 or (np.std(a) == 0 and np.std(b) == 0):
        return 0.0, 1.0
    return stats.ttest_ind(a, b, equal_var=False)


def deg_single_class_vs_rest(X, y_binary, gene_names,
                              top_n=DEG_TOP_N_PER_SUBTYPE_POOL,
                              effect_thresh=EFFECT_SIZE_THRESH,
                              qval_thresh=PVAL_THRESHOLD):
    mask_t = (y_binary == 1)
    mask_r = (y_binary == 0)
    if mask_t.sum() < 2 or mask_r.sum() < 2:
        return []
    Xt, Xr = X[mask_t], X[mask_r]
    pvals, zdiffs = [], []
    for j in range(X.shape[1]):
        _, pv = safe_ttest(Xt[:, j], Xr[:, j])
        pvals.append(pv)
        zdiffs.append(Xt[:, j].mean() - Xr[:, j].mean())
    pvals = np.array(pvals)
    zdiffs = np.array(zdiffs)
    qvals = sm.stats.multipletests(pvals, method="fdr_bh")[1]
    sig_mask = (qvals < qval_thresh) & (np.abs(zdiffs) > effect_thresh)
    idx_sig = np.where(sig_mask)[0]
    if len(idx_sig) == 0:
        return []
    pos_idx = idx_sig[zdiffs[idx_sig] > 0]
    neg_idx = idx_sig[zdiffs[idx_sig] < 0]
    pos_top = pos_idx[np.argsort(-zdiffs[pos_idx])[:top_n]]
    neg_top = neg_idx[np.argsort(zdiffs[neg_idx])[:top_n]]
    genes = [gene_names[i] for i in np.concatenate([pos_top, neg_top]).astype(int)]
    return sorted(set(genes))


def _pc_relevance_weights(scores, y_binary):
    weights = np.zeros(scores.shape[1])
    for k in range(scores.shape[1]):
        col = scores[:, k]
        if np.std(col) == 0 or np.std(y_binary) == 0:
            weights[k] = 0.0
        else:
            weights[k] = abs(np.corrcoef(col, y_binary)[0, 1])
    return weights


def select_top_genes_for_one_subtype(X_class_deg, y_binary, gene_names_class,
                                      top_n=TOP_N_SIGNATURE_PER_CLASS,
                                      variance_threshold=ORIGINAL_PCA_VARIANCE,
                                      random_state=RANDOM_STATE):
    """Relevance-weighted loading aggregated across ALL retained PCs"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_class_deg)
    pca_full = PCA(random_state=random_state)
    pca_full.fit(Xs)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= variance_threshold) + 1)
    n_components = max(1, min(n_components, Xs.shape[1]))
    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(Xs)
    weights = _pc_relevance_weights(scores, y_binary)
    loadings = pca.components_
    gene_scores = np.sum(np.abs(loadings) * weights[:, None], axis=0)
    top_idx = np.argsort(-gene_scores)[:min(top_n, len(gene_names_class))]
    return [gene_names_class[i] for i in top_idx], n_components


def build_signature_per_subtype(X, y, gene_names,
                                 top_n_per_class=TOP_N_SIGNATURE_PER_CLASS,
                                 deg_top_n=DEG_TOP_N_PER_SUBTYPE_POOL,
                                 variance_threshold=ORIGINAL_PCA_VARIANCE,
                                 random_state=RANDOM_STATE):
    """In the Fold-wise selection pipeline this is called with (X_train, y_train)
    ONLY. In the Full cohort pipeline (for comparison purposes only) it is
    called once with the Full cohort selectiion(X_full, y_full)."""
    classes = np.unique(y)
    all_selected = []
    n_pcs_list = []
    for cls in classes:
        mask_t = (y == cls)
        if mask_t.sum() < 2 or (~mask_t).sum() < 2:
            continue
        deg_genes_cls = deg_single_class_vs_rest(X, mask_t.astype(int), gene_names, top_n=deg_top_n)
        if len(deg_genes_cls) < 2:
            continue
        idx = [gene_names.index(g) for g in deg_genes_cls]
        Xc = X[:, idx]
        y_bin = mask_t.astype(int)
        top_genes, n_comp = select_top_genes_for_one_subtype(
            Xc, y_bin, deg_genes_cls, top_n=top_n_per_class,
            variance_threshold=variance_threshold, random_state=random_state
        )
        all_selected.extend(top_genes)
        n_pcs_list.append(n_comp)
    signature_genes = sorted(set(all_selected))
    avg_n_pcs = float(np.mean(n_pcs_list)) if n_pcs_list else np.nan
    return signature_genes, avg_n_pcs


def metrics_for_subset(y_true, y_pred, y_prob, classes):
    if len(np.unique(y_true)) < 2:
        return {"accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": np.nan, "mcc": np.nan, "macro_auc": np.nan}
    y_true_bin = label_binarize(y_true, classes=classes)
    try:
        auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc = np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "macro_auc": auc,
    }


def fit_and_score_fold(Xtr_full, Xte_full, ytr, yte, sig_genes, gene_cols, classes):
    sig_idx = [gene_cols.index(g) for g in sig_genes]
    Xtr_sig, Xte_sig = Xtr_full[:, sig_idx], Xte_full[:, sig_idx]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_sig)
    Xte_s = scaler.transform(Xte_sig)
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         objective="multi:softprob", eval_metric="mlogloss",
                         random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(Xtr_s, ytr)
    y_pred = clf.predict(Xte_s)
    y_prob = clf.predict_proba(Xte_s)
    return metrics_for_subset(yte, y_pred, y_prob, classes)


# ============================================================================
# Part A — Leakage sensitivity test (Full cohort vs. Fold-wise)
# ============================================================================

def run_leakage_sensitivity_test(df, gene_cols):
    X_full = df[gene_cols].values.astype(float)
    y_full = df[LABEL_COL].values
    classes = np.unique(y_full)

    print("  Deriving the Full cohort signature once on the FULL pooled dataset...")
    Full cohort_sig_genes, Full cohort_avg_pcs = build_signature_per_subtype(
        X_full, y_full, gene_cols, variance_threshold=ORIGINAL_PCA_VARIANCE
    )
    print(f"  Full cohort signature size: {len(Full cohort_sig_genes)} genes "
          f"(mean retained PCs/subtype: {Full cohort_avg_pcs:.1f}).\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_full, y_full), start=1):
        Xtr_full, Xte_full = X_full[tr_idx], X_full[te_idx]
        ytr, yte = y_full[tr_idx], y_full[te_idx]

        Full cohort_metrics = fit_and_score_fold(Xtr_full, Xte_full, ytr, yte, Full cohort_sig_genes, gene_cols, classes)
        Full cohort_metrics.update({"fold": fold_i, "pipeline": "Full cohort (DEG+PCA on full pooled data)",
                               "n_signature_genes": len(Full cohort_sig_genes), "avg_n_pcs_per_subtype": Full cohort_avg_pcs})
        rows.append(Full cohort_metrics)

        corrected_sig_genes, corrected_avg_pcs = build_signature_per_subtype(
            Xtr_full, ytr, gene_cols, variance_threshold=ORIGINAL_PCA_VARIANCE
        )
        if len(corrected_sig_genes) < 2:
            print(f"  Fold {fold_i}: too few corrected signature genes, skipping corrected pipeline.")
            continue
        corrected_metrics = fit_and_score_fold(Xtr_full, Xte_full, ytr, yte, corrected_sig_genes, gene_cols, classes)
        corrected_metrics.update({"fold": fold_i, "pipeline": "Fold-wise (DEG+PCA on training fold only)",
                                   "n_signature_genes": len(corrected_sig_genes), "avg_n_pcs_per_subtype": corrected_avg_pcs})
        rows.append(corrected_metrics)

        print(f"  Fold {fold_i}: Full cohort acc={Full cohort_metrics['accuracy']:.4f}  |  "
              f"CORRECTED acc={corrected_metrics['accuracy']:.4f}  "
              f"(delta={Full cohort_metrics['accuracy']-corrected_metrics['accuracy']:+.4f})")

    return pd.DataFrame(rows)


# ============================================================================
# Part B — PCA variance-threshold comparison (Fold-wise pipeline only)
# ============================================================================

def run_variance_threshold_comparison(df, gene_cols):
    X_full = df[gene_cols].values.astype(float)
    y_full = df[LABEL_COL].values
    classes = np.unique(y_full)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for threshold in VARIANCE_THRESHOLDS_TO_COMPARE:
        print(f"\n  Variance threshold = {threshold:.0%}")
        for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_full, y_full), start=1):
            Xtr_full, Xte_full = X_full[tr_idx], X_full[te_idx]
            ytr, yte = y_full[tr_idx], y_full[te_idx]
            sig_genes, avg_n_pcs = build_signature_per_subtype(Xtr_full, ytr, gene_cols, variance_threshold=threshold)
            if len(sig_genes) < 2:
                print(f"    Fold {fold_i}: too few signature genes, skipping.")
                continue
            m = fit_and_score_fold(Xtr_full, Xte_full, ytr, yte, sig_genes, gene_cols, classes)
            m.update({"fold": fold_i, "variance_threshold": threshold,
                       "n_signature_genes": len(sig_genes), "avg_n_pcs_per_subtype": avg_n_pcs})
            rows.append(m)
            print(f"    Fold {fold_i}: acc={m['accuracy']:.4f}, n_sig_genes={len(sig_genes)}, "
                  f"avg_pcs/subtype={avg_n_pcs:.1f}")

    return pd.DataFrame(rows)


# ============================================================================
# Part C — Leakage-control confirmation statement
# ============================================================================

def build_leakage_control_statement(leak_df):
    Full cohort = leak_df[leak_df["pipeline"].str.startswith("Full cohort")]
    corrected = leak_df[leak_df["pipeline"].str.startswith("Fold-wise")]
    Full cohort_mean, Full cohort_std = Full cohort["accuracy"].mean(), Full cohort["accuracy"].std()
    corr_mean, corr_std = corrected["accuracy"].mean(), corrected["accuracy"].std()
    delta = Full cohort_mean - corr_mean

    text = f"""LEAKAGE-CONTROL CONFIRMATION STATEMENT
========================================
Gene loadings and DEG filtering in the corrected (Fold-wise) pipeline
are computed EXCLUSIVELY on training-fold data. build_signature_per_subtype()
is called only with (X_train, y_train, gene_names) drawn from the outer
StratifiedKFold TRAINING indices; held-out test-fold rows are never passed
into deg_single_class_vs_rest() or select_top_genes_for_one_subtype() (the
functions performing DEG filtering and PCA respectively). Test-fold data
is introduced for the first time only at scaler.transform(Xte_sig) and
clf.predict(Xte_s), i.e. after all feature selection is already fixed.

Leakage sensitivity test result (this run, computed live -- not quoted
from any prior run or the manuscript):
  Full cohort pipeline (DEG+PCA on full pooled data):        {Full cohort_mean:.4f} +/- {Full cohort_std:.4f} accuracy
  Fold-wise pipeline (DEG+PCA on training fold only): {corr_mean:.4f} +/- {corr_std:.4f} accuracy
  Apparent inflation attributable to leakage: {delta:+.4f} accuracy points

Performance is summarized as mean +/- standard deviation across five
folds; no confidence interval is computed or claimed.
"""
    return text, Full cohort_mean, Full cohort_std, corr_mean, corr_std, delta


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(COMBINED_EXPR_CSV):
        raise FileNotFoundError(f"Could not find {COMBINED_EXPR_CSV}.")
    df = pd.read_csv(COMBINED_EXPR_CSV)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "BATCH" not in df.columns:
        raise KeyError("No 'BATCH' column found in the input file.")
    df["COHORT"] = df["BATCH"].map(BATCH_LABEL_MAP).fillna(df["BATCH"].astype(str))
    gene_cols = [c for c in df.columns if c not in META_COLS + ["COHORT"]]
    print(f"Detected {len(gene_cols)} gene expression columns, {len(df)} patients.\n")

    repro_df = reproducibility_table()
    repro_df.to_csv(os.path.join(OUTPUT_DIR, "Reproducibility_Info.csv"), index=False)

    # ---- Part (a) ----
    print("=" * 70); print("PART (a): Leakage sensitivity test (Full cohort vs. Fold-wise)"); print("=" * 70)
    leak_df = run_leakage_sensitivity_test(df, gene_cols)
    leak_df.to_csv(os.path.join(OUTPUT_DIR, "ST_LeakageSensitivity.csv"), index=False)
    leak_summary = leak_df.groupby("pipeline")[["accuracy", "macro_f1", "mcc", "macro_auc"]].agg(["mean", "std"])
    leak_summary.to_csv(os.path.join(OUTPUT_DIR, "ST_LeakageSensitivity_summary.csv"))
    print(leak_summary)

    # ---- Part (b) ----
    print("\n" + "=" * 70); print("PART (b): PCA variance threshold comparison (80% / 90% / 95%)"); print("=" * 70)
    var_df = run_variance_threshold_comparison(df, gene_cols)
    var_df.to_csv(os.path.join(OUTPUT_DIR, "ST_PCA_VarianceThreshold.csv"), index=False)
    var_summary = var_df.groupby("variance_threshold")[
        ["accuracy", "macro_f1", "mcc", "macro_auc", "n_signature_genes", "avg_n_pcs_per_subtype"]
    ].agg(["mean", "std"])
    var_summary.to_csv(os.path.join(OUTPUT_DIR, "ST_PCA_VarianceThreshold_summary.csv"))
    print(var_summary)

    # ---- Part (c) ----
    print("\n" + "=" * 70); print("PART (c): Leakage-control confirmation statement"); print("=" * 70)
    statement, Full cohort_mean, Full cohort_std, corr_mean, corr_std, delta = build_leakage_control_statement(leak_df)
    with open(os.path.join(OUTPUT_DIR, "Leakage_Control_Statement.txt"), "w") as f:
        f.write(statement)
    print(statement)

    # ---- Auto-generated bullet summary (no manuscript/rebuttal prose) ----
    var_acc = var_df.groupby("variance_threshold")["accuracy"].mean()
    var_pcs = var_df.groupby("variance_threshold")["avg_n_pcs_per_subtype"].mean()
    var_genes = var_df.groupby("variance_threshold")["n_signature_genes"].mean()

    summary_lines = [
        "COMMENT 2 -- ANALYSIS SUMMARY",
        "=" * 40,
        f"- Leakage sensitivity test: Full cohort pipeline {Full cohort_mean:.3f} +/- {Full cohort_std:.3f} accuracy "
        f"vs. Fold-wise pipeline {corr_mean:.3f} +/- {corr_std:.3f} accuracy "
        f"(delta = {delta:+.3f}), both computed live in this run on identical folds/seed/classifier.",
        "- Variance threshold comparison (Fold-wise pipeline, mean accuracy / mean retained "
        "PCs-per-subtype / mean signature genes):",
    ]
    for t in VARIANCE_THRESHOLDS_TO_COMPARE:
        summary_lines.append(
            f"    {t:.0%}: acc={var_acc.get(t, float('nan')):.3f}, "
            f"PCs/subtype~{var_pcs.get(t, float('nan')):.1f}, "
            f"signature genes~{var_genes.get(t, float('nan')):.1f}"
        )
    summary_lines += [
        "- Gene loadings and DEG statistics confirmed (by code inspection, not by claim) to be "
        "computed exclusively from training-fold data in the Fold-wise pipeline; see "
        "Leakage_Control_Statement.txt.",
        "- Gene ranking method: relevance-weighted loading aggregated across all retained PCs "
        "per subtype, matching the manuscript's plural 'principal componentS' wording.",
        "- Performance reported as mean +/- SD across 5 folds; no confidence interval computed or claimed.",
        "- No external validation content in this script (out of scope for Comment 2).",
        "- Reproducibility info (package versions, random seed) saved to Reproducibility_Info.csv.",
    ]
    summary_text = "\n".join(summary_lines)
    with open(os.path.join(OUTPUT_DIR, "Analysis_summary.txt"), "w") as f:
        f.write(summary_text + "\n")
    print("\n" + summary_text)
    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
