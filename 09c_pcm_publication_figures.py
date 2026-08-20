# ============================================================
# CODE 6 — PUBLICATION-GRADE FIGURES & STATISTICS
# 09c — Publication Figures (paths: see PATHS section below)
#
# WHAT THIS SCRIPT DOES:
# Generates all additional figures and statistical tables needed
# for a manuscript submission. Run AFTER CODE4 and CODE5.
#
# OUTPUTS (written to OUTPUT_DIR/09_pcm_training_screening/publication/):
#   FIGURES (600 DPI PNG, publication ready):
#     Fig1_model_performance.png        — CV metrics bar chart with CI
#     Fig2_ROC_curves.png               — 5-fold ROC with mean
#     Fig3_PR_curves.png                — 5-fold PR with mean
#     Fig4_calibration.png              — reliability diagram
#     Fig5_confusion_aggregate.png      — aggregate confusion across folds
#     Fig6_per_target_performance.png   — per-target ROC-AUC
#     Fig7_y_randomization.png          — permutation test
#     Fig8_AD_coverage.png              — applicability domain coverage
#     Fig9_pchembl_distributions.png    — input data quality
#     Fig10_screening_scores.png        — virtual screening overview
#     Fig11_polypharmacology.png        — multi-target activity
#     Fig12_top_hits_heatmap.png        — top 50 heatmap
#     Fig13_target_correlation.png      — inter-target score correlation
#
#   TABLES:
#     Table1_cv_performance.csv         — main CV metrics table
#     Table2_per_target_metrics.csv     — per-target performance
#     Table3_top_hits_per_target.csv    — top 10 per target consolidated
#     Table4_polypharmacology.csv       — multi-target hits
#     Table5_dataset_statistics.csv     — data composition
#     Table6_y_randomization.csv        — permutation test results
#     Supplementary_full_cv_metrics.csv — all fold details
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef,
    balanced_accuracy_score, accuracy_score,
)
from sklearn.calibration import calibration_curve
from scipy import stats

# ============================================================
# PATHS
# ============================================================

BASE_DIR   = r"D:\MultiModal_Classification"
TRAIN_DIR  = os.path.join(BASE_DIR, "output", "09_pcm_training_screening", "training_cv")       # 09a output
SCREEN_DIR = os.path.join(BASE_DIR, "output", "09_pcm_training_screening", "virtual_screening")  # 09b output
FEAT_DIR   = os.path.join(BASE_DIR, "output", "08_pcm_features", "hybrid_matrix")                # 08d output
DATA_DIR   = os.path.join(BASE_DIR, "output", "08_pcm_features", "chembl")                       # 08b output
PUB_DIR    = os.path.join(BASE_DIR, "output", "09_pcm_training_screening", "publication")
os.makedirs(PUB_DIR, exist_ok=True)

PRED_DIR   = os.path.join(TRAIN_DIR, "predictions")
METRIC_DIR = os.path.join(TRAIN_DIR, "metrics")
AD_DIR     = os.path.join(TRAIN_DIR, "applicability_domain")

# ============================================================
# LOAD DATA
# ============================================================

print("Loading training results...")
cv_metrics   = pd.read_csv(os.path.join(METRIC_DIR, "cv_metrics.csv"))
with open(os.path.join(METRIC_DIR, "final_summary.json")) as f:
    summary = json.load(f)
with open(os.path.join(METRIC_DIR, "y_randomization.json")) as f:
    y_rand = json.load(f)

predictions = {}
for fold_i in range(1, 6):
    predictions[fold_i] = pd.read_csv(
        os.path.join(PRED_DIR, f"fold_{fold_i}_predictions.csv")
    )

ad_data = {}
for fold_i in range(1, 6):
    ad_data[fold_i] = pd.read_csv(
        os.path.join(AD_DIR, f"fold_{fold_i}_AD.csv")
    )

print("Loading screening results...")
vs_df    = pd.read_csv(os.path.join(SCREEN_DIR, "Complete_VS_Results.csv"))
poly_df  = pd.read_csv(os.path.join(SCREEN_DIR, "Polypharmacology_hits.csv"))
layout_f = os.path.join(FEAT_DIR, "feature_layout.json")
with open(layout_f) as f:
    layout = json.load(f)

TARGET_SCORE_COLS = [c for c in vs_df.columns if c.endswith("_score")
                     and c not in ("max_score", "mean_score")]
TARGET_NAMES      = [c.replace("_score", "") for c in TARGET_SCORE_COLS]
N_TARGETS         = len(TARGET_NAMES)

print(f"  Targets: {TARGET_NAMES}")

# Matplotlib style
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.linewidth": 1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":     150,
})
DPI = 600

# ============================================================
# FIGURE 1: Model Performance Bar Chart (main metric summary)
# ============================================================

print("Figure 1: Model performance summary...")
metric_keys  = ["roc_auc", "pr_auc", "f1", "mcc", "balanced_acc"]
metric_labels= ["ROC-AUC", "PR-AUC", "F1", "MCC", "Balanced Acc"]
means = [summary[m]["mean"] for m in metric_keys]
cis   = [summary[m]["ci_95"] for m in metric_keys]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
bars = ax.bar(metric_labels, means, yerr=cis, capsize=7,
              color=colors, alpha=0.85, edgecolor="black", width=0.55)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("PCM Model Performance — 5-Fold Cross-Validation\n"
             "(mean ± 95% CI)", fontsize=13)
ax.axhline(0.5, color="gray", lw=1, linestyle=":")
for bar, m, ci in zip(bars, means, cis):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + ci + 0.02,
            f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig1_model_performance.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 2: ROC Curves (5 folds + mean)
# ============================================================

print("Figure 2: ROC curves...")
mean_fpr = np.linspace(0, 1, 200)
tprs, aucs = [], []

fig, ax = plt.subplots(figsize=(6.5, 6))
fold_colors = ["#90CAF9", "#80CBC4", "#A5D6A7", "#FFCC80", "#EF9A9A"]

for fold_i in range(1, 6):
    pred = predictions[fold_i]
    fpr, tpr, _ = roc_curve(pred["true"], pred["prob"])
    auc_val = roc_auc_score(pred["true"], pred["prob"])
    aucs.append(auc_val)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    ax.plot(fpr, tpr, lw=1.2, alpha=0.6, color=fold_colors[fold_i-1],
            label=f"Fold {fold_i} (AUC={auc_val:.3f})")

mean_tpr    = np.mean(tprs, axis=0)
std_tpr     = np.std(tprs,  axis=0)
mean_auc    = summary["roc_auc"]["mean"]
ci          = summary["roc_auc"]["ci_95"]

ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                alpha=0.15, color="navy")
ax.plot(mean_fpr, mean_tpr, "navy", lw=2.5,
        label=f"Mean (AUC={mean_auc:.3f} ± {ci:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Receiver Operating Characteristic — 5-Fold CV", fontsize=12)
ax.legend(fontsize=8.5, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig2_ROC_curves.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 3: PR Curves
# ============================================================

print("Figure 3: PR curves...")
fig, ax = plt.subplots(figsize=(6.5, 6))

for fold_i in range(1, 6):
    pred   = predictions[fold_i]
    p, r, _ = precision_recall_curve(pred["true"], pred["prob"])
    ap      = average_precision_score(pred["true"], pred["prob"])
    ax.plot(r, p, lw=1.5, alpha=0.65, color=fold_colors[fold_i-1],
            label=f"Fold {fold_i} (AP={ap:.3f})")

mean_pr = summary["pr_auc"]["mean"]
ci_pr   = summary["pr_auc"]["ci_95"]
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title(f"Precision-Recall Curves — 5-Fold CV\n"
             f"Mean AP = {mean_pr:.3f} ± {ci_pr:.3f}", fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim([0, 1.01])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig3_PR_curves.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 4: Calibration Plot
# ============================================================

print("Figure 4: Calibration...")
all_true, all_prob = [], []
for fold_i in range(1, 6):
    pred = predictions[fold_i]
    all_true.extend(pred["true"].tolist())
    all_prob.extend(pred["prob"].tolist())

prob_true, prob_pred = calibration_curve(all_true, all_prob, n_bins=10)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(prob_pred, prob_true, "o-", lw=2, color="steelblue", label="PCM Model")
ax.fill_between(prob_pred, prob_true, prob_pred,
                alpha=0.15, color="steelblue")
ax.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig4_calibration.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 5: Aggregate Confusion Matrix
# ============================================================

print("Figure 5: Aggregate confusion matrix...")
agg_cm = np.zeros((2, 2), dtype=int)
for fold_i in range(1, 6):
    pred = predictions[fold_i]
    thr  = cv_metrics.loc[cv_metrics["fold"]==fold_i, "threshold"].values[0]
    y_pred = (pred["prob"] >= thr).astype(int)
    agg_cm += confusion_matrix(pred["true"], y_pred)

# Normalize for percentage display
cm_norm = agg_cm.astype(float) / agg_cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

sns.heatmap(agg_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Inactive", "Active"],
            yticklabels=["Inactive", "Active"],
            cbar_kws={"label": "Count"})
axes[0].set_xlabel("Predicted", fontsize=11)
axes[0].set_ylabel("True", fontsize=11)
axes[0].set_title("Aggregate Confusion Matrix\n(counts, all 5 folds)", fontsize=11)

sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", ax=axes[1],
            xticklabels=["Inactive", "Active"],
            yticklabels=["Inactive", "Active"],
            cbar_kws={"label": "Fraction"}, vmin=0, vmax=1)
axes[1].set_xlabel("Predicted", fontsize=11)
axes[1].set_ylabel("True", fontsize=11)
axes[1].set_title("Normalized Confusion Matrix\n(row-normalized, all 5 folds)", fontsize=11)

plt.suptitle("Confusion Matrices — Aggregated Across All Folds", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig5_confusion_aggregate.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 6: Per-Target Performance
# ============================================================

print("Figure 6: Per-target performance...")
pt_df_path = os.path.join(METRIC_DIR, "per_target_metrics.csv")
if os.path.exists(pt_df_path):
    pt_df = pd.read_csv(pt_df_path)
    pt_mean = pt_df.groupby("uniprot_id").agg(
        mean_auc=("roc_auc", "mean"),
        std_auc=("roc_auc", "std"),
        mean_f1=("f1", "mean"),
        n=("n", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(pt_mean))
    width  = 0.38
    colors_auc = plt.cm.Blues(np.linspace(0.45, 0.85, len(pt_mean)))
    colors_f1  = plt.cm.Greens(np.linspace(0.45, 0.85, len(pt_mean)))
    bars1 = ax.bar(x - width/2, pt_mean["mean_auc"],
                   width, yerr=pt_mean["std_auc"], capsize=5,
                   color=colors_auc, edgecolor="black", label="ROC-AUC")
    bars2 = ax.bar(x + width/2, pt_mean["mean_f1"],
                   width, color=colors_f1, edgecolor="black", label="F1")
    ax.axhline(0.5, color="red", lw=1, linestyle="--")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(pt_mean["uniprot_id"], rotation=30, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Target Model Performance (mean ± SD across folds)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PUB_DIR, "Fig6_per_target_performance.png"), dpi=DPI)
    plt.close()

# ============================================================
# FIGURE 7: Y-Randomization
# ============================================================

print("Figure 7: Y-randomization...")
rand_aucs   = y_rand["all_random_aucs"]
orig_auc    = y_rand["original_auc"]
rand_mean   = y_rand["random_mean_auc"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(rand_aucs, bins=8, color="#EF9A9A", edgecolor="black",
        alpha=0.85, label="Y-randomized models")
ax.axvline(orig_auc, color="#1565C0", lw=2.5,
           label=f"True model (AUC={orig_auc:.4f})")
ax.axvline(rand_mean, color="red", lw=1.5, linestyle="--",
           label=f"Random mean (AUC={rand_mean:.4f})")
ax.set_xlabel("ROC-AUC", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Y-Randomization Test\n"
             f"Gap = {y_rand['gap']:.4f} | {y_rand['verdict']}", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig7_y_randomization.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 8: Applicability Domain Coverage
# ============================================================

print("Figure 8: Applicability domain...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: AD coverage bar per fold
coverages = []
for fold_i in range(1, 6):
    ad   = ad_data[fold_i]
    cov  = ad["inside_AD"].mean()
    coverages.append(cov)

axes[0].bar(range(1, 6), coverages, color="teal", edgecolor="black", alpha=0.8)
axes[0].axhline(np.mean(coverages), color="red", lw=1.5, linestyle="--",
                label=f"Mean = {np.mean(coverages):.3f}")
axes[0].set_xlabel("Fold", fontsize=12)
axes[0].set_ylabel("Fraction within AD", fontsize=12)
axes[0].set_title("Applicability Domain Coverage per Fold", fontsize=12)
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=9)

# Right: AD distance distribution
all_dist_in  = []
all_dist_out = []
for fold_i in range(1, 6):
    ad = ad_data[fold_i]
    all_dist_in.extend( ad[ad["inside_AD"]==True]["distance"].tolist())
    all_dist_out.extend(ad[ad["inside_AD"]==False]["distance"].tolist())

axes[1].hist(all_dist_in,  bins=40, alpha=0.7, color="teal",   label="Inside AD", density=True)
axes[1].hist(all_dist_out, bins=40, alpha=0.7, color="salmon", label="Outside AD", density=True)
axes[1].set_xlabel("Distance to Nearest Training Compound", fontsize=12)
axes[1].set_ylabel("Density", fontsize=12)
axes[1].set_title("AD Distance Distribution", fontsize=12)
axes[1].legend(fontsize=10)

plt.suptitle("Applicability Domain Analysis", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig8_AD_coverage.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 9: pChEMBL Distributions (input data quality)
# ============================================================

print("Figure 9: pChEMBL distributions...")
if os.path.exists(os.path.join(DATA_DIR, "pchembl_distribution.csv")):
    dist_df = pd.read_csv(os.path.join(DATA_DIR, "pchembl_distribution.csv"))

    # Load clean dataset for violin plot
    clean_df = pd.read_csv(os.path.join(DATA_DIR, "PCM_dataset_clean.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: violin plot
    tnames = clean_df["target_name"].unique()
    data_by_target = [clean_df[clean_df["target_name"]==t]["pchembl_value"].values
                      for t in tnames]
    parts = axes[0].violinplot(data_by_target, positions=range(len(tnames)),
                               showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    axes[0].set_xticks(range(len(tnames)))
    axes[0].set_xticklabels(tnames, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("pChEMBL Value", fontsize=12)
    axes[0].set_title("pChEMBL Value Distribution per Target", fontsize=12)
    axes[0].axhline(6.0, color="red", lw=1, linestyle="--",
                    label="pChEMBL=6 (reference)")
    axes[0].legend(fontsize=9)

    # Right: compound count per target
    target_counts = clean_df.groupby("target_name").size().sort_values(ascending=False)
    axes[1].bar(target_counts.index, target_counts.values,
                color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].set_ylabel("Number of Compounds", fontsize=12)
    axes[1].set_title("Dataset Composition per Target", fontsize=12)
    axes[1].set_xticklabels(target_counts.index, rotation=30, ha="right")
    for i, (_, v) in enumerate(target_counts.items()):
        axes[1].text(i, v+2, str(v), ha="center", va="bottom", fontsize=8)

    plt.suptitle("Training Dataset Quality Overview", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PUB_DIR, "Fig9_pchembl_distributions.png"), dpi=DPI)
    plt.close()

# ============================================================
# FIGURE 10: Screening Score Overview
# ============================================================

print("Figure 10: Screening score overview...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: max score distribution
axes[0].hist(vs_df["max_score"], bins=50, color="steelblue",
             edgecolor="black", alpha=0.8)
axes[0].set_xlabel("Maximum Score", fontsize=12)
axes[0].set_ylabel("Number of Compounds", fontsize=12)
axes[0].set_title("Distribution of Maximum Scores\n(NPASS Library)", fontsize=12)

# Right: cumulative active fraction vs threshold
thresholds = np.linspace(0, 1, 100)
n_active   = [(vs_df["max_score"] >= t).mean() for t in thresholds]
axes[1].plot(thresholds, n_active, "steelblue", lw=2)
axes[1].set_xlabel("Score Threshold", fontsize=12)
axes[1].set_ylabel("Fraction of Library Active", fontsize=12)
axes[1].set_title("Active Fraction vs Score Threshold", fontsize=12)
axes[1].axvline(0.5, color="red", lw=1.5, linestyle="--",
                label="Threshold=0.5")
axes[1].legend(fontsize=10)

plt.suptitle("Virtual Screening Overview", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig10_screening_scores.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 11: Polypharmacology Profile
# ============================================================

print("Figure 11: Polypharmacology...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: distribution of active target count
counts = vs_df["n_active_targets"].value_counts().sort_index()
axes[0].bar(counts.index, counts.values, color="teal",
            edgecolor="black", alpha=0.85)
axes[0].set_xlabel("Number of Active Targets per Compound", fontsize=12)
axes[0].set_ylabel("Number of Compounds", fontsize=12)
axes[0].set_title("Polypharmacology Distribution", fontsize=12)
axes[0].set_xticks(range(0, N_TARGETS + 1))
for x, y in zip(counts.index, counts.values):
    axes[0].text(x, y+1, str(y), ha="center", va="bottom", fontsize=9)

# Right: target co-activity matrix (how often pairs of targets share hits)
score_mat = vs_df[TARGET_SCORE_COLS].values
thr_val   = 0.5
active_mat = (score_mat >= thr_val).astype(int)
co_activity = active_mat.T @ active_mat     # (N_TARGETS, N_TARGETS)

# Normalize by geometric mean of individual activity counts
diag    = np.diag(co_activity).astype(float)
denom   = np.sqrt(np.outer(diag, diag)) + 1e-8
co_norm = co_activity / denom

co_df   = pd.DataFrame(co_norm, index=TARGET_NAMES, columns=TARGET_NAMES)
sns.heatmap(co_df, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1,
            linewidths=0.5, ax=axes[1],
            cbar_kws={"label": "Normalised co-activity"})
axes[1].set_title("Target Co-Activity Matrix\n(fraction of shared hits)", fontsize=12)
axes[1].set_xticklabels(TARGET_NAMES, rotation=30, ha="right")

plt.suptitle("Polypharmacology Analysis", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig11_polypharmacology.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 12: Top Hits Heatmap
# ============================================================

print("Figure 12: Top hits heatmap...")
top50 = vs_df.nlargest(50, "max_score")[
    ["molecule_id"] + TARGET_SCORE_COLS
].set_index("molecule_id")

fig, ax = plt.subplots(figsize=(12, 16))
sns.heatmap(top50.values, xticklabels=TARGET_NAMES,
            yticklabels=top50.index.tolist(),
            cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=0.3, linecolor="lightgray",
            cbar_kws={"label": "Ensemble Score"},
            ax=ax)
ax.set_xticklabels(TARGET_NAMES, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
ax.set_title("Top 50 Compounds — Ensemble Prediction Scores\nAcross All 8 Targets",
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig12_top_hits_heatmap.png"), dpi=DPI)
plt.close()

# ============================================================
# FIGURE 13: Inter-Target Score Correlation
# ============================================================

print("Figure 13: Target score correlation...")
corr = vs_df[TARGET_SCORE_COLS].corr()
corr.index   = TARGET_NAMES
corr.columns = TARGET_NAMES

mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pearson r"})
ax.set_title("Inter-Target Score Correlation\n(Pearson r, lower triangle)",
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PUB_DIR, "Fig13_target_correlation.png"), dpi=DPI)
plt.close()

# ============================================================
# TABLES
# ============================================================

print("\nGenerating publication tables...")

# Table 1: Main CV performance
t1_rows = []
for m in ["roc_auc", "pr_auc", "accuracy", "f1", "mcc",
          "balanced_acc", "precision", "recall"]:
    t1_rows.append({
        "Metric": m.replace("_", " ").title(),
        "Mean":   summary[m]["mean"],
        "95% CI": summary[m]["ci_95"],
        "Summary": summary[m]["str"],
        "Fold 1": summary[m]["values"][0],
        "Fold 2": summary[m]["values"][1],
        "Fold 3": summary[m]["values"][2],
        "Fold 4": summary[m]["values"][3],
        "Fold 5": summary[m]["values"][4],
    })
t1 = pd.DataFrame(t1_rows)
t1.to_csv(os.path.join(PUB_DIR, "Table1_cv_performance.csv"), index=False)
print("  Table 1: CV performance")

# Table 2: Per-target metrics
if os.path.exists(os.path.join(METRIC_DIR, "per_target_metrics.csv")):
    pt_df = pd.read_csv(os.path.join(METRIC_DIR, "per_target_metrics.csv"))
    t2 = pt_df.groupby("uniprot_id").agg(
        N_compounds_mean=("n", "mean"),
        ROC_AUC_mean=("roc_auc", "mean"),
        ROC_AUC_std=("roc_auc", "std"),
        F1_mean=("f1", "mean"),
        F1_std=("f1", "std"),
    ).round(4).reset_index()
    t2.to_csv(os.path.join(PUB_DIR, "Table2_per_target_metrics.csv"), index=False)
    print("  Table 2: Per-target metrics")

# Table 3: Top 10 per target (consolidated)
t3_rows = []
for tname, col in zip(TARGET_NAMES, TARGET_SCORE_COLS):
    top10 = vs_df.nlargest(10, col)[["molecule_id", "smiles", col]].copy()
    top10["rank"]   = range(1, 11)
    top10["target"] = tname
    t3_rows.append(top10)
t3 = pd.concat(t3_rows, ignore_index=True)
t3.to_csv(os.path.join(PUB_DIR, "Table3_top_hits_per_target.csv"), index=False)
print("  Table 3: Top 10 per target")

# Table 4: Polypharmacology hits
t4 = poly_df[["molecule_id", "smiles", "n_active_targets", "mean_score", "max_score"]
             + TARGET_SCORE_COLS].copy()
t4.to_csv(os.path.join(PUB_DIR, "Table4_polypharmacology.csv"), index=False)
print("  Table 4: Polypharmacology hits")

# Table 5: Dataset statistics
feat_stats_path = os.path.join(FEAT_DIR, "dataset_statistics.csv")
if os.path.exists(feat_stats_path):
    t5 = pd.read_csv(feat_stats_path)
    t5.to_csv(os.path.join(PUB_DIR, "Table5_dataset_statistics.csv"), index=False)
    print("  Table 5: Dataset statistics")

# Table 6: Y-randomization
t6 = pd.DataFrame([{
    "Original ROC-AUC":    y_rand["original_auc"],
    "Random Mean AUC":     y_rand["random_mean_auc"],
    "Random SD":           y_rand["random_std"],
    "Gap":                 y_rand["gap"],
    "Verdict":             y_rand["verdict"],
}])
t6.to_csv(os.path.join(PUB_DIR, "Table6_y_randomization.csv"), index=False)
print("  Table 6: Y-randomization")

# Supplementary: full CV metrics
cv_metrics.to_csv(
    os.path.join(PUB_DIR, "Supplementary_full_cv_metrics.csv"), index=False
)
print("  Supplementary: full CV metrics")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CODE 6 COMPLETE — All publication outputs generated")
print("=" * 60)
print(f"  Output directory: {PUB_DIR}")
print(f"\n  Figures (13):")
for i in range(1, 14):
    fname = [f for f in os.listdir(PUB_DIR) if f.startswith(f"Fig{i}_")]
    if fname:
        print(f"    {fname[0]}")
print(f"\n  Tables (7):")
for f in sorted(os.listdir(PUB_DIR)):
    if f.startswith("Table") or f.startswith("Supplementary"):
        print(f"    {f}")