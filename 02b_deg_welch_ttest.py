import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings("ignore")

# ============================================================
# PATHS & PARAMETERS
# ============================================================
BASE_DIR    = r"D:\MultiModal_Classification"
STAGE_DIR   = os.path.join(BASE_DIR, "output", "02_batch_correction_deg_pca")
INPUT_FILE  = os.path.join(STAGE_DIR, "combined_after_batch_correction.csv")
OUTPUT_DIR  = os.path.join(STAGE_DIR, "DE_Welch")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBTYPE_COLUMN = "SUBTYPE"
ID_COLUMNS = ["SUBTYPE", "TUMOR_STAGE", "OS_MONTHS", "OS_STATUS", "AGE", "SEX", "BATCH"]
QVAL_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.3
TOP_N_GENES = 50
PLOT_DPI = 600
SUBTYPE_MAP = {0: "Normal", 1: "Luminal_A", 2: "Luminal_B", 3: "Her2", 4: "Basal"}

plt.style.use("default")
rcParams["figure.dpi"] = PLOT_DPI
rcParams["savefig.dpi"] = PLOT_DPI


# ============================================================
# HELPERS
# ============================================================
def safe_ttest(group1, group2):
    """Welch's t-test with safe handling of extreme p-values."""
    try:
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False, nan_policy="omit")
        if p_val < 1e-300:
            p_val = 1e-300
        elif p_val == 0:
            p_val = np.finfo(float).tiny
        return t_stat, p_val
    except Exception:
        return np.nan, np.nan


def calculate_effect_size(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return 0 if pooled_std == 0 else (mean1 - mean2) / pooled_std


def create_volcano_plot(results_df, subtype, qval_thresh, effect_thresh, outdir):
    plt.figure(figsize=(10, 8))
    conditions = [
        (results_df["q_value"] < qval_thresh) & (results_df["z_difference"].abs() > effect_thresh),
        (results_df["q_value"] < qval_thresh) & (results_df["z_difference"].abs() <= effect_thresh),
        (results_df["q_value"] >= qval_thresh) & (results_df["z_difference"].abs() > effect_thresh),
        (results_df["q_value"] >= qval_thresh) & (results_df["z_difference"].abs() <= effect_thresh),
    ]
    categories = ["Significant", "Significant (weak effect)", "Non-sig (strong effect)", "Non-significant"]
    colors = ["red", "orange", "blue", "gray"]
    for condition, color, label in zip(conditions, colors, categories):
        subset = results_df[condition]
        if len(subset) > 0:
            plt.scatter(subset["z_difference"], -np.log10(subset["p_value"]),
                        c=color, alpha=0.6, s=20, label=label)
    plt.axhline(-np.log10(qval_thresh), color="black", linestyle="--", alpha=0.8, linewidth=1)
    plt.axvline(effect_thresh, color="black", linestyle="--", alpha=0.8, linewidth=1)
    plt.axvline(-effect_thresh, color="black", linestyle="--", alpha=0.8, linewidth=1)
    plt.xlabel("Z-score Difference (Target - Rest)")
    plt.ylabel("-log10(p-value)")
    plt.title(f"Volcano Plot: {subtype} vs Rest")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{subtype}_vs_Rest_Volcano_Zscore.png"), bbox_inches="tight")
    plt.close()


def create_summary_bar_plot(summary_data, outdir):
    summary_df = pd.DataFrame(summary_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    subtypes = summary_df["stage"]
    x = np.arange(len(subtypes))
    width = 0.25

    ax1.bar(x - width, summary_df["significant_degs"], width, label="Total Significant", alpha=0.8)
    ax1.bar(x, summary_df["positive_effect"], width, label="Positive Effect", alpha=0.8)
    ax1.bar(x + width, summary_df["negative_effect"], width, label="Negative Effect", alpha=0.8)
    ax1.set_xlabel("Subtype"); ax1.set_ylabel("Number of Features")
    ax1.set_title("Significant Features by Subtype")
    ax1.set_xticks(x); ax1.set_xticklabels(subtypes, rotation=45)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.bar(x - width / 2, summary_df["top_positive_selected"], width, label="Top Positive", color="red", alpha=0.8)
    ax2.bar(x + width / 2, summary_df["top_negative_selected"], width, label="Top Negative", color="blue", alpha=0.8)
    ax2.set_xlabel("Subtype"); ax2.set_ylabel("Number of Features")
    ax2.set_title("Top Features Selected")
    ax2.set_xticks(x); ax2.set_xticklabels(subtypes, rotation=45)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Zscore_Analysis_Summary.png"), bbox_inches="tight")
    plt.close()


def create_effect_size_distribution_plot(all_results, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, (subtype, results) in enumerate(all_results.items()):
        if idx >= 4:
            break
        complete_results = results["complete"]
        axes[idx].hist(complete_results["z_difference"], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[idx].axvline(0, color="red", linestyle="--", alpha=0.8, label="Zero effect")
        axes[idx].axvline(EFFECT_SIZE_THRESHOLD, color="orange", linestyle="--", alpha=0.8, label=f"Threshold (±{EFFECT_SIZE_THRESHOLD})")
        axes[idx].axvline(-EFFECT_SIZE_THRESHOLD, color="orange", linestyle="--", alpha=0.8)
        axes[idx].set_xlabel("Z-score Difference"); axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"Effect Size Distribution: {subtype}")
        axes[idx].legend(); axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Effect_Size_Distributions.png"), bbox_inches="tight")
    plt.close()


def run_zscore_differential_analysis(file_path, id_cols, subtype_col, qval_thresh, effect_thresh, top_n, outdir):
    print("Loading batch-corrected pooled matrix...")
    data = pd.read_csv(file_path, index_col=0).reset_index()
    patient_id_col = data.columns[0]
    data[subtype_col] = data[subtype_col].map(SUBTYPE_MAP)
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    clinical_cols_with_id = [patient_id_col] + [c for c in id_cols if c in data.columns]
    gene_columns = [c for c in data.columns if c not in clinical_cols_with_id and c != subtype_col]
    for col in gene_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    gene_columns = [c for c in gene_columns if data[c].notna().any()]
    print(f"Testing {len(gene_columns)} genes across {data[subtype_col].nunique()} subtypes")

    all_results, summary_data = {}, []
    for target_subtype in data[subtype_col].unique():
        group_target = data[data[subtype_col] == target_subtype]
        group_rest = data[data[subtype_col] != target_subtype]
        if len(group_target) < 2 or len(group_rest) < 2:
            continue
        print(f"  {target_subtype} vs Rest  (n={len(group_target)} vs n={len(group_rest)})")

        rows = []
        for gene in gene_columns:
            expr_target = group_target[gene].dropna().values
            expr_rest = group_rest[gene].dropna().values
            if len(expr_target) < 2 or len(expr_rest) < 2:
                continue
            t_stat, p_val = safe_ttest(expr_target, expr_rest)
            if np.isnan(p_val):
                continue
            z_diff = expr_target.mean() - expr_rest.mean()
            rows.append({
                "gene": gene, "mean_target": expr_target.mean(), "mean_rest": expr_rest.mean(),
                "z_difference": z_diff, "cohens_d": calculate_effect_size(expr_target, expr_rest),
                "t_stat": t_stat, "p_value": p_val,
                "effect_direction": "positive" if z_diff > 0 else "negative",
            })
        if not rows:
            continue

        results_df = pd.DataFrame(rows)
        results_df["q_value"] = sm.stats.multipletests(results_df["p_value"].values, method="fdr_bh")[1]
        sig = results_df[(results_df["q_value"] < qval_thresh) & (results_df["z_difference"].abs() > effect_thresh)].copy()
        print(f"    -> {len(sig)} significant DEGs "
              f"({(sig['effect_direction'] == 'positive').sum()} up / {(sig['effect_direction'] == 'negative').sum()} down)")

        results_df.sort_values("p_value").to_csv(
            os.path.join(outdir, f"{target_subtype}_vs_Rest_COMPLETE_ZSCORE.csv"), index=False)

        if len(sig) > 0:
            top_pos = sig[sig["effect_direction"] == "positive"].sort_values("z_difference", ascending=False).head(top_n)
            top_neg = sig[sig["effect_direction"] == "negative"].sort_values("z_difference", ascending=True).head(top_n)
            top_pos.to_csv(os.path.join(outdir, f"{target_subtype}_vs_Rest_TOP{top_n}_POSITIVE.csv"), index=False)
            top_neg.to_csv(os.path.join(outdir, f"{target_subtype}_vs_Rest_TOP{top_n}_NEGATIVE.csv"), index=False)
            combined_top = pd.concat([top_pos, top_neg], axis=0)
            combined_top.to_csv(os.path.join(outdir, f"{target_subtype}_vs_Rest_TOP{top_n*2}_COMBINED_ZSCORE.csv"), index=False)

            all_results[target_subtype] = {"complete": results_df, "top_positive": top_pos, "top_negative": top_neg}
            summary_data.append({
                "stage": target_subtype, "total_features": len(results_df), "significant_degs": len(sig),
                "positive_effect": (sig["effect_direction"] == "positive").sum(),
                "negative_effect": (sig["effect_direction"] == "negative").sum(),
                "top_positive_selected": len(top_pos), "top_negative_selected": len(top_neg),
            })

        create_volcano_plot(results_df, target_subtype, qval_thresh, effect_thresh, outdir)

    if summary_data:
        create_summary_bar_plot(summary_data, outdir)
        create_effect_size_distribution_plot(all_results, outdir)
        pd.DataFrame(summary_data).to_csv(os.path.join(outdir, "ZSCORE_ANALYSIS_SUMMARY.csv"), index=False)
        with open(os.path.join(outdir, "ANALYSIS_PARAMETERS.txt"), "w") as f:
            f.write(f"Q-value threshold: {qval_thresh}\nEffect size threshold: +/-{effect_thresh}\n"
                    f"Top N per direction: {top_n}\nGenes tested: {len(gene_columns)}\n")
    return all_results, summary_data


def combine_top_deg_files(pattern, outdir, output_file, gene_col):
    """Merge all per-subtype TOP-combined CSVs into one deduplicated gene list."""
    all_files = glob.glob(os.path.join(outdir, pattern))
    combined = pd.concat([pd.read_csv(f)[[gene_col]] for f in all_files], axis=0)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    return combined


def build_deg_expression_matrix(deg_list_path, expression_matrix_path, output_path, gene_col="gene"):
    deg_genes = pd.read_csv(deg_list_path)[gene_col].astype(str).tolist()
    expr_df = pd.read_csv(expression_matrix_path, index_col=0)
    meta_cols = ["PATIENT_ID", "AGE", "SEX", "OS_STATUS", "OS_MONTHS", "SUBTYPE", "TUMOR_STAGE"]
    existing_meta = [c for c in meta_cols if c in expr_df.columns]
    meta_df = expr_df[existing_meta]
    common_genes = [g for g in deg_genes if g in expr_df.columns]
    final_matrix = pd.concat([meta_df, expr_df[common_genes]], axis=1)
    final_matrix.to_csv(output_path, index=True)
    print(f"DEG expression matrix saved: {output_path}  (shape={final_matrix.shape}, "
          f"{len(common_genes)}/{len(deg_genes)} DEGs found)")
    return final_matrix


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: One-vs-rest Welch's t-test DEG analysis")
    print("=" * 70)
    run_zscore_differential_analysis(
        file_path=INPUT_FILE, id_cols=ID_COLUMNS, subtype_col=SUBTYPE_COLUMN,
        qval_thresh=QVAL_THRESHOLD, effect_thresh=EFFECT_SIZE_THRESHOLD,
        top_n=TOP_N_GENES, outdir=OUTPUT_DIR,
    )

    print("\n" + "=" * 70)
    print("STEP 2: Combine + deduplicate top DEGs across subtypes")
    print("=" * 70)
    dedup_path = os.path.join(STAGE_DIR, "final_combined_top500_DEGs.csv")
    dedup_genes = combine_top_deg_files("*_TOP100_COMBINED_ZSCORE.csv", OUTPUT_DIR, dedup_path, gene_col="gene")
    print(f"Deduplicated DEG candidate set: {len(dedup_genes)} genes -> {dedup_path}")

    print("\n" + "=" * 70)
    print("STEP 3: Build 426-gene DEG expression matrix")
    print("=" * 70)
    build_deg_expression_matrix(
        deg_list_path=dedup_path,
        expression_matrix_path=INPUT_FILE,
        output_path=os.path.join(STAGE_DIR, "SUBTYPE_DEG_expression_matrix.csv"),
        gene_col="gene",
    )
