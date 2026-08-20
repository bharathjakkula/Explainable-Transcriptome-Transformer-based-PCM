import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS & PARAMETERS
# ============================================================
BASE_DIR = r"D:\MultiModal_Classification"
INPUT_FILE = os.path.join(BASE_DIR, "output", "05_shap", "mean_shap_per_gene_per_class.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "05_shap", "specificity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBTYPES = ["Normal", "LuminalA", "LuminalB", "Her2", "Basal"]
MIN_SHAP_ABSOLUTE = 0.00020
RELATIVE_FACTOR = 1.0
TOP_N_PER_SUBTYPE = 20


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0)
    print(f"Loaded SHAP matrix: {df.shape}")

    enriched_dict = {}
    all_enriched_genes = []

    for target in SUBTYPES:
        if target not in df.columns:
            continue
        others = [s for s in SUBTYPES if s != target]
        max_others = df[others].abs().max(axis=1)

        high_shap = df[target].abs() >= MIN_SHAP_ABSOLUTE
        enriched = df[target].abs() >= RELATIVE_FACTOR * max_others
        both = high_shap & enriched

        print(f"{target:10} | |SHAP|>={MIN_SHAP_ABSOLUTE}: {high_shap.sum():3d} | "
              f"Enriched >= {RELATIVE_FACTOR}x max_others: {enriched.sum():3d} | Both: {both.sum():3d}")

        if both.sum() > 0:
            enriched_df = df.loc[both].copy()
            enriched_df["max_other"] = max_others[both]
            enriched_df = enriched_df.sort_values(by=target, ascending=False).head(TOP_N_PER_SUBTYPE)
            enriched_dict[target] = enriched_df
            all_enriched_genes.append(enriched_df)

    df.to_csv(os.path.join(OUTPUT_DIR, "shap_matrix_with_analysis.csv"))

    for subtype, df_sub in enriched_dict.items():
        df_sub.to_csv(os.path.join(OUTPUT_DIR, f"enriched_genes_{subtype}.csv"))
        print(f"Saved {len(df_sub)} enriched genes for {subtype}")

    if all_enriched_genes:
        combined = pd.concat(all_enriched_genes)
        unique = combined[~combined.index.duplicated(keep="first")]
        unique.to_csv(os.path.join(OUTPUT_DIR, "all_unique_enriched_genes.csv"))
        print(f"Total unique enriched genes across all subtypes: {len(unique)}")
        her2_basal = pd.concat([enriched_dict.get("Her2", pd.DataFrame()), enriched_dict.get("Basal", pd.DataFrame())])
        her2_basal = her2_basal[~her2_basal.index.duplicated(keep="first")]
        print(f"HER2 + Basal-like candidate target genes: {len(her2_basal)} "
              f"(manuscript reports 16 HER2-specific + 14 Basal-specific = 30)")

    # Heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[SUBTYPES], cmap="viridis", linewidths=0.1, cbar_kws={"label": "Mean |SHAP|"})
    plt.title("SHAP Values Across Subtypes")
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_all_shap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if all_enriched_genes:
        enriched_genes = list(set().union(*[d.index for d in enriched_dict.values()]))
        if enriched_genes:
            plt.figure(figsize=(10, max(6, len(enriched_genes) * 0.4)))
            sns.heatmap(df.loc[enriched_genes, SUBTYPES], cmap="viridis", annot=True, fmt=".5f",
                        linewidths=0.5, cbar_kws={"label": "Mean |SHAP|"})
            plt.title("SHAP Values - Enriched Genes per Subtype")
            plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_enriched_genes.png"), dpi=300, bbox_inches="tight")
            plt.close()

    if enriched_dict:
        counts = {k: len(v) for k, v in enriched_dict.items()}
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
        plt.title(f"Enriched Genes per Subtype (>= {RELATIVE_FACTOR}x max other, |SHAP| >= {MIN_SHAP_ABSOLUTE})")
        plt.ylabel("Number of genes")
        plt.savefig(os.path.join(OUTPUT_DIR, "bar_enriched_counts.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nDone. SHAP specificity screen complete. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
