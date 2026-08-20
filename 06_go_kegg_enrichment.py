# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"D:\MultiModal_Classification"
INPUT_FILE = os.path.join(BASE_DIR, "output", "02_batch_correction_deg_pca", "final_combined_top50_PCA_DEGs_subtype.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "06_enrichment")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENE_COL = "Gene"   # column name in the 50-gene signature CSV (see 02c output)
CUTOFF = 0.05

GO_SETS = {
    "BP": "GO_Biological_Process_2023",
    "CC": "GO_Cellular_Component_2023",
    "MF": "GO_Molecular_Function_2023",
}
KEGG_SET = "KEGG_2021_Human"


def main():
    df = pd.read_csv(INPUT_FILE)
    gene_list = df[GENE_COL].dropna().astype(str).tolist()
    print(f"Loaded {len(gene_list)} signature genes for enrichment.")

    for go_type, lib in GO_SETS.items():
        gp.enrichr(gene_list=gene_list, gene_sets=lib, organism="Human",
                   outdir=os.path.join(OUTPUT_DIR, f"GO_{go_type}"), cutoff=CUTOFF)
        print(f"GO-{go_type} ({lib}) enrichment completed.")

    gp.enrichr(gene_list=gene_list, gene_sets=KEGG_SET, organism="Human",
               outdir=os.path.join(OUTPUT_DIR, "KEGG"), cutoff=CUTOFF)
    print(f"KEGG ({KEGG_SET}) enrichment completed.")

    print(f"\nDone. All enrichment results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
