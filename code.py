import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = r"D:\MultiModal_Classification"
INPUT_DIR  = os.path.join(BASE_DIR, "input", "raw_cohorts", "tcga2012")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "01_preprocessing", "tcga2012")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPR_FILE      = os.path.join(INPUT_DIR, "data_mrna_agilent_microarray_zscores_ref_all_samples.csv")
CLINICAL_PATIENT_FILE = os.path.join(INPUT_DIR, "data_clinical_patient.csv")
CLINICAL_SAMPLE_FILE  = os.path.join(INPUT_DIR, "data_clinical_sample.csv")

MISSING_COL_KEEP_THRESHOLD = 0.70   # keep a gene column only if >=70% samples have a value

# ============================================================
# STEP 1 — EXPRESSION MATRIX: transpose, clean, impute
# ============================================================
print("[1/3] Loading and cleaning TCGA-2012 expression matrix...")

expr_raw = pd.read_csv(EXPR_FILE)
expr = expr_raw.transpose()
expr.columns = expr.iloc[0]
expr = expr.drop(expr.index[0])
expr.insert(0, "SAMPLE_ID", expr.index)
expr = expr.loc[:, expr.columns.notna()]          # drop columns with NaN gene names

# drop genes missing in >30% of samples
required_non_null = int(MISSING_COL_KEEP_THRESHOLD * expr.shape[0])
expr = expr.dropna(axis=1, thresh=required_non_null)

# median-impute remaining missing values (gene columns only)
expr = expr.reset_index(drop=True)
sample_ids = expr["SAMPLE_ID"]
gene_df = expr.drop(columns=["SAMPLE_ID"]).apply(pd.to_numeric, errors="coerce")
gene_df = gene_df.fillna(gene_df.median())
expr = pd.concat([sample_ids, gene_df], axis=1)

print(f"    Expression matrix ready: {expr.shape[0]} samples x {expr.shape[1] - 1} genes")

# ============================================================
# STEP 2 — CLINICAL DATA: merge, clean, encode
# ============================================================
print("[2/3] Loading and encoding TCGA-2012 clinical data...")

clinical_patient = pd.read_csv(CLINICAL_PATIENT_FILE)
clinical_sample = pd.read_csv(CLINICAL_SAMPLE_FILE)
clinical = pd.merge(clinical_patient, clinical_sample, on="PATIENT_ID", how="inner")

DROP_COLS = [
    "METASTASIS", "ER_STATUS", "PR_STATUS", "HER2_STATUS", "TUMOR_T1_CODED",
    "NODES", "NODE_CODED", "METASTASIS_CODED", "CONVERTED_STAGE",
    "SURVIVAL_DATA_FORM", "SIGCLUST_UNSUPERVISED_MRNA", "SIGCLUST_INTRINSIC_MRNA",
    "MIRNA_CLUSTER", "METHYLATION_CLUSTER", "RPPA_CLUSTER", "CN_CLUSTER",
    "INTEGRATED_CLUSTERS_WITH_PAM50", "INTEGRATED_CLUSTERS_NO_EXP",
    "INTEGRATED_CLUSTERS_UNSUP_EXP", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE",
    "SOMATIC_STATUS", "TMB_NONSYNONYMOUS", "SAMPLE_TYPE", "CANCER_TYPE",
]
clinical = clinical.drop(columns=DROP_COLS, errors="ignore")
clinical = clinical.rename(columns={"PAM50_SUBTYPE": "SUBTYPE"})
clinical = clinical.drop_duplicates()

# drop rows missing the two required labels, then encode
clinical = clinical.dropna(subset=["TUMOR_STAGE", "SUBTYPE"])

clinical.to_csv(os.path.join(OUTPUT_DIR, "merged_clinical_2012_Harmonised.csv"), index=False)

SUBTYPE_MAP = {"Normal-like": 0, "Luminal A": 1, "Luminal B": 2, "HER2-enriched": 3, "Basal-like": 4}
clinical["SUBTYPE"] = clinical["SUBTYPE"].map(SUBTYPE_MAP)

def encode_stage_tcga2012(stage):
    if pd.isna(stage):
        return None
    stage = str(stage).upper().strip()
    if stage in ["T1", "TX"]:
        return 0
    if stage == "T2":
        return 1
    if stage == "T3":
        return 2
    if stage == "T4":
        return 3
    return None

clinical["TUMOR_STAGE"] = clinical["TUMOR_STAGE"].apply(encode_stage_tcga2012)
clinical["OS_STATUS"] = clinical["OS_STATUS"].map({"0:LIVING": 0, "1:DECEASED": 1})
clinical["SEX"] = clinical["SEX"].map({"Female": 0, "Male": 1})

print(f"    Clinical table ready: {clinical.shape[0]} patients")

# ============================================================
# STEP 3 — MERGE EXPRESSION + CLINICAL, SAVE
# ============================================================
print("[3/3] Merging expression with clinical data and saving...")

merged = clinical.merge(expr, on="SAMPLE_ID", how="inner")
merged = merged.drop(columns=["SAMPLE_ID"])
merged = merged[["PATIENT_ID"] + [c for c in merged.columns if c != "PATIENT_ID"]]
merged = merged.drop_duplicates()

out_path = os.path.join(OUTPUT_DIR, "merged_df_2012.csv")
merged.to_csv(out_path, index=True)

print(f"\nDone. TCGA-2012 preprocessing complete.")
print(f"Final shape: {merged.shape[0]} patients x {merged.shape[1]} columns")
print(f"Saved to: {out_path}")

# Fullcode will be shared upon request
