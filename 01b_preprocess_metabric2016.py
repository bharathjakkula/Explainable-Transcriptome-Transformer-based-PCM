import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = r"D:\MultiModal_Classification"
INPUT_DIR  = os.path.join(BASE_DIR, "input", "raw_cohorts", "metabric2016")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "01_preprocessing", "metabric2016")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPR_FILE = os.path.join(INPUT_DIR, "data_mrna_illumina_microarray_zscores_ref_all_samples.csv")
CLINICAL_PATIENT_FILE = os.path.join(INPUT_DIR, "data_clinical_patient.csv")
CLINICAL_SAMPLE_FILE  = os.path.join(INPUT_DIR, "data_clinical_sample.csv")

MISSING_COL_KEEP_THRESHOLD = 0.70

# ============================================================
# STEP 1 — EXPRESSION MATRIX: transpose, clean, impute
# ============================================================
print("[1/3] Loading and cleaning METABRIC-2016 expression matrix...")

expr_raw = pd.read_csv(EXPR_FILE)
expr = expr_raw.transpose()
expr.columns = expr.iloc[0]
expr = expr.drop(expr.index[0])
expr.insert(0, "SAMPLE_ID", expr.index)
expr = expr.loc[:, expr.columns.notna()]

required_non_null = int(MISSING_COL_KEEP_THRESHOLD * expr.shape[0])
expr = expr.dropna(axis=1, thresh=required_non_null)

expr = expr.reset_index(drop=True)
sample_ids = expr["SAMPLE_ID"].astype(str).str.strip().str.upper()
gene_df = expr.drop(columns=["SAMPLE_ID"]).apply(pd.to_numeric, errors="coerce")
gene_df = gene_df.fillna(gene_df.median())
expr = pd.concat([sample_ids, gene_df], axis=1)

print(f"    Expression matrix ready: {expr.shape[0]} samples x {expr.shape[1] - 1} genes")

# ============================================================
# STEP 2 — CLINICAL DATA: merge, clean, encode
# ============================================================
print("[2/3] Loading and encoding METABRIC-2016 clinical data...")

clinical_patient = pd.read_csv(CLINICAL_PATIENT_FILE)
clinical_sample = pd.read_csv(CLINICAL_SAMPLE_FILE)
clinical = pd.merge(clinical_patient, clinical_sample, on="PATIENT_ID", how="inner")

KEEP_COLS = [
    "PATIENT_ID", "SAMPLE_ID", "AGE_AT_DIAGNOSIS", "SEX",
    "OS_STATUS", "OS_MONTHS", "p_SUBTYPE", "TUMOR_STAGE",
    "GRADE", "TUMOR_SIZE",
]
clinical = clinical[[c for c in KEEP_COLS if c in clinical.columns]].copy()
clinical = clinical.rename(columns={"AGE_AT_DIAGNOSIS": "AGE", "p_SUBTYPE": "SUBTYPE"})
clinical = clinical.drop_duplicates()

# drop rows missing the two required labels, then encode
clinical = clinical.dropna(subset=["TUMOR_STAGE", "SUBTYPE"])

clinical.to_csv(os.path.join(OUTPUT_DIR, "merged_clinical_2016_Harmonised.csv"), index=False)

SUBTYPE_MAP = {"Normal": 0, "LumA": 1, "LumB": 2, "Her2": 3, "Basal": 4}
clinical["SUBTYPE"] = clinical["SUBTYPE"].map(SUBTYPE_MAP)

def encode_stage_metabric(stage):
    if pd.isna(stage):
        return None
    stage = str(stage).upper().strip()
    if stage in ["0.0", "0", "1.0", "1"]:
        return 0
    if stage in ["2.0", "2"]:
        return 1
    if stage in ["3.0", "3"]:
        return 2
    if stage in ["4.0", "4"]:
        return 3
    return None

clinical["TUMOR_STAGE"] = clinical["TUMOR_STAGE"].apply(encode_stage_metabric)
clinical["OS_STATUS"] = clinical["OS_STATUS"].map({"0:LIVING": 0, "1:DECEASED": 1})
clinical["SEX"] = clinical["SEX"].map({"Female": 0, "Male": 1})

# drop rows where SUBTYPE became NaN after mapping (NC / unmapped)
clinical = clinical.dropna(subset=["SUBTYPE"])

print(f"    Clinical table ready: {clinical.shape[0]} patients")

# ============================================================
# STEP 3 — MERGE EXPRESSION + CLINICAL, SAVE
# ============================================================
print("[3/3] Merging expression with clinical data and saving...")

clinical["SAMPLE_ID"] = clinical["SAMPLE_ID"].astype(str).str.strip().str.upper()
merged = clinical.merge(expr, on="SAMPLE_ID", how="inner")
merged = merged.drop(columns=["SAMPLE_ID"])
merged = merged[["PATIENT_ID"] + [c for c in merged.columns if c != "PATIENT_ID"]]
merged = merged.drop_duplicates()

out_path = os.path.join(OUTPUT_DIR, "merged_df_2016.csv")
merged.to_csv(out_path, index=True)

print(f"\nDone. METABRIC-2016 preprocessing complete.")
print(f"Final shape: {merged.shape[0]} patients x {merged.shape[1]} columns")
print(f"Saved to: {out_path}")
