import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = r"D:\MultiModal_Classification"
INPUT_DIR  = os.path.join(BASE_DIR, "input", "raw_cohorts", "tcga2018")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "01_preprocessing", "tcga2018")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPR_FILE = os.path.join(INPUT_DIR, "data_mrna_seq_v2_rsem_zscores_ref_all_samples.csv")
CLINICAL_PATIENT_FILE   = os.path.join(INPUT_DIR, "data_clinical_patient.csv")
CLINICAL_SAMPLE_FILE    = os.path.join(INPUT_DIR, "data_clinical_sample.csv")
CLINICAL_TREATMENT_FILE = os.path.join(INPUT_DIR, "data_timeline_treatment.csv")

MISSING_COL_KEEP_THRESHOLD = 0.70

# ============================================================
# STEP 1 — EXPRESSION MATRIX: transpose, standardize PATIENT_ID, clean, impute
# ============================================================
print("[1/3] Loading and cleaning TCGA-2018 expression matrix...")

try:
    expr_raw = pd.read_csv(EXPR_FILE, sep=",", encoding="utf-8-sig", index_col=0)
except Exception:
    expr_raw = pd.read_csv(EXPR_FILE, sep=r"\s+", encoding="utf-8-sig", index_col=0)

expr_raw.index.name = "Gene_ID"
expr = expr_raw.T.reset_index()
expr = expr.rename(columns={expr.columns[0]: "PATIENT_ID"})

# TCGA-2018 barcodes carry sample-level suffixes (e.g. -01); truncate to the
# 3-segment patient-level barcode (TCGA-XX-XXXX) for merging with clinical data
expr["PATIENT_ID"] = (
    expr["PATIENT_ID"].astype(str).str.strip().str.upper()
    .str.split("-").str[:3].str.join("-")
)

expr = expr.loc[:, expr.columns.notna()]

required_non_null = int(MISSING_COL_KEEP_THRESHOLD * expr.shape[0])
expr = expr.dropna(axis=1, thresh=required_non_null)

patient_ids = expr["PATIENT_ID"]
gene_df = expr.drop(columns=["PATIENT_ID"]).apply(pd.to_numeric, errors="coerce")
gene_df = gene_df.fillna(gene_df.median(numeric_only=True))
expr = pd.concat([patient_ids, gene_df], axis=1)

print(f"    Expression matrix ready: {expr.shape[0]} samples x {expr.shape[1] - 1} genes")

# ============================================================
# STEP 2 — CLINICAL DATA: merge, clean, encode
# ============================================================
print("[2/3] Loading and encoding TCGA-2018 clinical data...")

clinical_patient   = pd.read_csv(CLINICAL_PATIENT_FILE)
clinical_treatment = pd.read_csv(CLINICAL_TREATMENT_FILE)
clinical_sample     = pd.read_csv(CLINICAL_SAMPLE_FILE)

clinical = pd.merge(clinical_patient, clinical_treatment, on="PATIENT_ID", how="inner")
clinical = pd.merge(clinical, clinical_sample, on="PATIENT_ID", how="inner")
clinical = clinical.rename(columns={"AJCC_PATHOLOGIC_TUMOR_STAGE": "TUMOR_STAGE"})

DROP_COLS = [
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT", "PERSON_NEOPLASM_CANCER_STATUS",
    "PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT", "RADIATION_THERAPY", "WEIGHT",
    "DFS_STATUS", "DFS_MONTHS", "PFS_STATUS", "PFS_MONTHS",
    "GENETIC_ANCESTRY_LABEL", "TREATMENT_TYPE", "AGENT",
    "ROUTE_OF_ADMINISTRATION", "RADIATION_DOSAGE", "DSS_STATUS", "DSS_MONTHS",
]
clinical = clinical.drop(columns=DROP_COLS, errors="ignore")
clinical = clinical.drop_duplicates()

label_columns = ["TUMOR_STAGE", "SUBTYPE"]
clinical = clinical.dropna(subset=[c for c in label_columns if c in clinical.columns])

clinical.to_csv(os.path.join(OUTPUT_DIR, "merged_clinical_2018_Harmonised.csv"), index=False)

SUBTYPE_MAP = {"BRCA_Normal": 0, "BRCA_LumA": 1, "BRCA_LumB": 2, "BRCA_Her2": 3, "BRCA_Basal": 4}
if "SUBTYPE" in clinical.columns:
    clinical["SUBTYPE"] = clinical["SUBTYPE"].map(SUBTYPE_MAP)

def encode_stage_tcga2018(stage):
    if pd.isna(stage):
        return None
    stage = str(stage).upper().strip()
    if stage in ["STAGE I", "STAGE IA", "STAGE IB", "STAGE X"]:
        return 0
    if stage in ["STAGE II", "STAGE IIA", "STAGE IIB"]:
        return 1
    if stage in ["STAGE III", "STAGE IIIA", "STAGE IIIB", "STAGE IIIC"]:
        return 2
    if stage == "STAGE IV":
        return 3
    return None

clinical["TUMOR_STAGE"] = clinical["TUMOR_STAGE"].apply(encode_stage_tcga2018)
clinical["OS_STATUS"] = clinical["OS_STATUS"].map({"0:LIVING": 0, "1:DECEASED": 1})
clinical["SEX"] = clinical["SEX"].map({"Female": 0, "Male": 1})

print(f"    Clinical table ready: {clinical.shape[0]} patients")

# ============================================================
# STEP 3 — MERGE EXPRESSION + CLINICAL, SAVE
# ============================================================
print("[3/3] Merging expression with clinical data and saving...")

clinical["PATIENT_ID"] = (
    clinical["PATIENT_ID"].astype(str).str.strip().str.upper()
    .str.split("-").str[:3].str.join("-")
)
merged = clinical.merge(expr, on="PATIENT_ID", how="inner")
merged = merged.drop(columns=["SAMPLE_ID"], errors="ignore")
merged = merged[["PATIENT_ID"] + [c for c in merged.columns if c != "PATIENT_ID"]]
merged = merged.drop_duplicates()

out_path = os.path.join(OUTPUT_DIR, "merged_df_2018.csv")
merged.to_csv(out_path, index=True)

print(f"\nDone. TCGA-2018 preprocessing complete.")
print(f"Final shape: {merged.shape[0]} patients x {merged.shape[1]} columns")
print(f"Saved to: {out_path}")
