import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR   = r"D:\MultiModal_Classification"
DATA_DIR   = os.path.join(BASE_DIR, "output", "08_pcm_features", "chembl")
CKPT_DIR   = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 08a output
PROT_CSV   = os.path.join(BASE_DIR, "output", "08_pcm_features", "fasta", "target_fasta_sequences.csv")
OUTPUT_CSV = os.path.join(DATA_DIR,  "PCM_dataset_clean.csv")
DIST_CSV   = os.path.join(DATA_DIR,  "pchembl_distribution.csv")
AUDIT_CSV  = os.path.join(DATA_DIR,  "target_audit.csv")

CKPT_RAW    = os.path.join(CKPT_DIR, "step2_raw_activities.csv")
CKPT_AGG    = os.path.join(CKPT_DIR, "step3_aggregated.csv")
CKPT_SMILES = os.path.join(CKPT_DIR, "step4_with_smiles.csv")
CKPT_CLEAN  = os.path.join(CKPT_DIR, "step5_rdkit_clean.csv")

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS  = {"Accept": "application/json"}
SLEEP    = 0.05
MIN_SAMPLES_PER_TARGET = 20

TARGET_UNIPROT_IDS = {
    "FOXA1":  "P55317",
    "CA12":   "O43570",
    "CDH3":   "P22223",
    "ESR1":   "P03372",
    "GATA3":  "P23771",
    "XBP1":   "P17861",
    "FOXC1":  "Q12948",
    "ERBB2":  "P04626",
    "MED1":   "Q15648",
    "CIRBP":  "Q14011",
    "GRB7":   "Q14451",
    "CCNE1":  "P24864",
    "CDCA8":  "Q53HL2",
    "TOP2A":  "P11388",
    "TTK":    "P33981",
    "MELK":   "Q14680",
    "EGFR":   "P00533",
    "NUSAP1": "Q9BXS6",
    "CENPA":  "P49450",
    "CDC20":  "Q12834",
    "TTLL4":  "Q14679",
    "E2F3":   "O00716",
    "KRT5":   "P13647",
    "G6PD":   "P11413",
    "DNALI1": "O14645",
    "PROS1":  "P07225",
    "PIK3CA": "P42336",
    "KMO":    "O15229",
    "FGFR4":  "P22455",
    "STARD3": "Q14849",
}
UNIPROT_TO_NAME = {v: k for k, v in TARGET_UNIPROT_IDS.items()}

# ============================================================
# SESSION WITH RETRIES
# ============================================================

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)))

# ============================================================
# STEP 1: UniProt → ChEMBL SINGLE PROTEIN target IDs 
# ============================================================

def get_single_protein_targets(uniprot_id):
    """
    Return ChEMBL target IDs for a UniProt accession.
    STRICT: target_type=SINGLE PROTEIN only.
    Returns list of chembl_target_id strings.
    """
    url    = f"{BASE_URL}/target.json"
    params = {
        "target_components__accession": uniprot_id,
        "target_type":                  "SINGLE PROTEIN",   # ← KEY CHANGE
        "limit":                        100,
    }
    r = session.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    targets = r.json().get("targets", [])
    return [(t["target_chembl_id"],
             t.get("pref_name", "unknown"))
            for t in targets]

print("=" * 60)
print("STEP 1: Mapping UniProt IDs → ChEMBL SINGLE PROTEIN targets")
print("        (protein complexes excluded)")
print("=" * 60)

uniprot_to_chembl = {}   # uid → list of chembl_target_ids
audit_rows        = []   # one row per target for audit table

for name, uid in TARGET_UNIPROT_IDS.items():
    hits   = get_single_protein_targets(uid)
    tids   = [h[0] for h in hits]
    uniprot_to_chembl[uid] = tids

    status = "found" if tids else "no_single_protein_target"
    print(f"  {name:8s} ({uid}) → {tids if tids else '[NONE — no single-protein target]'}")

    audit_rows.append({
        "target_name":       name,
        "uniprot_id":        uid,
        "n_chembl_targets":  len(tids),
        "chembl_target_ids": ";".join(tids),
        "step1_status":      status,
    })

# ============================================================
# STEP 2: Fetch activity data (with checkpoint)
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Fetching activity data from ChEMBL")
print("        (assay_type=B, pChEMBL not null)")
print("=" * 60)

if os.path.exists(CKPT_RAW):
    print(f"  Checkpoint found — loading saved data")
    df_raw = pd.read_csv(CKPT_RAW)
    print(f"  Loaded {len(df_raw)} records")
else:
    records = []
    for uid, chembl_target_ids in uniprot_to_chembl.items():
        tname = UNIPROT_TO_NAME[uid]

        if not chembl_target_ids:
            print(f"  Skipping {tname} — no ChEMBL single-protein targets")
            continue

        print(f"\n  Fetching {tname} ({uid})...")
        for tid in chembl_target_ids:
            offset = 0
            while True:
                r = session.get(
                    f"{BASE_URL}/activity.json",
                    params={
                        "target_chembl_id":      tid,
                        "assay_type":            "B",
                        "pchembl_value__isnull": "false",
                        "limit":                 1000,
                        "offset":                offset,
                    },
                    headers=HEADERS, timeout=30
                )
                if not r.ok:
                    break
                data       = r.json()
                activities = data.get("activities", [])
                if not activities:
                    break
                for a in activities:
                    records.append({
                        "molecule_chembl_id": a["molecule_chembl_id"],
                        "uniprot_id":         uid,
                        "target_name":        tname,
                        "pchembl_value":      a["pchembl_value"],
                    })
                if data.get("page_meta", {}).get("next") is None:
                    break
                offset += 1000
                time.sleep(SLEEP)

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(CKPT_RAW, index=False)
    print(f"\n  Raw records fetched: {len(df_raw)}")

# Update audit with raw record counts
raw_counts = df_raw.groupby("uniprot_id").size().to_dict() if len(df_raw) else {}
for row in audit_rows:
    row["n_raw_activities"] = raw_counts.get(row["uniprot_id"], 0)

print("\n  Raw record counts per target:")
print(df_raw.groupby("target_name").size().to_string() if len(df_raw) else "  (no data)")

# ============================================================
# STEP 3: Median aggregation (with checkpoint)
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Aggregating duplicate compound-target pairs by median")
print("=" * 60)

if os.path.exists(CKPT_AGG):
    print(f"  Checkpoint found — loading saved data")
    df_agg = pd.read_csv(CKPT_AGG)
else:
    df_raw["pchembl_value"] = pd.to_numeric(df_raw["pchembl_value"], errors="coerce")
    df_raw = df_raw.dropna(subset=["pchembl_value"])
    df_agg = (
        df_raw
        .groupby(["molecule_chembl_id", "uniprot_id", "target_name"], as_index=False)
        .agg({"pchembl_value": "median"})
    )
    df_agg.to_csv(CKPT_AGG, index=False)

print(f"  After aggregation: {len(df_agg)} rows")

agg_counts = df_agg.groupby("uniprot_id").size().to_dict() if len(df_agg) else {}
for row in audit_rows:
    row["n_after_aggregation"] = agg_counts.get(row["uniprot_id"], 0)

# ============================================================
# STEP 4: Bulk SMILES fetch (with checkpoint)
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Fetching SMILES (bulk, up to 1000 IDs per request)")
print("=" * 60)

if os.path.exists(CKPT_SMILES):
    print(f"  Checkpoint found — loading saved data")
    df_agg = pd.read_csv(CKPT_SMILES)
else:
    unique_ids = df_agg["molecule_chembl_id"].unique().tolist()
    BULK_SIZE  = 1000
    smiles_map = {}

    print(f"  Unique compounds: {len(unique_ids)}")
    print(f"  Bulk requests needed: {len(unique_ids) // BULK_SIZE + 1}")

    for i in tqdm(range(0, len(unique_ids), BULK_SIZE), desc="  Bulk SMILES"):
        batch_ids = unique_ids[i: i + BULK_SIZE]
        id_string = ",".join(batch_ids)
        try:
            r = session.get(
                f"{BASE_URL}/molecule.json",
                params={
                    "molecule_chembl_id__in": id_string,
                    "limit":                  BULK_SIZE,
                },
                headers=HEADERS,
                timeout=120,
            )
            if r.ok:
                for mol in r.json().get("molecules", []):
                    cid = mol.get("molecule_chembl_id")
                    smi = (mol.get("molecule_structures") or {}).get("canonical_smiles")
                    smiles_map[cid] = smi
            else:
                print(f"\n  Bulk request failed (HTTP {r.status_code}), "
                      f"falling back to individual for batch {i//BULK_SIZE}")
                for cid in batch_ids:
                    try:
                        r2 = session.get(
                            f"{BASE_URL}/molecule/{cid}.json",
                            headers=HEADERS, timeout=10
                        )
                        if r2.ok:
                            smi = (r2.json().get("molecule_structures") or {}).get(
                                "canonical_smiles"
                            )
                            smiles_map[cid] = smi
                    except Exception:
                        smiles_map[cid] = None
                    time.sleep(SLEEP)
        except Exception as e:
            print(f"\n  Warning: batch {i//BULK_SIZE} error ({e}), skipping")
        time.sleep(SLEEP)

    df_agg["smiles"] = df_agg["molecule_chembl_id"].map(smiles_map)
    df_agg = df_agg.dropna(subset=["smiles"])
    df_agg.to_csv(CKPT_SMILES, index=False)

print(f"  After SMILES fetch: {len(df_agg)} rows")

# ============================================================
# STEP 5: RDKit SMILES cleaning (with checkpoint)
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Cleaning SMILES with RDKit")
print("=" * 60)

if os.path.exists(CKPT_CLEAN):
    print(f"  Checkpoint found — loading saved data")
    df_agg = pd.read_csv(CKPT_CLEAN)
else:
    remover = SaltRemover.SaltRemover()

    def clean_smiles(smi):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            mol   = remover.StripMol(mol, dontRemoveEverything=True)
            frags = Chem.GetMolFrags(mol, asMols=True)
            if not frags:
                return None
            largest = max(frags, key=lambda m: m.GetNumAtoms())
            return Chem.MolToSmiles(largest, canonical=True)
        except Exception:
            return None

    df_agg["smiles"] = df_agg["smiles"].apply(clean_smiles)
    df_agg = df_agg.dropna(subset=["smiles"])
    df_agg.to_csv(CKPT_CLEAN, index=False)

print(f"  After RDKit cleaning: {len(df_agg)} rows")

# ============================================================
# STEP 6: Deduplicate at (smiles, uniprot_id) level
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Deduplication")
print("=" * 60)

df_agg = (
    df_agg
    .sort_values("pchembl_value", ascending=False)
    .drop_duplicates(subset=["smiles", "uniprot_id"])
    .reset_index(drop=True)
)
print(f"  After deduplication: {len(df_agg)} rows")

dedup_counts = df_agg.groupby("uniprot_id").size().to_dict()
for row in audit_rows:
    row["n_after_dedup"] = dedup_counts.get(row["uniprot_id"], 0)

# ============================================================
# STEP 7: Filter targets with too few samples 
# ============================================================

print("\n" + "=" * 60)
print(f"STEP 7: Filtering targets < {MIN_SAMPLES_PER_TARGET} compounds")
print("        + Audit: which targets enter/skip training & why")
print("=" * 60)

counts     = df_agg.groupby("uniprot_id").size()
valid_uids = counts[counts >= MIN_SAMPLES_PER_TARGET].index.tolist()

# Assign reason + entered_training flag to every audit row
for row in audit_rows:
    uid = row["uniprot_id"]
    n_chembl    = row.get("n_chembl_targets",    0)
    n_raw       = row.get("n_raw_activities",    0)
    n_dedup     = row.get("n_after_dedup",       0)

    if uid in valid_uids:
        row["entered_training"] = True
        row["exclusion_reason"] = "—"
    else:
        row["entered_training"] = False
        if n_chembl == 0:
            row["exclusion_reason"] = (
                "REASON A: No ChEMBL single-protein target found. "
                "UniProt ID may only appear under protein complex "
                "targets in ChEMBL. Protein may be undruggable or "
                "not yet systematically screened as a monomer."
            )
        elif n_raw == 0:
            row["exclusion_reason"] = (
                "REASON B: ChEMBL single-protein target exists but "
                "zero binding assay (type=B) records have a pChEMBL "
                "value. Activity data may exist for functional or "
                "ADMET assays but not for binding affinity."
            )
        else:
            row["exclusion_reason"] = (
                f"REASON C: Data insufficient — only {n_dedup} unique "
                f"compounds after cleaning (minimum required: "
                f"{MIN_SAMPLES_PER_TARGET}). Too few data points for "
                f"reliable model training."
            )

# Apply filter
df_agg = df_agg[df_agg["uniprot_id"].isin(valid_uids)].reset_index(drop=True)
print(f"  After filtering: {len(df_agg)} rows")

# ── Targets ENTERING training ──────────────────────────────
print("\n  ✔  TARGETS ENTERING TRAINING:")
print(f"  {'Target':<10} {'UniProt':<10} {'Compounds':>10}")
print("  " + "-" * 32)
for row in audit_rows:
    if row["entered_training"]:
        n = counts.get(row["uniprot_id"], 0)
        print(f"  {row['target_name']:<10} {row['uniprot_id']:<10} {n:>10}")

# ── Targets EXCLUDED from training ────────────────────────
print("\n  ✘  TARGETS EXCLUDED FROM TRAINING:")
print(f"  {'Target':<10} {'UniProt':<10} {'RawActs':>8} "
      f"{'AfterClean':>11}  Reason")
print("  " + "-" * 75)
for row in audit_rows:
    if not row["entered_training"]:
        print(f"  {row['target_name']:<10} {row['uniprot_id']:<10} "
              f"{row.get('n_raw_activities', 0):>8} "
              f"{row.get('n_after_dedup', 0):>11}  "
              f"{row['exclusion_reason']}")

# ── Save audit CSV ─────────────────────────────────────────
audit_col_order = [
    "target_name", "uniprot_id",
    "n_chembl_targets", "chembl_target_ids", "step1_status",
    "n_raw_activities", "n_after_aggregation",
    "n_after_dedup", "entered_training", "exclusion_reason",
]
audit_col_order = [c for c in audit_col_order
                   if c in pd.DataFrame(audit_rows).columns]
audit_df = pd.DataFrame(audit_rows)[audit_col_order]
audit_df.to_csv(AUDIT_CSV, index=False)
print(f"\n  Full audit table saved → {AUDIT_CSV}")

# ============================================================
# STEP 8: Merge protein sequences
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Merging protein sequences")
print("=" * 60)

prot_df  = pd.read_csv(PROT_CSV)
required = {"uniprot_id", "protein_fasta"}
if not required.issubset(prot_df.columns):
    raise ValueError(
        f"Protein CSV must have columns: {required}\n"
        f"Found: {list(prot_df.columns)}"
    )

prot_df  = prot_df[["uniprot_id", "protein_fasta"]].drop_duplicates("uniprot_id")
df_final = df_agg.merge(prot_df, on="uniprot_id", how="inner")
df_final = df_final[df_final["protein_fasta"].str.len() >= 50].reset_index(drop=True)
print(f"  After sequence merge: {len(df_final)} rows")

# ============================================================
# STEP 9: pChEMBL distribution summary
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: pChEMBL distribution per target")
print("=" * 60)

dist_rows = []
for uid, sub in df_final.groupby("uniprot_id"):
    tname = UNIPROT_TO_NAME.get(uid, uid)
    vals  = sub["pchembl_value"]
    dist_rows.append({
        "target_name":    tname,
        "uniprot_id":     uid,
        "n_compounds":    len(sub),
        "pchembl_min":    round(float(vals.min()),          3),
        "pchembl_q1":     round(float(vals.quantile(0.25)), 3),
        "pchembl_median": round(float(vals.median()),       3),
        "pchembl_q3":     round(float(vals.quantile(0.75)), 3),
        "pchembl_max":    round(float(vals.max()),          3),
    })

dist_df = pd.DataFrame(dist_rows)
print(dist_df.to_string(index=False))
dist_df.to_csv(DIST_CSV, index=False)

# ============================================================
# SAVE FINAL DATASET
# ============================================================

col_order = [
    "molecule_chembl_id", "uniprot_id", "target_name",
    "smiles", "pchembl_value", "protein_fasta",
]
df_final = df_final[col_order]
df_final.to_csv(OUTPUT_CSV, index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CODE 1 COMPLETE")
print("=" * 60)
print(f"  Final dataset : {df_final.shape}")
print(f"  Saved to      : {OUTPUT_CSV}")
print(f"  Distribution  : {DIST_CSV}")
print(f"  Audit table   : {AUDIT_CSV}")

print(f"\n  Per-target compound counts (training targets only):")
print(df_final.groupby("target_name").size().to_string())

print(f"\n  SUMMARY — all 30 targets:")
print(f"  {'Target':<10} {'Entered Training':<18} {'Reason if excluded'}")
print("  " + "-" * 70)
for row in audit_rows:
    status = "YES" if row["entered_training"] else "NO"
    reason = "—" if row["entered_training"] else row["exclusion_reason"][:55] + "..."
    print(f"  {row['target_name']:<10} {status:<18} {reason}")

print(f"\n  Checkpoints in: {CKPT_DIR}")
print("  To do a clean re-run, delete the checkpoints folder first.")