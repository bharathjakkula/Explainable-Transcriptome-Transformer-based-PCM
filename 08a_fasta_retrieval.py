import os
import requests
import pandas as pd
import time

# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"D:\MultiModal_Classification"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "08_pcm_features", "fasta")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "target_fasta_sequences.csv")

# ============================================================
# INPUT: UniProt IDs (30 SHAP-derived candidate genes, Section 3.5)
# ============================================================
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
    "STARD3": "Q14849"
}

# ============================================================
# FUNCTION: Fetch sequence from UniProt
# ============================================================
def fetch_sequence(uid):
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
        r = requests.get(url, timeout=10)

        if not r.ok:
            print(f"Failed: {uid}")
            return None

        # Remove FASTA header and join sequence lines
        sequence = "".join(
            line.strip() for line in r.text.splitlines() if not line.startswith(">")
        )

        return sequence

    except Exception as e:
        print(f"Error for {uid}: {e}")
        return None

# ============================================================
# FETCH ALL SEQUENCES
# ============================================================
records = []

for protein_name, uid in TARGET_UNIPROT_IDS.items():
    seq = fetch_sequence(uid)

    if seq and len(seq) >= 50:  # optional filter
        records.append({
            "target_name": protein_name,
            "uniprot_id": uid,
            "protein_fasta": seq
        })

    time.sleep(0.2)  # avoid rate limiting

# ============================================================
# CREATE DATAFRAME
# ============================================================
df = pd.DataFrame(records)

# ============================================================
# SAVE
# ============================================================
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset saved:", df.shape)