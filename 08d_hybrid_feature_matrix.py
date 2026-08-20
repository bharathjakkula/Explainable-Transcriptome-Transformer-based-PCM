import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger
from sklearn.decomposition import PCA

RDLogger.DisableLog('rdApp.*')

# ============================================================
# PATHS
# ============================================================

BASE_DIR    = r"D:\MultiModal_Classification"
EMB_DIR     = os.path.join(BASE_DIR, "output", "08_pcm_features", "embeddings")   # 08c output
FEAT_DIR    = os.path.join(BASE_DIR, "output", "08_pcm_features", "hybrid_matrix")
PLOT_DIR    = os.path.join(FEAT_DIR, "plots")
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

CHEM_EMB_FILE  = os.path.join(EMB_DIR, "chem_embeddings.npy")
PROT_EMB_FILE  = os.path.join(EMB_DIR, "prot_embeddings.npy")
INDEX_CSV      = os.path.join(EMB_DIR, "embedding_index.csv")

X_OUT          = os.path.join(FEAT_DIR, "X_hybrid.npy")
Y_OUT          = os.path.join(FEAT_DIR, "y_raw.npy")
INDEX_OUT      = os.path.join(FEAT_DIR, "feature_index.csv")
LAYOUT_OUT     = os.path.join(FEAT_DIR, "feature_layout.json")

# ============================================================
# DESCRIPTOR SET
# ============================================================

DESCRIPTOR_FUNCTIONS = {
    "MolWt":              Descriptors.MolWt,
    "ExactMolWt":         Descriptors.ExactMolWt,
    "MolLogP":            Descriptors.MolLogP,
    "TPSA":               Descriptors.TPSA,
    "FractionCSP3":       Descriptors.FractionCSP3,
    "NumHDonors":         Descriptors.NumHDonors,
    "NumHAcceptors":      Descriptors.NumHAcceptors,
    "NumRotatableBonds":  Descriptors.NumRotatableBonds,
    "RingCount":          Descriptors.RingCount,
    "NumAromaticRings":   Descriptors.NumAromaticRings,
    "HeavyAtomCount":     Descriptors.HeavyAtomCount,
    "NumValenceElectrons":Descriptors.NumValenceElectrons,
    "MaxPartialCharge":   Descriptors.MaxPartialCharge,
    "MinPartialCharge":   Descriptors.MinPartialCharge,
    "BertzCT":            Descriptors.BertzCT,
    "BalabanJ":           Descriptors.BalabanJ,
    "Chi0":               Descriptors.Chi0,
    "Chi1":               Descriptors.Chi1,
    "Kappa1":             Descriptors.Kappa1,
    "Kappa2":             Descriptors.Kappa2,
}

DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCTIONS.keys())
N_DESCRIPTORS    = len(DESCRIPTOR_NAMES)    # 20

# ============================================================
# FEATURE LAYOUT CONSTANTS 
# ============================================================

FP_BITS      = 2048
CHEM_DIM     = 768
DESC_DIM     = N_DESCRIPTORS            # 20
FP_DIM       = FP_BITS                  # 2048
PROT_DIM     = 1024
TOTAL_DIM    = CHEM_DIM + DESC_DIM + FP_DIM + PROT_DIM  # 3860

CHEM_START   = 0
CHEM_END     = CHEM_DIM                 # 768
DESC_START   = CHEM_END                 # 768
DESC_END     = DESC_START + DESC_DIM    # 788
FP_START     = DESC_END                 # 788
FP_END       = FP_START + FP_DIM        # 2836
PROT_START   = FP_END                   # 2836
PROT_END     = PROT_START + PROT_DIM    # 3860

layout = {
    "CHEM_DIM":    CHEM_DIM,
    "DESC_DIM":    DESC_DIM,
    "FP_DIM":      FP_DIM,
    "PROT_DIM":    PROT_DIM,
    "TOTAL_DIM":   TOTAL_DIM,
    "CHEM_START":  CHEM_START,
    "CHEM_END":    CHEM_END,
    "DESC_START":  DESC_START,
    "DESC_END":    DESC_END,
    "FP_START":    FP_START,
    "FP_END":      FP_END,
    "PROT_START":  PROT_START,
    "PROT_END":    PROT_END,
    "FP_RADIUS":   2,
    "FP_BITS":     FP_BITS,
    "DESCRIPTOR_NAMES": DESCRIPTOR_NAMES,
}

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading embeddings and index...")
print("=" * 60)

chem_embeddings = np.load(CHEM_EMB_FILE)
prot_embeddings = np.load(PROT_EMB_FILE)
df              = pd.read_csv(INDEX_CSV)

smiles_list     = df["smiles"].values
pchembl_values  = df["pchembl_value"].values
target_names    = df["target_name"].values

print(f"  ChemBERTa embeddings : {chem_embeddings.shape}")
print(f"  ProtBERT embeddings  : {prot_embeddings.shape}")
print(f"  Dataset rows         : {len(df)}")

assert chem_embeddings.shape[0] == len(df), \
    "chem_embeddings row count does not match index CSV"
assert prot_embeddings.shape[0] == len(df), \
    "prot_embeddings row count does not match index CSV"
assert chem_embeddings.shape[1] == CHEM_DIM, \
    f"ChemBERTa dim mismatch: got {chem_embeddings.shape[1]}, expected {CHEM_DIM}"
assert prot_embeddings.shape[1] == PROT_DIM, \
    f"ProtBERT dim mismatch: got {prot_embeddings.shape[1]}, expected {PROT_DIM}"

# ============================================================
# STEP 1: Molecular Descriptors
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Computing molecular descriptors (raw, unscaled)")
print("=" * 60)

fp_generator = GetMorganGenerator(radius=2, fpSize=FP_BITS)
desc_data    = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        desc_data.append([0.0] * N_DESCRIPTORS)
        continue
    row = []
    for func in DESCRIPTOR_FUNCTIONS.values():
        try:
            val = func(mol)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
        except Exception:
            val = 0.0
        row.append(float(val))
    desc_data.append(row)

desc_array = np.array(desc_data, dtype=np.float32)
print(f"  Descriptor matrix shape : {desc_array.shape}")

# Save raw descriptor values for reference
desc_df = pd.DataFrame(desc_array, columns=DESCRIPTOR_NAMES)
desc_df.to_csv(os.path.join(FEAT_DIR, "descriptor_values_raw.csv"), index=False)

# ============================================================
# STEP 2: Morgan Fingerprints
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Computing Morgan fingerprints (radius=2, 2048 bits)")
print("=" * 60)

fp_data = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    arr = np.zeros(FP_BITS, dtype=np.float32)
    if mol is not None:
        fp = fp_generator.GetFingerprint(mol)
        ConvertToNumpyArray(fp, arr)
    fp_data.append(arr)

fp_array = np.array(fp_data, dtype=np.float32)
print(f"  Fingerprint matrix shape : {fp_array.shape}")
print(f"  Mean bit density         : {fp_array.mean():.4f}")

# ============================================================
# STEP 3: Assemble Hybrid Feature Matrix
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Assembling hybrid feature matrix")
print("=" * 60)

X_hybrid = np.concatenate(
    [chem_embeddings, desc_array, fp_array, prot_embeddings],
    axis=1
).astype(np.float32)

print(f"  Final X_hybrid shape : {X_hybrid.shape}")
print(f"  Expected             : ({len(df)}, {TOTAL_DIM})")
assert X_hybrid.shape[1] == TOTAL_DIM, \
    f"Feature dimension mismatch: got {X_hybrid.shape[1]}, expected {TOTAL_DIM}"

# ============================================================
# STEP 4: Save Outputs
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Saving outputs")
print("=" * 60)

np.save(X_OUT, X_hybrid)
np.save(Y_OUT, pchembl_values.astype(np.float32))

df.to_csv(INDEX_OUT, index=False)

with open(LAYOUT_OUT, "w") as f:
    json.dump(layout, f, indent=2)

print(f"  X_hybrid.npy     : {X_hybrid.shape}  → {X_OUT}")
print(f"  y_raw.npy        : {pchembl_values.shape} → {Y_OUT}")
print(f"  feature_index.csv: {len(df)} rows")
print(f"  feature_layout.json saved")

# ============================================================
# STEP 5: Diagnostic Plots
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Generating diagnostic plots")
print("=" * 60)

# Plot 1: pChEMBL distribution per target
fig, ax = plt.subplots(figsize=(10, 6))
for tname in np.unique(target_names):
    vals = pchembl_values[target_names == tname]
    ax.hist(vals, bins=30, alpha=0.5, label=tname, density=True)
ax.set_xlabel("pChEMBL Value", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("pChEMBL Value Distribution per Target", fontsize=14)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pchembl_distribution.png"), dpi=300)
plt.close()

# Plot 2: Fingerprint bit density distribution
fig, ax = plt.subplots(figsize=(8, 5))
density = fp_array.sum(axis=1) / FP_BITS
ax.hist(density, bins=40, edgecolor="black", color="steelblue", alpha=0.8)
ax.set_xlabel("Fingerprint Bit Density", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Morgan Fingerprint Density Distribution", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "fingerprint_density.png"), dpi=300)
plt.close()

# Plot 3: Descriptor distribution (boxplot, raw values)
fig, ax = plt.subplots(figsize=(14, 6))
ax.boxplot(desc_array, labels=DESCRIPTOR_NAMES, vert=True)
ax.set_xticklabels(DESCRIPTOR_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Raw Value", fontsize=12)
ax.set_title("Molecular Descriptor Distributions (Raw, Unscaled)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "descriptor_distributions.png"), dpi=300)
plt.close()

# Plot 4: PCA of full hybrid features colored by target
pca    = PCA(n_components=2, random_state=42)
X_pca  = pca.fit_transform(X_hybrid)
unique_targets = np.unique(target_names)
cmap   = plt.cm.get_cmap("tab10", len(unique_targets))

fig, ax = plt.subplots(figsize=(9, 7))
for i, tname in enumerate(unique_targets):
    mask = target_names == tname
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[cmap(i)], label=tname, alpha=0.5, s=15)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title("PCA of Hybrid Feature Matrix (colored by target)", fontsize=14)
ax.legend(fontsize=8, markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "PCA_hybrid_features.png"), dpi=300)
plt.close()

# Plot 5: Dataset statistics table
stats = {
    "Total samples":       len(df),
    "Total features":      TOTAL_DIM,
    "ChemBERTa dim":       CHEM_DIM,
    "Descriptor dim (raw)":DESC_DIM,
    "Fingerprint dim":     FP_DIM,
    "ProtBERT dim":        PROT_DIM,
    "Unique SMILES":       df["smiles"].nunique(),
    "Unique proteins":     df["uniprot_id"].nunique(),
}
stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
stats_df.to_csv(os.path.join(FEAT_DIR, "dataset_statistics.csv"), index=False)
print("  Dataset statistics:")
print(stats_df.to_string(index=False))

print("\n" + "=" * 60)
print("CODE 3 COMPLETE")
print("=" * 60)
print(f"  All outputs in : {FEAT_DIR}")