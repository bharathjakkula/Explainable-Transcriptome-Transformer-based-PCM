import os
import gc
import json
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger
from transformers import AutoTokenizer, AutoModel

RDLogger.DisableLog('rdApp.*')

# ============================================================
# PATHS
# ============================================================

BASE_DIR   = r"D:\MultiModal_Classification"
FEAT_DIR   = os.path.join(BASE_DIR, "output", "08_pcm_features", "hybrid_matrix")  # 08d output
TRAIN_DIR  = os.path.join(BASE_DIR, "output", "09_pcm_training_screening", "training_cv")  # 09a output (trained models)
EMB_DIR    = os.path.join(BASE_DIR, "output", "08_pcm_features", "embeddings")     # 08c output
INPUT_DIR  = os.path.join(BASE_DIR, "input", "pcm")   # place NPASS_screening_library.csv here
SCREEN_DIR = os.path.join(BASE_DIR, "output", "09_pcm_training_screening", "virtual_screening")
PLOT_DIR   = os.path.join(SCREEN_DIR, "plots")
TARGET_DIR = os.path.join(SCREEN_DIR, "Top100_per_target")
MODEL_DIR  = os.path.join(TRAIN_DIR, "models")

for d in [SCREEN_DIR, PLOT_DIR, TARGET_DIR]:
    os.makedirs(d, exist_ok=True)

LAYOUT_FILE   = os.path.join(FEAT_DIR, "feature_layout.json")
PROT_EMB_FILE = os.path.join(EMB_DIR,  "unique_prot_embeddings.npy")
PROT_IDX_FILE = os.path.join(EMB_DIR,  "unique_prot_embedding_index.csv")
LIBRARY_CSV   = os.path.join(INPUT_DIR,"NPASS_screening_library.csv")

# ============================================================
# LOAD FEATURE LAYOUT
# ============================================================

with open(LAYOUT_FILE) as f:
    layout = json.load(f)

TOTAL_DIM        = layout["TOTAL_DIM"]
DESC_START       = layout["DESC_START"]
DESC_END         = layout["DESC_END"]
FP_BITS          = layout["FP_BITS"]
DESCRIPTOR_NAMES = layout["DESCRIPTOR_NAMES"]
N_DESCRIPTORS    = layout["DESC_DIM"]
CHEM_DIM         = layout["CHEM_DIM"]
PROT_DIM         = layout["PROT_DIM"]

print("=" * 60)
print("HYBRID PCM VIRTUAL SCREENING")
print("=" * 60)
print(f"  Total dim   : {TOTAL_DIM}")
print(f"  Descriptors : {N_DESCRIPTORS} ({DESCRIPTOR_NAMES[:3]}...)")
print(f"  FP bits     : {FP_BITS}")

# ============================================================
# LOCKED DESCRIPTOR FUNCTIONS
# ============================================================

DESCRIPTOR_FUNCTIONS = {
    "MolWt":               Descriptors.MolWt,
    "ExactMolWt":          Descriptors.ExactMolWt,
    "MolLogP":             Descriptors.MolLogP,
    "TPSA":                Descriptors.TPSA,
    "FractionCSP3":        Descriptors.FractionCSP3,
    "NumHDonors":          Descriptors.NumHDonors,
    "NumHAcceptors":       Descriptors.NumHAcceptors,
    "NumRotatableBonds":   Descriptors.NumRotatableBonds,
    "RingCount":           Descriptors.RingCount,
    "NumAromaticRings":    Descriptors.NumAromaticRings,
    "HeavyAtomCount":      Descriptors.HeavyAtomCount,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "MaxPartialCharge":    Descriptors.MaxPartialCharge,
    "MinPartialCharge":    Descriptors.MinPartialCharge,
    "BertzCT":             Descriptors.BertzCT,
    "BalabanJ":            Descriptors.BalabanJ,
    "Chi0":                Descriptors.Chi0,
    "Chi1":                Descriptors.Chi1,
    "Kappa1":              Descriptors.Kappa1,
    "Kappa2":              Descriptors.Kappa2,
}

assert list(DESCRIPTOR_FUNCTIONS.keys()) == DESCRIPTOR_NAMES, \
    "Descriptor name/order mismatch with feature_layout.json"

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class HybridPCM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# ============================================================
# LOAD PROTEIN EMBEDDINGS + TARGET MAPPING
# ============================================================

print("\n" + "=" * 60)
print("Loading protein embeddings...")
print("=" * 60)

prot_emb_matrix = np.load(PROT_EMB_FILE)
prot_idx_df     = pd.read_csv(PROT_IDX_FILE)

assert prot_emb_matrix.shape[0] == len(prot_idx_df), \
    "Protein embedding row count does not match index CSV"
assert prot_emb_matrix.shape[1] == PROT_DIM, \
    f"ProtBERT dim mismatch: {prot_emb_matrix.shape[1]} vs {PROT_DIM}"

N_TARGETS         = prot_emb_matrix.shape[0]
TARGET_NAMES      = prot_idx_df["target_name"].tolist()
UNIPROT_IDS       = prot_idx_df["uniprot_id"].tolist()
TARGET_SCORE_COLS = [f"{t}_score" for t in TARGET_NAMES]

print(f"  Targets ({N_TARGETS}):")
for i, (uid, tname) in enumerate(zip(UNIPROT_IDS, TARGET_NAMES)):
    print(f"    Row {i}: {tname} ({uid})")

# ============================================================
# LOAD ALL FOLD MODELS + SCALERS + THRESHOLDS
# Discovers models automatically — works for any number of folds
# ============================================================

print("\n" + "=" * 60)
print("Loading fold models (ensemble)...")
print("=" * 60)

fold_models     = []
fold_scalers    = []
fold_thresholds = []

model_files = sorted([
    f for f in os.listdir(MODEL_DIR)
    if f.startswith("model_fold_") and f.endswith(".pt")
])

if not model_files:
    raise FileNotFoundError(
        f"No fold models found in {MODEL_DIR}.\n"
        "Expected files named model_fold_1.pt, model_fold_2.pt, ..."
    )

for mf in model_files:
    fold_num    = mf.replace("model_fold_", "").replace(".pt", "")
    model_path  = os.path.join(MODEL_DIR, mf)
    scaler_path = os.path.join(MODEL_DIR,
                               f"descriptor_scaler_fold_{fold_num}.pkl")
    thr_path    = os.path.join(MODEL_DIR,
                               f"threshold_fold_{fold_num}.json")

    if not os.path.exists(scaler_path) or not os.path.exists(thr_path):
        print(f"  Skipping fold {fold_num} — missing scaler or threshold")
        continue

    model = HybridPCM(TOTAL_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    fold_models.append(model)

    fold_scalers.append(joblib.load(scaler_path))

    with open(thr_path) as f:
        thr = json.load(f)["optimal_threshold"]
    fold_thresholds.append(float(thr))

    print(f"  Fold {fold_num}: loaded | threshold={fold_thresholds[-1]:.4f}")

N_FOLDS            = len(fold_models)
ENSEMBLE_THRESHOLD = float(np.mean(fold_thresholds))
print(f"\n  Total fold models    : {N_FOLDS}")
print(f"  Ensemble threshold   : {ENSEMBLE_THRESHOLD:.4f}")

# ============================================================
# LOAD ChemBERTa
# ============================================================

print("\n" + "=" * 60)
print("Loading ChemBERTa...")
print("=" * 60)

chem_tokenizer = AutoTokenizer.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1"
)
chemberta = AutoModel.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1"
).to(device).eval()
fp_generator = GetMorganGenerator(radius=2, fpSize=FP_BITS)
print("  ChemBERTa loaded.")

# ============================================================
# FEATURE GENERATION
# ============================================================

@torch.no_grad()
def get_chemberta_cls(smiles_list):
    enc = chem_tokenizer(
        smiles_list, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt"
    ).to(device)
    out = chemberta(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy()

def get_descriptors(smiles_list):
    data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            data.append([0.0] * N_DESCRIPTORS)
            continue
        row = []
        for func in DESCRIPTOR_FUNCTIONS.values():
            try:
                val = func(mol)
                val = 0.0 if (np.isnan(val) or np.isinf(val)) else val
            except Exception:
                val = 0.0
            row.append(float(val))
        data.append(row)
    return np.array(data, dtype=np.float32)

def get_fingerprints(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(FP_BITS, dtype=np.float32)
        if mol is not None:
            ConvertToNumpyArray(fp_generator.GetFingerprint(mol), arr)
        fps.append(arr)
    return np.array(fps, dtype=np.float32)

# ============================================================
# BATCH SCORING
# ============================================================

def score_batch(smiles_list):
    B            = len(smiles_list)
    chem_emb     = get_chemberta_cls(smiles_list)
    descriptors  = get_descriptors(smiles_list)
    fingerprints = get_fingerprints(smiles_list)

    ensemble_probs = np.zeros((B, N_TARGETS), dtype=np.float64)

    for model, scaler in zip(fold_models, fold_scalers):
        desc_scaled = scaler.transform(descriptors)

        for t_idx in range(N_TARGETS):
            prot_tile = np.tile(
                prot_emb_matrix[t_idx], (B, 1)
            ).astype(np.float32)

            X = np.concatenate(
                [chem_emb, desc_scaled, fingerprints, prot_tile],
                axis=1
            ).astype(np.float32)

            assert X.shape[1] == TOTAL_DIM, \
                f"Feature dim mismatch: {X.shape[1]} vs {TOTAL_DIM}"

            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(X_t)).cpu().numpy()
            ensemble_probs[:, t_idx] += probs

    return (ensemble_probs / N_FOLDS).astype(np.float32)

# ============================================================
# LOAD AND VALIDATE LIBRARY
# ============================================================

print("\n" + "=" * 60)
print("Loading screening library...")
print("=" * 60)

lib_df = pd.read_csv(LIBRARY_CSV)
print(f"  Loaded: {len(lib_df)} compounds")

if "smiles" not in lib_df.columns or "molecule_id" not in lib_df.columns:
    raise ValueError(
        "Library must have columns 'smiles' and 'molecule_id'.\n"
        f"Found: {list(lib_df.columns)}"
    )

def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(str(smi)) is not None
    except Exception:
        return False

lib_df["_valid"] = lib_df["smiles"].apply(is_valid_smiles)
n_bad = (~lib_df["_valid"]).sum()
if n_bad > 0:
    print(f"  Removed {n_bad} invalid SMILES")
lib_df = lib_df[lib_df["_valid"]].drop(
    columns=["_valid"]
).reset_index(drop=True)
print(f"  Valid compounds: {len(lib_df)}")

# ============================================================
# MAIN SCREENING LOOP
# ============================================================

print("\n" + "=" * 60)
print("RUNNING VIRTUAL SCREENING")
print("=" * 60)
print(f"  Compounds  : {len(lib_df)}")
print(f"  Targets    : {N_TARGETS} ({', '.join(TARGET_NAMES)})")
print(f"  Ensemble   : {N_FOLDS} fold models")
print(f"  Threshold  : {ENSEMBLE_THRESHOLD:.4f}")
print("=" * 60)

BATCH_SIZE  = 16
all_results = []

for i in tqdm(range(0, len(lib_df), BATCH_SIZE), desc="Screening"):
    batch        = lib_df.iloc[i: i + BATCH_SIZE]
    smiles_batch = batch["smiles"].tolist()

    try:
        scores = score_batch(smiles_batch)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            scores_list = []
            for smi in smiles_batch:
                try:
                    s = score_batch([smi])
                    scores_list.append(s[0])
                except Exception:
                    scores_list.append(
                        np.zeros(N_TARGETS, dtype=np.float32)
                    )
            scores = np.array(scores_list)
        else:
            raise e

    for j, (_, row) in enumerate(batch.iterrows()):
        res = {
            "molecule_id": row["molecule_id"],
            "smiles":      row["smiles"],
        }
        for t_idx, col in enumerate(TARGET_SCORE_COLS):
            res[col] = float(scores[j, t_idx])
        res["max_score"]        = float(scores[j].max())
        res["mean_score"]       = float(scores[j].mean())
        res["n_active_targets"] = int(
            (scores[j] >= ENSEMBLE_THRESHOLD).sum()
        )
        all_results.append(res)

    if i % (BATCH_SIZE * 20) == 0 and i > 0:
        torch.cuda.empty_cache()
        gc.collect()

# ============================================================
# BUILD RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(all_results)
print(f"\n  Screening complete: {len(results_df)} compounds scored")

results_df.to_csv(
    os.path.join(SCREEN_DIR, "Complete_VS_Results.csv"), index=False
)

# ============================================================
# OUTPUT A: PER-TARGET TOP 100
# ============================================================

print("\n  Per-target activity:")
for tname, col in zip(TARGET_NAMES, TARGET_SCORE_COLS):
    n_act = (results_df[col] >= ENSEMBLE_THRESHOLD).sum()
    print(f"    {tname:12s}: {n_act:4d} active")
    top100 = results_df.nlargest(100, col)[
        ["molecule_id", "smiles", col, "max_score", "n_active_targets"]
    ].reset_index(drop=True)
    top100["rank"] = top100.index + 1
    top100.to_csv(
        os.path.join(TARGET_DIR, f"Top100_{tname}.csv"), index=False
    )

# ============================================================
# OUTPUT B: POLYPHARMACOLOGY HITS
# ============================================================

poly_df = results_df[
    results_df["n_active_targets"] >= 2
].sort_values(
    ["n_active_targets", "mean_score"], ascending=False
).reset_index(drop=True)
poly_df.to_csv(
    os.path.join(SCREEN_DIR, "Polypharmacology_hits.csv"), index=False
)

pan_df = results_df[
    results_df["n_active_targets"] == N_TARGETS
].sort_values("mean_score", ascending=False).reset_index(drop=True)
pan_df.to_csv(
    os.path.join(SCREEN_DIR, "Pan_active_hits.csv"), index=False
)

print(f"\n  Polypharmacology (>=2 targets): {len(poly_df)}")
print(f"  Pan-active (all {N_TARGETS} targets): {len(pan_df)}")

activity_breakdown = []
for tname, col in zip(TARGET_NAMES, TARGET_SCORE_COLS):
    n_act = (results_df[col] >= ENSEMBLE_THRESHOLD).sum()
    activity_breakdown.append({
        "target":     tname,
        "n_active":   int(n_act),
        "mean_score": round(float(results_df[col].mean()), 4),
        "max_score":  round(float(results_df[col].max()),  4),
    })
pd.DataFrame(activity_breakdown).to_csv(
    os.path.join(SCREEN_DIR, "per_target_activity.csv"), index=False
)

summary = {
    "total_screened":         int(len(results_df)),
    "n_targets":              int(N_TARGETS),
    "n_folds_ensemble":       int(N_FOLDS),
    "ensemble_threshold":     round(ENSEMBLE_THRESHOLD, 4),
    "mean_max_score":         round(float(results_df["max_score"].mean()), 4),
    "std_max_score":          round(float(results_df["max_score"].std()),  4),
    "n_polypharmacology_ge2": int(len(poly_df)),
    "n_pan_active":           int(len(pan_df)),
}
pd.DataFrame([summary]).T.to_csv(
    os.path.join(SCREEN_DIR, "screening_summary.csv"), header=False
)

# ============================================================
# PLOTS
# ============================================================

print("\n  Generating publication plots...")

# ----------------------------------------------------------
# Plot 1: Max score distribution
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(results_df["max_score"], bins=50, color="steelblue",
        edgecolor="black", alpha=0.8)
ax.axvline(ENSEMBLE_THRESHOLD, color="red", lw=2, linestyle="--",
           label=f"Threshold = {ENSEMBLE_THRESHOLD:.3f}")
ax.set_xlabel("Maximum Score (across all targets)", fontsize=12)
ax.set_ylabel("Number of Compounds",               fontsize=12)
ax.set_title("Distribution of Maximum Ensemble Scores", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "score_distribution.png"), dpi=300)
plt.close()

# ----------------------------------------------------------
# Plot 2: Per-target violin plots
# Figure width scales with N_TARGETS
# ----------------------------------------------------------
fig_w = max(10, N_TARGETS * 1.4)
fig, ax = plt.subplots(figsize=(fig_w, 6))
score_data = [results_df[col].values for col in TARGET_SCORE_COLS]
parts = ax.violinplot(score_data, positions=range(N_TARGETS),
                      showmedians=True)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.7)
ax.axhline(ENSEMBLE_THRESHOLD, color="red", lw=1.5, linestyle="--",
           label=f"Threshold = {ENSEMBLE_THRESHOLD:.3f}")
ax.set_xticks(range(N_TARGETS))                          # set ticks first
ax.set_xticklabels(TARGET_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Ensemble Score", fontsize=12)
ax.set_title("Per-Target Score Distributions (Violin)", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "per_target_violin.png"), dpi=300)
plt.close()

# ----------------------------------------------------------
# Plot 3: Active compound counts per target
# ----------------------------------------------------------
fig_w = max(10, N_TARGETS * 1.2)
fig, ax = plt.subplots(figsize=(fig_w, 5))
n_actives = [(results_df[col] >= ENSEMBLE_THRESHOLD).sum()
             for col in TARGET_SCORE_COLS]
colors = plt.cm.tab20(np.linspace(0, 1, N_TARGETS))
bars   = ax.bar(range(N_TARGETS), n_actives,
                color=colors, edgecolor="black", alpha=0.85)
ax.set_xticks(range(N_TARGETS))                          # set ticks first
ax.set_xticklabels(TARGET_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Number of Active Compounds", fontsize=12)
ax.set_title(
    f"Per-Target Active Compounds (score ≥ {ENSEMBLE_THRESHOLD:.3f})",
    fontsize=13
)
for bar, n in zip(bars, n_actives):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        str(n), ha="center", va="bottom", fontsize=8
    )
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "per_target_active_counts.png"),
            dpi=300)
plt.close()

# ----------------------------------------------------------
# Plot 4: Polypharmacology distribution
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
counts = results_df["n_active_targets"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color="teal",
       edgecolor="black", alpha=0.8)
ax.set_xticks(range(0, N_TARGETS + 1))
ax.set_xlabel("Number of Active Targets per Compound", fontsize=12)
ax.set_ylabel("Number of Compounds",                   fontsize=12)
ax.set_title("Polypharmacology Profile of Screening Library",
             fontsize=13)
for x, y in zip(counts.index, counts.values):
    ax.text(x, y + 0.3, str(y), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "polypharmacology_distribution.png"),
            dpi=300)
plt.close()

# ----------------------------------------------------------
# Plot 5: Top 50 heatmap
# ----------------------------------------------------------
top50        = results_df.nlargest(50, "max_score")
heatmap_data = top50[TARGET_SCORE_COLS].values

fig_w = max(10, N_TARGETS * 1.1)
fig, ax = plt.subplots(figsize=(fig_w, 14))
sns.heatmap(
    heatmap_data,
    xticklabels=TARGET_NAMES,
    yticklabels=top50["molecule_id"].values,
    cmap="YlOrRd", vmin=0, vmax=1,
    linewidths=0.3, annot=False,
    cbar_kws={"label": "Ensemble Score"},
    ax=ax,
)
ax.set_xticklabels(TARGET_NAMES, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
ax.set_title("Top 50 Compounds — Ensemble Scores Across All Targets",
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "top50_heatmap.png"), dpi=300)
plt.close()

# ----------------------------------------------------------
# Plot 6: Target score correlation heatmap
# ----------------------------------------------------------
corr = results_df[TARGET_SCORE_COLS].corr()
corr.index = corr.columns = TARGET_NAMES

fig_s = max(7, N_TARGETS * 0.85)
fig, ax = plt.subplots(figsize=(fig_s, fig_s - 1))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pearson r"})
ax.set_title("Inter-Target Score Correlation", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "target_score_correlation.png"),
            dpi=300)
plt.close()

# ----------------------------------------------------------
# Plot 7: Top 10 compounds per target — DYNAMIC GRID
#
# FIX: grid dimensions computed from N_TARGETS at runtime.
# ncols is capped at 4 for readability; nrows computed to fit
# all N_TARGETS panels. Extra (empty) panels are hidden.
# This works for any N_TARGETS value.
# ----------------------------------------------------------
NCOLS  = min(4, N_TARGETS)
NROWS  = math.ceil(N_TARGETS / NCOLS)
fig_w  = NCOLS * 5
fig_h  = NROWS * 4

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(fig_w, fig_h))
axes = np.array(axes).flatten()   # always 1-D, regardless of shape

for t_idx, (tname, col) in enumerate(zip(TARGET_NAMES, TARGET_SCORE_COLS)):
    ax    = axes[t_idx]
    top10 = results_df.nlargest(10, col)
    y_pos = range(len(top10))
    ax.barh(list(y_pos), top10[col].values,
            color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top10["molecule_id"].values, fontsize=7)
    ax.set_xlabel("Score", fontsize=9)
    ax.set_title(f"Top 10 — {tname}", fontsize=10)
    ax.axvline(ENSEMBLE_THRESHOLD, color="red", lw=1, linestyle="--")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

# Hide any unused panels in the grid
for ax in axes[N_TARGETS:]:
    ax.set_visible(False)

plt.suptitle("Top 10 Compounds per Target", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "top10_per_target.png"),
            dpi=300, bbox_inches="tight")
plt.close()

print("  All plots saved.")

# ============================================================
# SCREENING REPORT
# ============================================================

report_lines = [
    "=" * 72,
    "           VIRTUAL SCREENING REPORT — PCM PIPELINE",
    "=" * 72,
    "",
    f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "INPUT",
    "-" * 40,
    f"  Screening library   : {LIBRARY_CSV}",
    f"  Total compounds     : {len(lib_df)}",
    f"  Valid screened      : {len(results_df)}",
    f"  Targets             : {N_TARGETS} ({', '.join(TARGET_NAMES)})",
    "",
    "MODEL",
    "-" * 40,
    f"  Architecture        : HybridPCM (4-layer MLP, {TOTAL_DIM} input dim)",
    f"  Ensemble            : {N_FOLDS}-fold models (mean probability)",
    f"  Threshold           : {ENSEMBLE_THRESHOLD:.4f} (mean of fold thresholds)",
    "",
    "FEATURE COMPOSITION",
    "-" * 40,
    f"  ChemBERTa CLS       : {CHEM_DIM}",
    f"  Descriptors (scaled): {N_DESCRIPTORS}",
    f"  Morgan FP           : {FP_BITS} bits (radius=2)",
    f"  ProtBERT CLS        : {PROT_DIM}",
    f"  Total               : {TOTAL_DIM}",
    "",
    "RESULTS",
    "-" * 40,
    f"  Mean max score      : {summary['mean_max_score']:.4f}"
    f" ± {summary['std_max_score']:.4f}",
    f"  Polypharmacology (≥2): {len(poly_df)} compounds",
    f"  Pan-active (all {N_TARGETS}): {len(pan_df)} compounds",
    "",
    "PER-TARGET ACTIVITY",
    "-" * 40,
]
for tname, col in zip(TARGET_NAMES, TARGET_SCORE_COLS):
    n_act = (results_df[col] >= ENSEMBLE_THRESHOLD).sum()
    report_lines.append(
        f"  {tname:12s}: {n_act:4d} active | "
        f"mean={results_df[col].mean():.4f} | "
        f"max={results_df[col].max():.4f}"
    )

report_lines += [
    "",
    "OUTPUT FILES",
    "-" * 40,
    f"  Complete_VS_Results.csv        "
    f"({len(results_df)} rows × {len(results_df.columns)} cols)",
    f"  Top100_per_target/             ({N_TARGETS} files, 100 hits each)",
    f"  Polypharmacology_hits.csv      ({len(poly_df)} compounds)",
    f"  Pan_active_hits.csv            ({len(pan_df)} compounds)",
    f"  per_target_activity.csv",
    f"  screening_summary.csv",
    f"  plots/                         (7 publication figures)",
    "",
    "=" * 72,
    "SCREENING COMPLETE",
    "=" * 72,
]

report_text = "\n".join(report_lines)
with open(os.path.join(SCREEN_DIR, "screening_report.txt"),
          "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n" + report_text)
print(f"\nAll results saved to: {SCREEN_DIR}")