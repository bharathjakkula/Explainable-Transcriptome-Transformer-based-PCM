import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ============================================================
# PATHS
# ============================================================

BASE_DIR   = r"D:\MultiModal_Classification"
EMB_DIR    = os.path.join(BASE_DIR, "output", "08_pcm_features", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# 08b output
INPUT_CSV       = os.path.join(BASE_DIR, "output", "08_pcm_features", "chembl", "PCM_dataset_clean.csv")
CHEM_EMB_FILE   = os.path.join(EMB_DIR, "chem_embeddings.npy")
PROT_EMB_FILE   = os.path.join(EMB_DIR, "prot_embeddings.npy")
INDEX_CSV       = os.path.join(EMB_DIR, "embedding_index.csv")

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"  Total compound-protein pairs : {len(df)}")
print(f"  Unique SMILES                : {df['smiles'].nunique()}")
print(f"  Unique proteins              : {df['uniprot_id'].nunique()}")
print(f"  Targets: {df['target_name'].unique().tolist()}")

# ============================================================
# LOAD MODELS
# ============================================================

print("\n" + "=" * 60)
print("Loading ChemBERTa...")
print("=" * 60)
chem_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chem_model     = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chem_model     = chem_model.to(device).eval()
print("  ChemBERTa loaded.")

print("\n" + "=" * 60)
print("Loading ProtBERT...")
print("=" * 60)
prot_tokenizer = AutoTokenizer.from_pretrained(
    "Rostlab/prot_bert_bfd",
    do_lower_case=False
)
prot_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
prot_model = prot_model.to(device).eval()
print("  ProtBERT loaded.")

# ============================================================
# EMBEDDING FUNCTION
# CLS TOKEN USED CONSISTENTLY FOR BOTH MODELS
# ============================================================

@torch.no_grad()
def get_cls_embeddings(texts, tokenizer, model, max_length, is_protein=False):
    """
    Extract CLS token embeddings from a batch of texts.

    For ProtBERT: amino acids must be space-separated before tokenizing.
    For ChemBERTa: SMILES strings are passed as-is.

    Returns: numpy array of shape (batch_size, hidden_dim)
    """
    if is_protein:
        # ProtBERT requirement: single space between each amino acid character
        texts = [" ".join(list(seq)) for seq in texts]

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # CLS token = position 0 in last hidden state
    # Shape: (batch_size, hidden_dim)
    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_emb

# ============================================================
# STEP 1: ChemBERTa embeddings for UNIQUE SMILES
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Extracting ChemBERTa embeddings (unique SMILES only)")
print("=" * 60)

CHEM_BATCH = 32
MAX_SMILES_LEN = 128   # ChemBERTa max context

unique_smiles    = df["smiles"].unique().tolist()
smiles_to_idx    = {smi: i for i, smi in enumerate(unique_smiles)}
n_unique_smiles  = len(unique_smiles)
print(f"  Unique SMILES to embed: {n_unique_smiles}")

unique_chem_emb = []
for i in tqdm(range(0, n_unique_smiles, CHEM_BATCH), desc="  ChemBERTa"):
    batch = unique_smiles[i: i + CHEM_BATCH]
    emb   = get_cls_embeddings(batch, chem_tokenizer, chem_model,
                               max_length=MAX_SMILES_LEN, is_protein=False)
    unique_chem_emb.append(emb)

unique_chem_emb = np.vstack(unique_chem_emb)  # (n_unique_smiles, 768)
print(f"  Unique ChemBERTa embedding shape: {unique_chem_emb.shape}")

# Expand to full pair list (each row in df gets its compound's embedding)
chem_row_indices  = df["smiles"].map(smiles_to_idx).values
chem_embeddings   = unique_chem_emb[chem_row_indices]    # (N, 768)
print(f"  Expanded ChemBERTa embedding shape: {chem_embeddings.shape}")

# ============================================================
# STEP 2: ProtBERT embeddings for UNIQUE SEQUENCES
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Extracting ProtBERT embeddings (unique sequences only)")
print("=" * 60)

PROT_BATCH    = 4      # ProtBERT is very large; small batch to avoid OOM
MAX_PROT_LEN  = 1024   # ProtBERT context window

unique_seqs   = df["protein_fasta"].unique().tolist()
seq_to_idx    = {seq: i for i, seq in enumerate(unique_seqs)}
n_unique_seqs = len(unique_seqs)
print(f"  Unique protein sequences to embed: {n_unique_seqs}")

unique_prot_emb = []
for i in tqdm(range(0, n_unique_seqs, PROT_BATCH), desc="  ProtBERT"):
    batch = unique_seqs[i: i + PROT_BATCH]
    emb   = get_cls_embeddings(batch, prot_tokenizer, prot_model,
                               max_length=MAX_PROT_LEN, is_protein=True)
    unique_prot_emb.append(emb)

unique_prot_emb = np.vstack(unique_prot_emb)  # (n_unique_seqs, 1024)
print(f"  Unique ProtBERT embedding shape: {unique_prot_emb.shape}")

# Save unique protein embeddings separately (used in virtual screening)
unique_prot_file = os.path.join(EMB_DIR, "unique_prot_embeddings.npy")
np.save(unique_prot_file, unique_prot_emb)

# Save the sequence-to-row mapping for screening use
seq_map_df = pd.DataFrame({
    "row_index":  range(n_unique_seqs),
    "protein_fasta": unique_seqs,
})
# Merge target names for readability
uid_name_map = df[["protein_fasta", "uniprot_id", "target_name"]].drop_duplicates("protein_fasta")
seq_map_df   = seq_map_df.merge(uid_name_map, on="protein_fasta", how="left")
seq_map_df.to_csv(os.path.join(EMB_DIR, "unique_prot_embedding_index.csv"), index=False)
print(f"  Unique protein embeddings saved: {unique_prot_file}")
print("  Protein embedding order:")
print(seq_map_df[["row_index", "uniprot_id", "target_name"]].to_string(index=False))

# Expand to full pair list
prot_row_indices = df["protein_fasta"].map(seq_to_idx).values
prot_embeddings  = unique_prot_emb[prot_row_indices]    # (N, 1024)
print(f"  Expanded ProtBERT embedding shape: {prot_embeddings.shape}")

# ============================================================
# SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 60)
print("Saving embeddings...")
print("=" * 60)

np.save(CHEM_EMB_FILE, chem_embeddings)
np.save(PROT_EMB_FILE, prot_embeddings)

# Save index file (full pair list, same row order as .npy files)
df.to_csv(INDEX_CSV, index=False)

print(f"  chem_embeddings.npy  : {chem_embeddings.shape} → {CHEM_EMB_FILE}")
print(f"  prot_embeddings.npy  : {prot_embeddings.shape} → {PROT_EMB_FILE}")
print(f"  embedding_index.csv  : {len(df)} rows → {INDEX_CSV}")
print(f"  unique_prot_embeddings.npy : {unique_prot_emb.shape} → {unique_prot_file}")

print("\n" + "=" * 60)
print("CODE 2 COMPLETE")
print("=" * 60)