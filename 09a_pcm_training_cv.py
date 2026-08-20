import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import  calibration_curve
from scipy import stats
from scipy.spatial.distance import cdist

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ============================================================
# PATHS
# ============================================================

BASE_DIR  = r"D:\MultiModal_Classification"
FEAT_DIR  = os.path.join(BASE_DIR, "output", "08_pcm_features", "hybrid_matrix")   # 08d output
TRAIN_DIR = os.path.join(BASE_DIR, "output", "09_pcm_training_screening", "training_cv")

MODEL_DIR = os.path.join(TRAIN_DIR, "models")
METRIC_DIR= os.path.join(TRAIN_DIR, "metrics")
PRED_DIR  = os.path.join(TRAIN_DIR, "predictions")
PLOT_DIR  = os.path.join(TRAIN_DIR, "plots")
AD_DIR    = os.path.join(TRAIN_DIR, "applicability_domain")

for d in [MODEL_DIR, METRIC_DIR, PRED_DIR, PLOT_DIR, AD_DIR]:
    os.makedirs(d, exist_ok=True)

X_FILE       = os.path.join(FEAT_DIR, "X_hybrid.npy")
Y_RAW_FILE   = os.path.join(FEAT_DIR, "y_raw.npy")
INDEX_FILE   = os.path.join(FEAT_DIR, "feature_index.csv")
LAYOUT_FILE  = os.path.join(FEAT_DIR, "feature_layout.json")

# ============================================================
# FEATURE LAYOUT
# ============================================================

with open(LAYOUT_FILE) as f:
    layout = json.load(f)

DESC_START = layout["DESC_START"]
DESC_END   = layout["DESC_END"]
TOTAL_DIM  = layout["TOTAL_DIM"]

print("Feature layout loaded:")
print(f"  ChemBERTa  : [{layout['CHEM_START']} : {layout['CHEM_END']})")
print(f"  Descriptors: [{layout['DESC_START']} : {layout['DESC_END']})")
print(f"  Fingerprints:[{layout['FP_START']} : {layout['FP_END']})")
print(f"  ProtBERT   : [{layout['PROT_START']} : {layout['PROT_END']})")
print(f"  TOTAL DIM  : {TOTAL_DIM}")

# ============================================================
# DATASET CLASS
# ============================================================

class HybridDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class HybridPCM(nn.Module):
    """
    4-layer MLP for binary activity classification.
    Input: 3860-dimensional hybrid feature vector.
    Output: single logit (apply sigmoid for probability).
    """
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
# QUARTILE LABELING (on training fold only)
# ============================================================

def assign_quartile_labels(pchembl_train, pchembl_all, uniprot_train, uniprot_all):
    """
    Assign binary labels based on per-target Q1/Q3 thresholds.

    Thresholds computed from TRAINING data only.
    Applied to full fold (train + test) but thresholds never see test data.

    Returns:
        labels: array of 0, 1, or NaN (-1 = middle 50%, exclude)
        thresholds: dict of {uniprot_id: (q1, q3)}
    """
    thresholds  = {}
    n           = len(pchembl_all)
    labels      = np.full(n, np.nan)

    for uid in np.unique(uniprot_train):
        train_mask = uniprot_train == uid
        train_vals = pchembl_train[train_mask]

        q1 = np.percentile(train_vals, 25)
        q3 = np.percentile(train_vals, 75)
        thresholds[uid] = (float(q1), float(q3))

        all_mask   = uniprot_all == uid
        all_vals   = pchembl_all[all_mask]

        fold_labels = np.full(len(all_vals), np.nan)
        fold_labels[all_vals >= q3] = 1.0
        fold_labels[all_vals <= q1] = 0.0

        labels[all_mask] = fold_labels

    return labels, thresholds

# ============================================================
# FIND BEST THRESHOLD (on validation set only)
# ============================================================

def find_best_threshold(model, loader, device):
    model.eval()
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X)).cpu().numpy()
            y_prob_all.extend(probs)
            y_true_all.extend(y.numpy())

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1[:-1])
    return float(thresholds[best_idx])

# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X)).cpu().numpy()
            y_prob_all.extend(probs)
            y_true_all.extend(y.numpy())

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc":      roc_auc_score(y_true, y_prob),
        "pr_auc":       average_precision_score(y_true, y_prob),
        "accuracy":     accuracy_score(y_true, y_pred),
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "mcc":          matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "threshold":    float(threshold),
    }, y_true, y_prob, y_pred

# ============================================================
# APPLICABILITY DOMAIN
# ============================================================

def compute_applicability_domain(X_train, X_test, percentile=95):
    """
    Leverage-based applicability domain.
    Threshold computed from training pairwise distances only.
    """
    # Use PCA-reduced space for efficiency (otherwise cdist on 3860 dims is slow)
    from sklearn.decomposition import PCA
    pca     = PCA(n_components=50, random_state=42)
    Xtr_red = pca.fit_transform(X_train)
    Xte_red = pca.transform(X_test)

    # Training pairwise min distances for threshold
    d_train      = cdist(Xtr_red, Xtr_red, metric="euclidean")
    np.fill_diagonal(d_train, np.inf)
    min_d_train  = d_train.min(axis=1)
    ad_threshold = np.percentile(min_d_train, percentile)

    # Test set: distance to nearest training point
    d_test       = cdist(Xte_red, Xtr_red, metric="euclidean")
    min_d_test   = d_test.min(axis=1)
    inside_ad    = min_d_test <= ad_threshold

    return min_d_test, inside_ad, float(ad_threshold)

# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(model, train_loader, val_loader, device, criterion,
                max_epochs=80, patience=10):
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    best_val_loss = np.inf
    best_state    = None
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history

# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def run_training():
    print("\n" + "=" * 60)
    print("Loading hybrid features...")
    print("=" * 60)

    X        = np.load(X_FILE)
    y_raw    = np.load(Y_RAW_FILE)
    df_index = pd.read_csv(INDEX_FILE)

    uniprot_ids = df_index["uniprot_id"].values
    target_names= df_index["target_name"].values

    print(f"  X shape    : {X.shape}")
    print(f"  y_raw shape: {y_raw.shape}")
    print(f"  pChEMBL range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # We need a preliminary labeling just for stratified split purposes.
    # Use global Q1/Q3 here ONLY for fold split, NOT for actual training labels.
    prelim_labels = np.full(len(y_raw), -1, dtype=int)
    for uid in np.unique(uniprot_ids):
        mask = uniprot_ids == uid
        q1   = np.percentile(y_raw[mask], 25)
        q3   = np.percentile(y_raw[mask], 75)
        prelim_labels[mask & (y_raw >= q3)] = 1
        prelim_labels[mask & (y_raw <= q1)] = 0

    # Only use labeled rows for CV (middle 50% excluded)
    labeled_mask = prelim_labels >= 0
    X_labeled    = X[labeled_mask]
    y_raw_labeled= y_raw[labeled_mask]
    uid_labeled  = uniprot_ids[labeled_mask]
    tname_labeled= target_names[labeled_mask]
    strat_labels = prelim_labels[labeled_mask]

    print(f"\n  Labeled samples (top/bottom quartiles): {labeled_mask.sum()}")
    print(f"  Active (1): {(strat_labels==1).sum()}")
    print(f"  Inactive (0): {(strat_labels==0).sum()}")

    # Create a compound stratification key: label + target
    # This ensures each fold has proportional representation of each target
    strat_key = np.array([f"{uid}_{lbl}"
                          for uid, lbl in zip(uid_labeled, strat_labels)])

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    for fold, (train_val_idx, test_idx) in enumerate(
            skf.split(X_labeled, strat_key), start=1):

        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        # ---- CORRECT LABELING: Q1/Q3 from TRAINING SPLIT ONLY ----
        X_train_val  = X_labeled[train_val_idx]
        y_raw_tv     = y_raw_labeled[train_val_idx]
        uid_tv       = uid_labeled[train_val_idx]

        X_test_raw   = X_labeled[test_idx]
        y_raw_test   = y_raw_labeled[test_idx]
        uid_test     = uid_labeled[test_idx]

        # Compute Q1/Q3 thresholds from training data, apply to both
        all_idx      = np.concatenate([train_val_idx, test_idx])
        y_raw_all    = y_raw_labeled[all_idx]
        uid_all      = uid_labeled[all_idx]

        # train_val portion indices within the concatenated array
        tv_pos       = np.arange(len(train_val_idx))

        labels_all, thresholds = assign_quartile_labels(
            pchembl_train = y_raw_tv,
            pchembl_all   = y_raw_all,
            uniprot_train = uid_tv,
            uniprot_all   = uid_all,
        )

        labels_tv   = labels_all[:len(train_val_idx)]
        labels_test = labels_all[len(train_val_idx):]

        # Remove middle-zone rows (NaN labels)
        tv_keep     = ~np.isnan(labels_tv)
        test_keep   = ~np.isnan(labels_test)

        X_tv_clean   = X_train_val[tv_keep]
        y_tv_clean   = labels_tv[tv_keep].astype(int)
        uid_tv_clean = uid_tv[tv_keep]

        X_test_clean  = X_test_raw[test_keep]
        y_test_clean  = labels_test[test_keep].astype(int)
        uid_test_clean= uid_test[test_keep]

        print(f"  Train+val samples : {len(y_tv_clean)}")
        print(f"  Test samples      : {len(y_test_clean)}")
        print(f"  Thresholds (Q1, Q3) per target:")
        for uid, (q1, q3) in thresholds.items():
            tname = df_index[df_index["uniprot_id"]==uid]["target_name"].iloc[0]
            print(f"    {tname:8s} | Q1={q1:.3f} | Q3={q3:.3f}")

        # Save thresholds
        with open(os.path.join(MODEL_DIR, f"thresholds_fold_{fold}.json"), "w") as f:
            json.dump(thresholds, f, indent=2)

        # ---- CORRECT SCALING: fit on TRAINING SPLIT ONLY ----
        tr_idx_inner, val_idx_inner = train_test_split(
            np.arange(len(X_tv_clean)),
            test_size=0.15,
            stratify=y_tv_clean,
            random_state=SEED
        )

        scaler = StandardScaler()
        X_tv_clean[:, DESC_START:DESC_END] = scaler.fit_transform(
            X_tv_clean[:, DESC_START:DESC_END]
        )
        # Apply same scaler to test
        X_test_clean[:, DESC_START:DESC_END] = scaler.transform(
            X_test_clean[:, DESC_START:DESC_END]
        )

        joblib.dump(scaler, os.path.join(MODEL_DIR, f"descriptor_scaler_fold_{fold}.pkl"))

        X_tr  = X_tv_clean[tr_idx_inner]
        y_tr  = y_tv_clean[tr_idx_inner]
        X_val = X_tv_clean[val_idx_inner]
        y_val = y_tv_clean[val_idx_inner]

        print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test_clean)}")
        print(f"  Train class balance: {y_tr.mean():.3f}")

        # ---- CLASS WEIGHT ----
        pos_count  = float(y_tr.sum())
        neg_count  = float(len(y_tr) - pos_count)
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)]).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ---- DATA LOADERS ----
        train_loader = DataLoader(HybridDataset(X_tr,  y_tr),
                                  batch_size=128, shuffle=True,  drop_last=True)
        val_loader   = DataLoader(HybridDataset(X_val, y_val),
                                  batch_size=128, shuffle=False)
        test_loader  = DataLoader(HybridDataset(X_test_clean, y_test_clean),
                                  batch_size=128, shuffle=False)

        # ---- TRAIN ----
        model = HybridPCM(TOTAL_DIM).to(device)
        model, history = train_model(
            model, train_loader, val_loader, device, criterion,
            max_epochs=100, patience=12
        )

        # ---- THRESHOLD FROM VALIDATION SET ----
        best_thr = find_best_threshold(model, val_loader, device)
        with open(os.path.join(MODEL_DIR, f"threshold_fold_{fold}.json"), "w") as f:
            json.dump({"optimal_threshold": best_thr}, f, indent=2)

        # ---- EVALUATE ON TEST SET ----
        metrics, y_true, y_prob, y_pred = evaluate(
            model, test_loader, device, threshold=best_thr
        )
        metrics["fold"] = fold
        results.append(metrics)

        print(f"\n  Test ROC-AUC : {metrics['roc_auc']:.4f}")
        print(f"  Test F1      : {metrics['f1']:.4f}")
        print(f"  Test MCC     : {metrics['mcc']:.4f}")
        print(f"  Threshold    : {best_thr:.4f}")

        # ---- SAVE PREDICTIONS ----
        pred_df = pd.DataFrame({
            "true":  y_true,
            "prob":  y_prob,
            "pred":  y_pred,
            "uniprot_id": uid_test_clean,
        })
        pred_df.to_csv(os.path.join(PRED_DIR, f"fold_{fold}_predictions.csv"),
                       index=False)

        # ---- CONFUSION MATRIX PLOT ----
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Inactive", "Active"],
                    yticklabels=["Inactive", "Active"])
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"Confusion Matrix — Fold {fold}\n"
                     f"ROC-AUC={metrics['roc_auc']:.3f}, F1={metrics['f1']:.3f}",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"confusion_fold_{fold}.png"), dpi=300)
        plt.close()

        # ---- TRAINING HISTORY PLOT ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["train_loss"], label="Train loss")
        ax.plot(history["val_loss"],   label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title(f"Training History — Fold {fold}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"training_history_fold_{fold}.png"), dpi=300)
        plt.close()

        # ---- APPLICABILITY DOMAIN ----
        min_dist, inside_ad, ad_thr = compute_applicability_domain(
            X_tr, X_test_clean
        )
        ad_df = pd.DataFrame({
            "distance":  min_dist,
            "inside_AD": inside_ad,
            "true":      y_true,
            "prob":      y_prob,
        })
        ad_df["uniprot_id"] = uid_test_clean
        ad_df.to_csv(os.path.join(AD_DIR, f"fold_{fold}_AD.csv"), index=False)

        # AD coverage
        ad_coverage = inside_ad.mean()
        print(f"  AD coverage  : {ad_coverage:.3f} ({inside_ad.sum()}/{len(inside_ad)})")

        # ---- SAVE MODEL ----
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, f"model_fold_{fold}.pt"))

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(METRIC_DIR, "cv_metrics.csv"), index=False)

    summary = {}
    for metric in ["roc_auc", "pr_auc", "f1", "mcc", "balanced_acc",
                   "accuracy", "precision", "recall"]:
        vals     = results_df[metric].values
        mean_val = float(np.mean(vals))
        ci       = float(stats.sem(vals) * stats.t.ppf(0.975, len(vals) - 1))
        summary[metric] = {
            "mean":   round(mean_val, 4),
            "ci_95":  round(ci, 4),
            "values": [round(v, 4) for v in vals.tolist()],
            "str":    f"{mean_val:.4f} ± {ci:.4f}",
        }

    with open(os.path.join(METRIC_DIR, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal CV Performance:")
    for m, v in summary.items():
        print(f"  {m:15s}: {v['str']}")
    print(f"\n  Mean optimal threshold: {results_df['threshold'].mean():.4f}")

    # ============================================================
    # PUBLICATION PLOTS
    # ============================================================

    # --- ROC curves ---
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs     = []
    for fold_i in range(1, 6):
        pred = pd.read_csv(os.path.join(PRED_DIR, f"fold_{fold_i}_predictions.csv"))
        fpr, tpr, _ = roc_curve(pred["true"], pred["prob"])
        auc_val = roc_auc_score(pred["true"], pred["prob"])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, alpha=0.4, lw=1, label=f"Fold {fold_i} (AUC={auc_val:.3f})")
    mean_tpr = np.mean(tprs, axis=0)
    ax.plot(mean_fpr, mean_tpr, "b-", lw=2.5,
            label=f"Mean (AUC={summary['roc_auc']['str']})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — 5-Fold Cross-Validation", fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ROC_curves.png"), dpi=600)
    plt.close()

    # --- PR curves ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for fold_i in range(1, 6):
        pred = pd.read_csv(os.path.join(PRED_DIR, f"fold_{fold_i}_predictions.csv"))
        p, r, _ = precision_recall_curve(pred["true"], pred["prob"])
        pr_auc  = average_precision_score(pred["true"], pred["prob"])
        ax.plot(r, p, alpha=0.5, lw=1.5, label=f"Fold {fold_i} (AP={pr_auc:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — 5-Fold CV", fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "PR_curves.png"), dpi=600)
    plt.close()

    # --- Calibration plot ---
    all_true, all_prob = [], []
    for fold_i in range(1, 6):
        pred = pd.read_csv(os.path.join(PRED_DIR, f"fold_{fold_i}_predictions.csv"))
        all_true.extend(pred["true"].tolist())
        all_prob.extend(pred["prob"].tolist())
    prob_true, prob_pred = calibration_curve(all_true, all_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "o-", lw=2, label="PCM Model")
    ax.plot([0,1],[0,1],"k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Plot (Reliability Diagram)", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "calibration_plot.png"), dpi=600)
    plt.close()

    # --- Metric summary bar chart ---
    metric_keys  = ["roc_auc", "pr_auc", "f1", "mcc", "balanced_acc"]
    metric_means = [summary[m]["mean"] for m in metric_keys]
    metric_cis   = [summary[m]["ci_95"] for m in metric_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metric_keys, metric_means, yerr=metric_cis,
                  capsize=6, color="steelblue", alpha=0.85, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("PCM Model Performance (5-Fold CV, mean ± 95% CI)", fontsize=12)
    for bar, val, ci in zip(bars, metric_means, metric_cis):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + ci + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "metric_summary.png"), dpi=600)
    plt.close()

    # --- Per-target metrics ---
    per_target_rows = []
    for fold_i in range(1, 6):
        pred = pd.read_csv(os.path.join(PRED_DIR, f"fold_{fold_i}_predictions.csv"))
        for uid in pred["uniprot_id"].unique():
            sub = pred[pred["uniprot_id"] == uid]
            if len(sub) < 5 or sub["true"].nunique() < 2:
                continue
            per_target_rows.append({
                "fold":       fold_i,
                "uniprot_id": uid,
                "n":          len(sub),
                "roc_auc":    roc_auc_score(sub["true"], sub["prob"]),
                "f1":         f1_score(sub["true"],
                                       (sub["prob"] >= 0.5).astype(int),
                                       zero_division=0),
            })
    if per_target_rows:
        pt_df = pd.DataFrame(per_target_rows)
        pt_df.to_csv(os.path.join(METRIC_DIR, "per_target_metrics.csv"), index=False)

        pt_mean = pt_df.groupby("uniprot_id")["roc_auc"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(pt_mean["uniprot_id"], pt_mean["roc_auc"],
               color="teal", alpha=0.8, edgecolor="black")
        ax.axhline(0.5, color="red", linestyle="--", lw=1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean ROC-AUC (across folds)", fontsize=11)
        ax.set_title("Per-Target ROC-AUC", fontsize=12)
        ax.set_xticklabels(pt_mean["uniprot_id"], rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "per_target_roc_auc.png"), dpi=600)
        plt.close()

    print(f"\n  All outputs saved to: {TRAIN_DIR}")
    return results_df, summary, X_labeled, strat_labels

# ============================================================
# Y-RANDOMIZATION TEST
# ============================================================

def y_randomization_test(X, y, original_auc, n_iter=10):
    """
    Permutation test: shuffle labels and train a lightweight proxy model.
    A valid model should show >> 0.20 gap between original and random AUC.
    """
    print("\n" + "=" * 60)
    print("Y-RANDOMIZATION TEST")
    print("=" * 60)
    print(f"  Running {n_iter} permutation iterations...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler as SS

    random_aucs = []
    for i in range(n_iter):
        y_perm  = np.random.permutation(y)
        pipe    = Pipeline([("sc", SS()), ("lr", LogisticRegression(max_iter=500,
                                                                      random_state=i))])
        # Use only a random 30% feature subset for speed
        feat_idx = np.random.choice(X.shape[1], size=min(500, X.shape[1]), replace=False)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(pipe, X[:, feat_idx], y_perm,
                                 cv=5, scoring="roc_auc", n_jobs=-1)
        mean_auc = scores.mean()
        random_aucs.append(mean_auc)
        print(f"  Iter {i+1:2d}: AUC = {mean_auc:.4f}")

    mean_rand = float(np.mean(random_aucs))
    std_rand  = float(np.std(random_aucs))
    gap       = original_auc - mean_rand

    rand_results = {
        "original_auc":    round(original_auc, 4),
        "random_mean_auc": round(mean_rand, 4),
        "random_std":      round(std_rand, 4),
        "gap":             round(gap, 4),
        "all_random_aucs": [round(a, 4) for a in random_aucs],
        "verdict":         "VALID (gap > 0.30)" if gap > 0.30 else
                           "GOOD (gap > 0.20)"  if gap > 0.20 else
                           "WARNING: small gap — investigate",
    }

    with open(os.path.join(METRIC_DIR, "y_randomization.json"), "w") as f:
        json.dump(rand_results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(random_aucs, bins=10, color="salmon", edgecolor="black",
            alpha=0.8, label="Y-randomized AUC")
    ax.axvline(original_auc, color="steelblue", lw=2.5,
               label=f"Original AUC = {original_auc:.4f}")
    ax.axvline(mean_rand,   color="red",      lw=1.5, linestyle="--",
               label=f"Random mean = {mean_rand:.4f}")
    ax.set_xlabel("ROC-AUC", fontsize=12)
    ax.set_ylabel("Count",   fontsize=12)
    ax.set_title("Y-Randomization Test", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "y_randomization.png"), dpi=600)
    plt.close()

    print(f"\n  Original AUC   : {original_auc:.4f}")
    print(f"  Random mean    : {mean_rand:.4f} ± {std_rand:.4f}")
    print(f"  Gap            : {gap:.4f}")
    print(f"  Verdict        : {rand_results['verdict']}")

    return rand_results

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results_df, summary, X_labeled, y_labeled = run_training()
    original_auc = summary["roc_auc"]["mean"]
    y_randomization_test(X_labeled, y_labeled, original_auc, n_iter=10)