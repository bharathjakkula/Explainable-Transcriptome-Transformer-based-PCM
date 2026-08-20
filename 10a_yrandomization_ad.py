import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = r"D:\MultiModal_Classification"
FEAT_DIR = os.path.join(BASE_DIR, "output", "08_pcm_features", "hybrid_matrix")             # 08d output
TRAIN_DIR = os.path.join(BASE_DIR, "output", "09_pcm_training_screening", "training_cv")    # 09a output
METRIC_DIR = os.path.join(TRAIN_DIR, "metrics")

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "10_reviewer2_pt4_yrandom_ad")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_FILE = os.path.join(FEAT_DIR, "X_hybrid.npy")
Y_RAW_FILE = os.path.join(FEAT_DIR, "y_raw.npy")
INDEX_FILE = os.path.join(FEAT_DIR, "feature_index.csv")

SEED = 42
N_FOLDS = 5
N_PERMUTATIONS = 100     # literature-typical range 100-1000; reduce if too slow
AD_PERCENTILE = 95       # unchanged from the original AD function
AD_PCA_COMPONENTS = 50   # unchanged from the original AD function

# Fallback if cv_metrics.csv is not found -- set this to your verified
# reported mean ROC-AUC from the manuscript/notebook output if needed.
FALLBACK_REAL_AUC = 0.9696


# ============================================================================
# Verbatim
# ============================================================================

def assign_quartile_labels(pchembl_train, pchembl_all, uniprot_train, uniprot_all):
    thresholds = {}
    n = len(pchembl_all)
    labels = np.full(n, np.nan)
    for uid in np.unique(uniprot_train):
        train_mask = uniprot_train == uid
        train_vals = pchembl_train[train_mask]
        q1 = np.percentile(train_vals, 25)
        q3 = np.percentile(train_vals, 75)
        thresholds[uid] = (float(q1), float(q3))
        all_mask = uniprot_all == uid
        all_vals = pchembl_all[all_mask]
        fold_labels = np.full(len(all_vals), np.nan)
        fold_labels[all_vals >= q3] = 1.0
        fold_labels[all_vals <= q1] = 0.0
        labels[all_mask] = fold_labels
    return labels, thresholds


def compute_applicability_domain(X_train, X_test, percentile=AD_PERCENTILE):
    """Leverage-based applicability domain -- identical to the training
    notebook's function. Threshold computed from training pairwise
    distances only."""
    pca = PCA(n_components=AD_PCA_COMPONENTS, random_state=SEED)
    Xtr_red = pca.fit_transform(X_train)
    Xte_red = pca.transform(X_test)

    d_train = cdist(Xtr_red, Xtr_red, metric="euclidean")
    np.fill_diagonal(d_train, np.inf)
    min_d_train = d_train.min(axis=1)
    ad_threshold = np.percentile(min_d_train, percentile)

    d_test = cdist(Xte_red, Xtr_red, metric="euclidean")
    min_d_test = d_test.min(axis=1)
    inside_ad = min_d_test <= ad_threshold

    return min_d_test, inside_ad, float(ad_threshold)


def load_labeled_data():
    X = np.load(X_FILE)
    y_raw = np.load(Y_RAW_FILE)
    df_index = pd.read_csv(INDEX_FILE)
    uniprot_ids = df_index["uniprot_id"].values
    target_names = df_index["target_name"].values

    prelim_labels = np.full(len(y_raw), -1, dtype=int)
    for uid in np.unique(uniprot_ids):
        mask = uniprot_ids == uid
        q1 = np.percentile(y_raw[mask], 25)
        q3 = np.percentile(y_raw[mask], 75)
        prelim_labels[mask & (y_raw >= q3)] = 1
        prelim_labels[mask & (y_raw <= q1)] = 0

    labeled_mask = prelim_labels >= 0
    X_labeled = X[labeled_mask]
    y_raw_labeled = y_raw[labeled_mask]
    uid_labeled = uniprot_ids[labeled_mask]
    tname_labeled = target_names[labeled_mask]
    strat_labels = prelim_labels[labeled_mask]

    print(f"Loaded {X.shape[0]} total rows, {X.shape[1]} features.")
    print(f"Labeled (top/bottom quartile) subset: {labeled_mask.sum()} rows.")
    return X_labeled, y_raw_labeled, uid_labeled, tname_labeled, strat_labels


# ============================================================================
# PART (a) — Full-dimension Y-randomization
# ============================================================================

def get_real_auc():
    cv_path = os.path.join(METRIC_DIR, "cv_metrics.csv")
    if os.path.exists(cv_path):
        cv_df = pd.read_csv(cv_path)
        if "roc_auc" in cv_df.columns:
            real_auc = float(cv_df["roc_auc"].mean())
            print(f"Real model mean ROC-AUC (from cv_metrics.csv): {real_auc:.4f}")
            return real_auc
    print(f"[WARNING] Could not find/read {cv_path} -- using FALLBACK_REAL_AUC "
          f"= {FALLBACK_REAL_AUC}. Verify this matches your reported CV summary.")
    return FALLBACK_REAL_AUC


def run_full_dimension_y_randomization(X, y, real_auc):
    proxies = {
        "LogisticRegression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None, n_jobs=-1
        ),
    }

    rows = []
    for proxy_name, base_pipe in proxies.items():
        print(f"\nRunning {N_PERMUTATIONS} full-dimension permutations with proxy: {proxy_name}")
        for i in range(N_PERMUTATIONS):
            y_perm = np.random.RandomState(SEED + i).permutation(y)
            if proxy_name == "LogisticRegression":
                pipe = Pipeline([("sc", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=1000, random_state=i))])
            else:
                pipe = RandomForestClassifier(n_estimators=100, max_depth=None,
                                               n_jobs=-1, random_state=i)
            scores = cross_val_score(pipe, X, y_perm, cv=N_FOLDS, scoring="roc_auc", n_jobs=-1)
            mean_auc = scores.mean()
            rows.append({"proxy": proxy_name, "iteration": i + 1, "auc": mean_auc})
            if (i + 1) % 10 == 0:
                print(f"  {proxy_name} iter {i+1}/{N_PERMUTATIONS}: AUC={mean_auc:.4f}")

    full_df = pd.DataFrame(rows)
    full_df.to_csv(os.path.join(OUTPUT_DIR, "ST_YRandomization_full.csv"), index=False)

    summary_rows = []
    for proxy_name in proxies:
        sub = full_df[full_df["proxy"] == proxy_name]["auc"].values
        mean_rand = float(sub.mean())
        std_rand = float(sub.std())
        gap = real_auc - mean_rand
        n_ge = int((sub >= real_auc).sum())
        p_value = (n_ge + 1) / (len(sub) + 1)
        summary_rows.append({
            "proxy": proxy_name,
            "real_auc": real_auc,
            "random_mean_auc": mean_rand,
            "random_std_auc": std_rand,
            "random_min_auc": float(sub.min()),
            "random_max_auc": float(sub.max()),
            "gap": gap,
            "n_permutations": len(sub),
            "n_permutations_ge_real": n_ge,
            "empirical_p_value": p_value,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "ST_YRandomization_summary.csv"), index=False)
    print("\n" + summary_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proxy_name in zip(axes, proxies):
        sub = full_df[full_df["proxy"] == proxy_name]["auc"].values
        ax.hist(sub, bins=20, color="salmon", edgecolor="black", alpha=0.8,
                label=f"{proxy_name} (permuted labels)")
        ax.axvline(real_auc, color="steelblue", lw=2.5, label=f"Real model AUC = {real_auc:.4f}")
        ax.axvline(sub.mean(), color="red", lw=1.5, linestyle="--",
                    label=f"Permuted mean = {sub.mean():.4f}")
        ax.set_xlabel("ROC-AUC")
        ax.set_ylabel("Count")
        ax.set_title(f"{proxy_name}\nFull-dimension ({X.shape[1]} features), "
                      f"{N_PERMUTATIONS} permutations")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_YRandomization_fulldim.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return summary_df


# ============================================================================
# PART (b) — Per-target applicability domain coverage
# ============================================================================

def run_per_target_ad(X_labeled, y_raw_labeled, uid_labeled, tname_labeled, strat_labels):
    strat_key = np.array([f"{uid}_{lbl}" for uid, lbl in zip(uid_labeled, strat_labels)])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_rows = []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_labeled, strat_key), start=1):
        print(f"\nReproducing Fold {fold} split for AD recomputation (no model training)...")

        X_train_val = X_labeled[train_val_idx]
        y_raw_tv = y_raw_labeled[train_val_idx]
        uid_tv = uid_labeled[train_val_idx]

        X_test_raw = X_labeled[test_idx]
        y_raw_test = y_raw_labeled[test_idx]
        uid_test = uid_labeled[test_idx]

        all_idx = np.concatenate([train_val_idx, test_idx])
        y_raw_all = y_raw_labeled[all_idx]
        uid_all = uid_labeled[all_idx]

        labels_all, _thresholds = assign_quartile_labels(
            pchembl_train=y_raw_tv, pchembl_all=y_raw_all,
            uniprot_train=uid_tv, uniprot_all=uid_all,
        )
        labels_tv = labels_all[:len(train_val_idx)]
        labels_test = labels_all[len(train_val_idx):]

        tv_keep = ~np.isnan(labels_tv)
        test_keep = ~np.isnan(labels_test)

        X_tv_clean = X_train_val[tv_keep].copy()
        uid_tv_clean = uid_tv[tv_keep]

        X_test_clean = X_test_raw[test_keep].copy()
        uid_test_clean = uid_test[test_keep]
        tname_test = tname_labeled[test_idx][test_keep]

        # Reproduce the same inner train/val split + descriptor scaling
        # sequence as the training notebook, so X_tr matches exactly.
        y_tv_clean = labels_tv[tv_keep].astype(int)
        with open(os.path.join(FEAT_DIR, "feature_layout.json")) as f:
            layout = json.load(f)
        DESC_START, DESC_END = layout["DESC_START"], layout["DESC_END"]

        tr_idx_inner, val_idx_inner = train_test_split(
            np.arange(len(X_tv_clean)), test_size=0.15,
            stratify=y_tv_clean, random_state=SEED
        )
        scaler = StandardScaler()
        X_tv_clean[:, DESC_START:DESC_END] = scaler.fit_transform(X_tv_clean[:, DESC_START:DESC_END])
        X_test_clean[:, DESC_START:DESC_END] = scaler.transform(X_test_clean[:, DESC_START:DESC_END])
        X_tr = X_tv_clean[tr_idx_inner]

        min_dist, inside_ad, ad_thr = compute_applicability_domain(X_tr, X_test_clean)
        print(f"  Fold {fold} AD coverage (overall, matches original): "
              f"{inside_ad.mean():.3f} ({inside_ad.sum()}/{len(inside_ad)})")

        fold_df = pd.DataFrame({
            "fold": fold,
            "uniprot_id": uid_test_clean,
            "target_name": tname_test,
            "distance": min_dist,
            "inside_AD": inside_ad,
            "ad_threshold": ad_thr,
        })
        all_rows.append(fold_df)

    all_ad_df = pd.concat(all_rows, ignore_index=True)
    all_ad_df.to_csv(os.path.join(OUTPUT_DIR, "ST_AD_per_target_full.csv"), index=False)

    per_target = all_ad_df.groupby("target_name").agg(
        N_test_rows=("inside_AD", "size"),
        N_inside_AD=("inside_AD", "sum"),
        AD_coverage=("inside_AD", "mean"),
    ).reset_index().sort_values("AD_coverage")
    per_target.to_csv(os.path.join(OUTPUT_DIR, "ST_AD_per_target_summary.csv"), index=False)
    print("\n" + per_target.to_string(index=False))

    overall_coverage = all_ad_df["inside_AD"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["salmon" if c < overall_coverage - 0.05 else "teal" for c in per_target["AD_coverage"]]
    ax.barh(per_target["target_name"], per_target["AD_coverage"], color=colors, edgecolor="black")
    ax.axvline(overall_coverage, color="black", linestyle="--", lw=1.5,
                label=f"Overall coverage = {overall_coverage:.3f}")
    ax.set_xlabel("AD coverage (fraction of test rows within domain)")
    ax.set_title("Per-Target Applicability Domain Coverage\n(all 5 folds combined)")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig_AD_per_target_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return per_target, overall_coverage


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Loading labeled dataset (same construction as training notebook)")
    print("=" * 70)
    X_labeled, y_raw_labeled, uid_labeled, tname_labeled, strat_labels = load_labeled_data()

    print("\n" + "=" * 70)
    print("PART (a): Full-dimension Y-randomization")
    print("=" * 70)
    real_auc = get_real_auc()
    yrand_summary = run_full_dimension_y_randomization(X_labeled, strat_labels, real_auc)

    print("\n" + "=" * 70)
    print("PART (b): Per-target applicability domain coverage")
    print("=" * 70)
    per_target_ad, overall_coverage = run_per_target_ad(
        X_labeled, y_raw_labeled, uid_labeled, tname_labeled, strat_labels
    )

    # ---- Auto-generated summary ----
    summary_lines = ["COMMENT 4 (a & b) -- ANALYSIS SUMMARY", "=" * 40]
    for _, row in yrand_summary.iterrows():
        summary_lines.append(
            f"- Y-randomization ({row['proxy']}, full {X_labeled.shape[1]}-dim features, "
            f"{int(row['n_permutations'])} permutations): permuted-label AUC = "
            f"{row['random_mean_auc']:.4f} +/- {row['random_std_auc']:.4f} "
            f"(range {row['random_min_auc']:.4f}-{row['random_max_auc']:.4f}); "
            f"real model AUC = {row['real_auc']:.4f}; gap = {row['gap']:.4f}; "
            f"empirical p-value = {row['empirical_p_value']:.4f} "
            f"({int(row['n_permutations_ge_real'])}/{int(row['n_permutations'])} permutations "
            f"reached or exceeded the real AUC)."
        )
    summary_lines.append(
        f"- Per-target AD coverage ranges {per_target_ad['AD_coverage'].min():.3f}-"
        f"{per_target_ad['AD_coverage'].max():.3f} across the 11 targets "
        f"(overall pooled coverage {overall_coverage:.3f}); see "
        f"ST_AD_per_target_summary.csv and Fig_AD_per_target_bar.png for the full breakdown."
    )
    summary_lines.append(
        "- AD recomputation used the identical fold splits, quartile-labeling rule, and "
        "leverage-based AD function already in the training notebook -- no retraining, "
        "no change to the underlying method, only the added target-identity breakdown."
    )
    summary_text = "\n".join(summary_lines)
    with open(os.path.join(OUTPUT_DIR, "Analysis_summary.txt"), "w") as f:
        f.write(summary_text + "\n")
    print("\n" + summary_text)
    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()