# =============================================================================
# Purpose:
#   1. Clinical cohort comparability
#   2. Leakage-safe stratified 5-fold CV
#   3. Per-cohort performance
#   4. 95% CI calculation for Accuracy, Macro-F1, MCC and Macro-AUC
#   5. TRIPOD-AI checklist
# =============================================================================

import os
import sys
import platform
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels
import xgboost

from scipy import stats
import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

from xgboost import XGBClassifier


# ============================================================================
# CONFIG — edit these four paths only
# ============================================================================

BASE_DIR = r"D:\MultiModal_Classification"

COMBINED_EXPR_CSV = os.path.join(
    BASE_DIR,
    "output",
    "02_batch_correction_deg_pca",
    "combined_after_batch_correction.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "07_cohort_comparability"
)

# Produced by the "Checkpoint" save added to 01a/01b/01c
CLINICAL_2012 = os.path.join(
    BASE_DIR,
    "output",
    "01_preprocessing",
    "tcga2012",
    "merged_clinical_2012_Harmonised.csv"
)

CLINICAL_2016 = os.path.join(
    BASE_DIR,
    "output",
    "01_preprocessing",
    "metabric2016",
    "merged_clinical_2016_Harmonised.csv"
)

CLINICAL_2018 = os.path.join(
    BASE_DIR,
    "output",
    "01_preprocessing",
    "tcga2018",
    "merged_clinical_2018_Harmonised.csv"
)


# ============================================================================
# VALUE MAPS
# ============================================================================

VALUE_MAPS = {
    "2012": {
        "SUBTYPE": {
            "Basal-like": "Basal",
            "Normal-like": "Normal",
            "HER2-enriched": "Her2"
        },
        "TUMOR_STAGE": {
            "T1": "STAGE I",
            "T2": "STAGE II",
            "T3": "STAGE III",
            "T4": "STAGE IV",
            "TX": "STAGE I"
        },
        "OS_STATUS": {
            "1:DECEASED": "DISEASED",
            "0:LIVING": "LIVING"
        },
    },

    "2016": {
        "SUBTYPE": {
            "LumA": "Luminal A",
            "LumB": "Luminal B"
        },
        "TUMOR_STAGE": {
            "0": "STAGE 0",
            "1": "STAGE I",
            "2": "STAGE II",
            "3": "STAGE III",
            "4": "STAGE IV"
        },
        "OS_STATUS": {
            "1:DECEASED": "DISEASED",
            "0:LIVING": "LIVING"
        },
    },

    "2018": {
        "SUBTYPE": {
            "BRCA_LumA": "Luminal A",
            "BRCA_LumB": "Luminal B",
            "BRCA_Her2": "Her2",
            "BRCA_Basal": "Basal",
            "BRCA_Normal": "Normal"
        },
        "TUMOR_STAGE": {
            "STAGE IA": "STAGE I",
            "STAGE IB": "STAGE I",
            "STAGE IIA": "STAGE II",
            "STAGE IIB": "STAGE II",
            "STAGE IIIA": "STAGE III",
            "STAGE IIIB": "STAGE III",
            "STAGE IIIC": "STAGE III",
            "STAGE X": "STAGE 0"
        },
        "OS_STATUS": {
            "1:DECEASED": "DISEASED",
            "0:LIVING": "LIVING"
        },
    },
}


# ============================================================================
# CONSTANTS
# ============================================================================

COMMON_SUBTYPES_A = [
    "Luminal A",
    "Luminal B",
    "Basal",
    "Her2",
    "Normal"
]

COMMON_STAGES_A = [
    "STAGE 0",
    "STAGE I",
    "STAGE II",
    "STAGE III",
    "STAGE IV"
]

REQUIRED_CLINICAL_COLS = [
    "PATIENT_ID",
    "SUBTYPE",
    "AGE",
    "SEX",
    "TUMOR_STAGE",
    "OS_STATUS",
    "OS_MONTHS"
]

META_COLS = [
    "PATIENT_ID",
    "SUBTYPE",
    "AGE",
    "SEX",
    "TUMOR_STAGE",
    "OS_STATUS",
    "OS_MONTHS",
    "BATCH"
]

LABEL_COL = "SUBTYPE"

BATCH_LABEL_MAP = {
    "batch_2012": "TCGA-BRCA-2012",
    "batch_2016": "METABRIC-BRCA-2016",
    "batch_2018": "TCGA-BRCA-2018",
}

SUBTYPE_MAP = {
    0: "Normal",
    1: "Luminal A",
    2: "Luminal B",
    3: "HER2",
    4: "Basal"
}

STAGE_MAP = {
    0: "Stage I",
    1: "Stage II",
    2: "Stage III",
    3: "Stage IV"
}


# ============================================================================
# CV / FEATURE-SELECTION SETTINGS
# ============================================================================

N_FOLDS = 5
RANDOM_STATE = 42

DEG_TOP_N_PER_SUBTYPE_POOL = 200
TOP_N_SIGNATURE_PER_CLASS = 10

PCA_VARIANCE = 0.90

EFFECT_SIZE_THRESH = 0.3
PVAL_THRESHOLD = 0.05


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def reproducibility_table():
    return pd.DataFrame([
        {
            "Item": "Python",
            "Value": platform.python_version()
        },
        {
            "Item": "numpy",
            "Value": np.__version__
        },
        {
            "Item": "pandas",
            "Value": pd.__version__
        },
        {
            "Item": "scipy",
            "Value": scipy.__version__
        },
        {
            "Item": "statsmodels",
            "Value": statsmodels.__version__
        },
        {
            "Item": "scikit-learn",
            "Value": sklearn.__version__
        },
        {
            "Item": "xgboost",
            "Value": xgboost.__version__
        },
        {
            "Item": "Random seed",
            "Value": str(RANDOM_STATE)
        },
        {
            "Item": "N folds",
            "Value": str(N_FOLDS)
        },
        {
            "Item": "PCA variance threshold",
            "Value": f"{PCA_VARIANCE:.0%}"
        },
        {
            "Item": "CV CI method",
            "Value": "Two-sided Student-t 95% CI"
        },
        {
            "Item": "CV CI unit",
            "Value": "Five fold-level estimates"
        }
    ])


# ============================================================================
# MODULE 1 — CLINICAL VALIDATION + COHORT COMPARABILITY
# ============================================================================

def _normalize_token(x):
    s = str(x).strip()

    try:
        f = float(s)

        if f == int(f):
            return str(int(f))

    except (
        ValueError,
        TypeError
    ):
        pass

    return s


def validate_clinical_file(
    df,
    cohort_label
):
    rows = []

    missing_cols = [
        c
        for c in REQUIRED_CLINICAL_COLS
        if c not in df.columns
    ]

    rows.append({
        "Cohort": cohort_label,
        "Check": "Required columns present",
        "Result": (
            "OK"
            if not missing_cols
            else f"MISSING: {missing_cols}"
        )
    })

    if "PATIENT_ID" in df.columns:

        n_dupe = (
            df["PATIENT_ID"]
            .duplicated()
            .sum()
        )

        rows.append({
            "Cohort": cohort_label,
            "Check": "Duplicate PATIENT_ID",
            "Result": (
                "OK"
                if n_dupe == 0
                else f"{n_dupe} duplicate IDs found"
            )
        })

    for col in REQUIRED_CLINICAL_COLS:

        if col in df.columns:

            n_missing = (
                df[col]
                .isna()
                .sum()
            )

            pct = (
                100 * n_missing / len(df)
                if len(df)
                else 0
            )

            rows.append({
                "Cohort": cohort_label,
                "Check": f"Missing values: {col}",
                "Result": f"{n_missing} ({pct:.1f}%)"
            })

    return rows


def load_individual_clinical_cohort(
    path,
    cohort_key,
    cohort_label,
    validation_rows
):

    df = pd.read_csv(path)

    maps = VALUE_MAPS[
        cohort_key
    ]

    for col, mapping in maps.items():

        if col in df.columns:

            df[col] = (
                df[col]
                .map(
                    lambda x:
                    _normalize_token(x)
                    if pd.notna(x)
                    else x
                )
            )

            df[col] = (
                df[col]
                .map(
                    lambda x, m=mapping:
                    m.get(x, x)
                    if pd.notna(x)
                    else x
                )
            )

    checks = {
        "SUBTYPE": COMMON_SUBTYPES_A,
        "TUMOR_STAGE": COMMON_STAGES_A,
        "OS_STATUS": [
            "DISEASED",
            "LIVING"
        ]
    }

    for col, valid in checks.items():

        if col in df.columns:

            unexpected = (
                set(
                    df[col]
                    .dropna()
                    .unique()
                )
                - set(valid)
            )

            validation_rows.append({
                "Cohort": cohort_label,
                "Check": f"Unexpected values: {col}",
                "Result": (
                    "OK"
                    if not unexpected
                    else f"UNMAPPED: {sorted(unexpected)}"
                )
            })

    validation_rows.extend(
        validate_clinical_file(
            df,
            cohort_label
        )
    )

    df["COHORT"] = cohort_label

    return df


def build_source_cohort_comparability_input(
    validation_rows
):

    df12 = (
        load_individual_clinical_cohort(
            CLINICAL_2012,
            "2012",
            "TCGA-BRCA-2012",
            validation_rows
        )
    )

    df16 = (
        load_individual_clinical_cohort(
            CLINICAL_2016,
            "2016",
            "METABRIC-BRCA-2016",
            validation_rows
        )
    )

    df18 = (
        load_individual_clinical_cohort(
            CLINICAL_2018,
            "2018",
            "TCGA-BRCA-2018",
            validation_rows
        )
    )

    common_cols = [
        "COHORT",
        "AGE",
        "TUMOR_STAGE",
        "SUBTYPE",
        "OS_STATUS",
        "OS_MONTHS"
    ]

    common_cols = [
        c
        for c in common_cols
        if all(
            c in d.columns
            for d in (
                df12,
                df16,
                df18
            )
        )
    ]

    df = pd.concat(
        [
            df12[common_cols],
            df16[common_cols],
            df18[common_cols]
        ],
        ignore_index=True
    )

    df["SUBTYPE_LABEL"] = (
        df["SUBTYPE"]
    )

    df["STAGE_LABEL"] = (
        df["TUMOR_STAGE"]
    )

    grade_summary = None

    if "GRADE" in df16.columns:

        vc = (
            df16["GRADE"]
            .value_counts(
                dropna=False
            )
        )

        pct = (
            100 * vc / vc.sum()
        )

        grade_summary = pd.DataFrame({
            "Grade":
                vc.index.astype(str),
            "N":
                vc.values,
            "Pct":
                [
                    f"{v:.1f}%"
                    for v in pct.values
                ]
        })

        grade_summary["Note"] = (
            "METABRIC-2016 only; not compared statistically "
            "across cohorts (grade is not comparably harmonized "
            "in the TCGA clinical files)"
        )

    counts = {
        "TCGA-BRCA-2012":
            len(df12),
        "METABRIC-BRCA-2016":
            len(df16),
        "TCGA-BRCA-2018":
            len(df18)
    }

    return (
        df,
        grade_summary,
        counts
    )


def kruskal_report(
    df,
    value_col,
    group_col
):

    groups = [
        g[value_col]
        .dropna()
        .values
        for _, g
        in df.groupby(group_col)
    ]

    groups = [
        g
        for g in groups
        if len(g) > 0
    ]

    if len(groups) < 2:
        return (
            np.nan,
            np.nan
        )

    h, pv = stats.kruskal(
        *groups
    )

    return (
        h,
        pv
    )


def chi2_report(
    df,
    value_col,
    group_col
):

    ct = pd.crosstab(
        df[group_col],
        df[value_col]
    )

    if (
        ct.shape[0] < 2
        or ct.shape[1] < 2
    ):
        return (
            np.nan,
            np.nan,
            np.nan
        )

    chi2, pv, dof, _ = (
        stats.chi2_contingency(ct)
    )

    return (
        chi2,
        pv,
        dof
    )


def median_iqr_by_group(
    df,
    value_col,
    group_col
):

    out = {}

    for grp, sub in (
        df.groupby(group_col)
    ):

        s = (
            sub[value_col]
            .dropna()
        )

        if len(s):

            out[grp] = (
                f"{s.median():.1f} "
                f"[{s.quantile(.25):.1f}-"
                f"{s.quantile(.75):.1f}] "
                f"(n={len(s)})"
            )

        else:
            out[grp] = "NA"

    return out


def pct_by_group(
    df,
    value_col,
    group_col
):

    out = {}

    ct = pd.crosstab(
        df[group_col],
        df[value_col]
    )

    pct = (
        ct
        .div(
            ct.sum(axis=1),
            axis=0
        )
        * 100
    )

    for grp in ct.index:

        cats = [
            (
                f"{cat}: "
                f"{ct.loc[grp, cat]} "
                f"({pct.loc[grp, cat]:.1f}%)"
            )
            for cat in ct.columns
        ]

        out[grp] = (
            "; ".join(cats)
        )

    return out


def run_cohort_comparability(
    df
):

    rows = []

    h, pv = (
        kruskal_report(
            df,
            "AGE",
            "COHORT"
        )
    )

    rows.append({
        "Variable":
            "Age at diagnosis (years)",
        "Test":
            "Kruskal-Wallis",
        "Statistic":
            h,
        "p_value":
            pv,
        **median_iqr_by_group(
            df,
            "AGE",
            "COHORT"
        )
    })

    if "OS_MONTHS" in df.columns:

        h, pv = (
            kruskal_report(
                df,
                "OS_MONTHS",
                "COHORT"
            )
        )

        rows.append({
            "Variable":
                "Follow-up (OS months)",
            "Test":
                "Kruskal-Wallis",
            "Statistic":
                h,
            "p_value":
                pv,
            **median_iqr_by_group(
                df,
                "OS_MONTHS",
                "COHORT"
            )
        })

    chi2, pv, dof = (
        chi2_report(
            df,
            "STAGE_LABEL",
            "COHORT"
        )
    )

    rows.append({
        "Variable":
            "Tumour stage distribution",
        "Test":
            f"Chi-square (df={dof})",
        "Statistic":
            chi2,
        "p_value":
            pv,
        **pct_by_group(
            df,
            "STAGE_LABEL",
            "COHORT"
        )
    })

    chi2, pv, dof = (
        chi2_report(
            df,
            "SUBTYPE_LABEL",
            "COHORT"
        )
    )

    rows.append({
        "Variable":
            "PAM50 subtype distribution",
        "Test":
            f"Chi-square (df={dof})",
        "Statistic":
            chi2,
        "p_value":
            pv,
        **pct_by_group(
            df,
            "SUBTYPE_LABEL",
            "COHORT"
        )
    })

    if "OS_STATUS" in df.columns:

        chi2, pv, dof = (
            chi2_report(
                df,
                "OS_STATUS",
                "COHORT"
            )
        )

        rows.append({
            "Variable":
                "Overall survival event status",
            "Test":
                f"Chi-square (df={dof})",
            "Statistic":
                chi2,
            "p_value":
                pv,
            **pct_by_group(
                df,
                "OS_STATUS",
                "COHORT"
            )
        })

    return pd.DataFrame(rows)


# ============================================================================
# PART B — SIGNATURE CONSTRUCTION
# ============================================================================

def safe_ttest(
    a,
    b
):

    if (
        len(a) < 2
        or len(b) < 2
        or (
            np.std(a) == 0
            and np.std(b) == 0
        )
    ):
        return (
            0.0,
            1.0
        )

    return stats.ttest_ind(
        a,
        b,
        equal_var=False
    )


def deg_single_class_vs_rest(
    X_train,
    y_binary,
    gene_names,
    top_n=DEG_TOP_N_PER_SUBTYPE_POOL,
    effect_thresh=EFFECT_SIZE_THRESH,
    qval_thresh=PVAL_THRESHOLD
):

    mask_t = (
        y_binary == 1
    )

    mask_r = (
        y_binary == 0
    )

    if (
        mask_t.sum() < 2
        or mask_r.sum() < 2
    ):
        return []

    Xt = X_train[
        mask_t
    ]

    Xr = X_train[
        mask_r
    ]

    pvals = []
    zdiffs = []

    for j in range(
        X_train.shape[1]
    ):

        _, pv = (
            safe_ttest(
                Xt[:, j],
                Xr[:, j]
            )
        )

        pvals.append(
            pv
        )

        zdiffs.append(
            Xt[:, j].mean()
            - Xr[:, j].mean()
        )

    pvals = np.array(
        pvals
    )

    zdiffs = np.array(
        zdiffs
    )

    qvals = (
        sm.stats.multipletests(
            pvals,
            method="fdr_bh"
        )[1]
    )

    sig_mask = (
        (qvals < qval_thresh)
        & (
            np.abs(zdiffs)
            > effect_thresh
        )
    )

    idx_sig = np.where(
        sig_mask
    )[0]

    if len(idx_sig) == 0:
        return []

    pos_idx = (
        idx_sig[
            zdiffs[idx_sig] > 0
        ]
    )

    neg_idx = (
        idx_sig[
            zdiffs[idx_sig] < 0
        ]
    )

    pos_top = (
        pos_idx[
            np.argsort(
                -zdiffs[pos_idx]
            )[:top_n]
        ]
    )

    neg_top = (
        neg_idx[
            np.argsort(
                zdiffs[neg_idx]
            )[:top_n]
        ]
    )

    genes = [
        gene_names[i]
        for i in np.concatenate(
            [
                pos_top,
                neg_top
            ]
        ).astype(int)
    ]

    return sorted(
        set(genes)
    )


def _pc_relevance_weights(
    scores,
    y_binary
):

    weights = np.zeros(
        scores.shape[1]
    )

    for k in range(
        scores.shape[1]
    ):

        col = scores[:, k]

        if (
            np.std(col) == 0
            or np.std(y_binary) == 0
        ):
            weights[k] = 0.0

        else:

            weights[k] = abs(
                np.corrcoef(
                    col,
                    y_binary
                )[0, 1]
            )

    return weights


def select_top_genes_for_one_subtype(
    X_train_class_deg,
    y_train_binary,
    gene_names_class,
    top_n=TOP_N_SIGNATURE_PER_CLASS,
    variance_threshold=PCA_VARIANCE,
    random_state=RANDOM_STATE
):

    scaler = StandardScaler()

    Xs = (
        scaler.fit_transform(
            X_train_class_deg
        )
    )

    pca_full = PCA(
        random_state=random_state
    )

    pca_full.fit(
        Xs
    )

    cum_var = (
        np.cumsum(
            pca_full
            .explained_variance_ratio_
        )
    )

    n_components = int(
        np.argmax(
            cum_var
            >= variance_threshold
        )
        + 1
    )

    n_components = max(
        1,
        min(
            n_components,
            Xs.shape[1]
        )
    )

    pca = PCA(
        n_components=n_components,
        random_state=random_state
    )

    scores = (
        pca.fit_transform(
            Xs
        )
    )

    weights = (
        _pc_relevance_weights(
            scores,
            y_train_binary
        )
    )

    loadings = (
        pca.components_
    )

    gene_scores = np.sum(
        np.abs(loadings)
        * weights[:, None],
        axis=0
    )

    top_idx = (
        np.argsort(
            -gene_scores
        )[
            :min(
                top_n,
                len(gene_names_class)
            )
        ]
    )

    return (
        [
            gene_names_class[i]
            for i in top_idx
        ],
        n_components
    )


def build_signature_per_subtype(
    X_train,
    y_train,
    gene_names,
    top_n_per_class=TOP_N_SIGNATURE_PER_CLASS,
    deg_top_n=DEG_TOP_N_PER_SUBTYPE_POOL,
    variance_threshold=PCA_VARIANCE,
    random_state=RANDOM_STATE
):
    """
    Called ONLY on outer-fold training data.
    No held-out samples are used for DEG/PCA feature discovery.
    """

    classes = np.unique(
        y_train
    )

    all_selected = []

    for cls in classes:

        mask_t = (
            y_train == cls
        )

        if (
            mask_t.sum() < 2
            or (~mask_t).sum() < 2
        ):
            continue

        deg_genes_cls = (
            deg_single_class_vs_rest(
                X_train,
                mask_t.astype(int),
                gene_names,
                top_n=deg_top_n
            )
        )

        if len(deg_genes_cls) < 2:
            continue

        idx = [
            gene_names.index(g)
            for g in deg_genes_cls
        ]

        Xc = X_train[
            :,
            idx
        ]

        y_bin = (
            mask_t.astype(int)
        )

        top_genes, _n_comp = (
            select_top_genes_for_one_subtype(
                Xc,
                y_bin,
                deg_genes_cls,
                top_n=top_n_per_class,
                variance_threshold=variance_threshold,
                random_state=random_state
            )
        )

        all_selected.extend(
            top_genes
        )

    return sorted(
        set(all_selected)
    )


# ============================================================================
# METRICS
# ============================================================================

def metrics_for_subset(
    y_true,
    y_pred,
    y_prob,
    classes
):

    if len(
        np.unique(y_true)
    ) < 2:

        return {
            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "macro_f1":
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),

            "mcc":
                matthews_corrcoef(
                    y_true,
                    y_pred
                ),

            "macro_auc":
                np.nan,

            "n":
                len(y_true),

            "n_classes":
                len(np.unique(y_true)),

            "auc_status":
                "Undefined: single class in subset"
        }

    y_true_bin = (
        label_binarize(
            y_true,
            classes=classes
        )
    )

    try:

        auc = (
            roc_auc_score(
                y_true_bin,
                y_prob,
                average="macro",
                multi_class="ovr"
            )
        )

        auc_status = "OK"

    except Exception as exc:

        auc = np.nan

        auc_status = (
            "Undefined: "
            + str(exc)
        )

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "mcc":
            matthews_corrcoef(
                y_true,
                y_pred
            ),

        "macro_auc":
            auc,

        "n":
            len(y_true),

        "n_classes":
            len(np.unique(y_true)),

        "auc_status":
            auc_status
    }


# ============================================================================
# 95% CONFIDENCE INTERVAL CALCULATION
# ============================================================================

def mean_sd_ci(
    values
):
    """
    Calculate mean, SD and two-sided 95% Student-t CI.

    The CI is calculated across the available fold-level estimates.

    Formula:
        CI = mean ± t_(0.975, n-1) × SD / sqrt(n)

    For the intended 5-fold analysis:
        df = 4
        t_(0.975, 4) ≈ 2.776

    No value is imputed if a metric is undefined in a fold.
    """

    values = (
        pd.Series(
            values,
            dtype="float64"
        )
        .dropna()
        .to_numpy()
    )

    n = len(
        values
    )

    if n == 0:

        return {
            "n_valid":
                0,

            "mean":
                np.nan,

            "sd":
                np.nan,

            "ci_lower":
                np.nan,

            "ci_upper":
                np.nan
        }

    mean_value = (
        np.mean(
            values
        )
    )

    if n == 1:

        return {
            "n_valid":
                1,

            "mean":
                mean_value,

            "sd":
                np.nan,

            "ci_lower":
                np.nan,

            "ci_upper":
                np.nan
        }

    sd_value = (
        np.std(
            values,
            ddof=1
        )
    )

    sem = (
        sd_value
        / np.sqrt(n)
    )

    t_critical = (
        stats.t.ppf(
            0.975,
            df=n - 1
        )
    )

    margin = (
        t_critical
        * sem
    )

    return {
        "n_valid":
            n,

        "mean":
            mean_value,

        "sd":
            sd_value,

        "ci_lower":
            mean_value - margin,

        "ci_upper":
            mean_value + margin
    }


def summarize_cv_with_ci(
    cv_results_df
):
    """
    Calculate mean, SD and 95% CI for each metric
    within each cohort.

    Metrics:
        Accuracy
        Macro-F1
        MCC
        Macro-AUC
    """

    rows = []

    metric_columns = [
        "accuracy",
        "macro_f1",
        "mcc",
        "macro_auc"
    ]

    for cohort_name, group in (
        cv_results_df.groupby(
            "cohort"
        )
    ):

        row = {
            "cohort":
                cohort_name,

            "n_folds_expected":
                N_FOLDS
        }

        for metric in metric_columns:

            result = (
                mean_sd_ci(
                    group[metric]
                )
            )

            row[
                f"{metric}_n_valid"
            ] = result[
                "n_valid"
            ]

            row[
                f"{metric}_mean"
            ] = result[
                "mean"
            ]

            row[
                f"{metric}_sd"
            ] = result[
                "sd"
            ]

            row[
                f"{metric}_95CI_lower"
            ] = result[
                "ci_lower"
            ]

            row[
                f"{metric}_95CI_upper"
            ] = result[
                "ci_upper"
            ]

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# COHORT PERFORMANCE TABLE
# ============================================================================

def create_manuscript_ready_table(
    cv_ci_summary
):

    manuscript_rows = []

    for _, row in (
        cv_ci_summary.iterrows()
    ):

        def format_metric(
            metric
        ):

            mean_val = row[
                f"{metric}_mean"
            ]

            ci_low = row[
                f"{metric}_95CI_lower"
            ]

            ci_high = row[
                f"{metric}_95CI_upper"
            ]

            n_valid = row[
                f"{metric}_n_valid"
            ]

            if (
                pd.isna(mean_val)
                or pd.isna(ci_low)
                or pd.isna(ci_high)
            ):

                return (
                    f"Not estimable "
                    f"(valid folds={int(n_valid)})"
                )

            return (
                f"{mean_val:.4f} "
                f"({ci_low:.4f}–"
                f"{ci_high:.4f})"
            )

        manuscript_rows.append({

            "Cohort":
                row["cohort"],

            "Accuracy (95% CI)":
                format_metric(
                    "accuracy"
                ),

            "Macro-F1 (95% CI)":
                format_metric(
                    "macro_f1"
                ),

            "MCC (95% CI)":
                format_metric(
                    "mcc"
                ),

            "Macro-AUC (95% CI)":
                format_metric(
                    "macro_auc"
                )
        })

    return pd.DataFrame(
        manuscript_rows
    )


# ============================================================================
# PART B — LEAKAGE-SAFE STRATIFIED CV WITH COHORT BREAKDOWN
# ============================================================================

def run_stratified_cv_with_cohort_breakdown(
    df,
    gene_cols
):
    """
    Leakage-safe stratified 5-fold CV.

    IMPORTANT:
    - CV remains stratified by SUBTYPE, matching Code1.
    - DEG filtering is performed only on the outer training fold.
    - PCA is fitted only on the outer training fold.
    - Gene loading calculation uses only training samples.
    - StandardScaler is fitted only on the training fold.
    - XGBoost is fitted only on the training fold.
    - Performance is evaluated on the held-out fold.
    """

    X_full = (
        df[
            gene_cols
        ]
        .values
        .astype(float)
    )

    y_full = (
        df[
            LABEL_COL
        ]
        .values
    )

    cohort_full = (
        df["COHORT"]
        .values
    )

    classes = np.unique(
        y_full
    )

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    rows = []

    signature_sizes = []

    for fold_i, (
        tr_idx,
        te_idx
    ) in enumerate(
        skf.split(
            X_full,
            y_full
        ),
        start=1
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"FOLD {fold_i}/{N_FOLDS}"
        )

        print(
            "-" * 70
        )

        Xtr_full = (
            X_full[
                tr_idx
            ]
        )

        Xte_full = (
            X_full[
                te_idx
            ]
        )

        ytr = (
            y_full[
                tr_idx
            ]
        )

        yte = (
            y_full[
                te_idx
            ]
        )

        cohort_te = (
            cohort_full[
                te_idx
            ]
        )

        # ---------------------------------------------------------------------
        # Fold-specific DEG + PCA feature selection
        # ---------------------------------------------------------------------

        sig_genes = (
            build_signature_per_subtype(
                Xtr_full,
                ytr,
                gene_cols
            )
        )

        if len(sig_genes) < 2:

            print(
                f"Fold {fold_i}: too few "
                f"signature genes ({len(sig_genes)}), "
                "skipping."
            )

            continue

        signature_sizes.append(
            len(sig_genes)
        )

        sig_idx = [
            gene_cols.index(
                g
            )
            for g in sig_genes
        ]

        Xtr_sig = (
            Xtr_full[
                :,
                sig_idx
            ]
        )

        Xte_sig = (
            Xte_full[
                :,
                sig_idx
            ]
        )

        # ---------------------------------------------------------------------
        # Scaling
        # ---------------------------------------------------------------------

        scaler = StandardScaler()

        Xtr_s = (
            scaler.fit_transform(
                Xtr_sig
            )
        )

        Xte_s = (
            scaler.transform(
                Xte_sig
            )
        )

        # ---------------------------------------------------------------------
        # XGBoost model used by this sensitivity analysis
        # ---------------------------------------------------------------------

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        clf.fit(
            Xtr_s,
            ytr
        )

        y_pred_all = (
            clf.predict(
                Xte_s
            )
        )

        y_prob_all = (
            clf.predict_proba(
                Xte_s
            )
        )

        # ---------------------------------------------------------------------
        # Overall held-out fold
        # ---------------------------------------------------------------------

        overall = (
            metrics_for_subset(
                yte,
                y_pred_all,
                y_prob_all,
                classes
            )
        )

        overall.update({
            "fold":
                fold_i,

            "cohort":
                "ALL (pooled test fold)",

            "n_signature_genes":
                len(sig_genes)
        })

        rows.append(
            overall
        )

        print(
            f"Overall: "
            f"n={len(yte)}, "
            f"accuracy={overall['accuracy']:.4f}, "
            f"macro-F1={overall['macro_f1']:.4f}, "
            f"MCC={overall['mcc']:.4f}, "
            f"macro-AUC={overall['macro_auc']:.4f}"
        )

        # ---------------------------------------------------------------------
        # Per-cohort held-out performance
        # ---------------------------------------------------------------------

        for cohort_name in np.unique(
            cohort_te
        ):

            mask = (
                cohort_te
                == cohort_name
            )

            if mask.sum() == 0:
                continue

            sub = (
                metrics_for_subset(
                    yte[mask],
                    y_pred_all[mask],
                    y_prob_all[mask],
                    classes
                )
            )

            sub.update({
                "fold":
                    fold_i,

                "cohort":
                    cohort_name,

                "n_signature_genes":
                    len(sig_genes)
            })

            rows.append(
                sub
            )

            auc_str = (
                f"{sub['macro_auc']:.4f}"
                if pd.notna(
                    sub["macro_auc"]
                )
                else "UNDEFINED"
            )

            f1_str = (
                f"{sub['macro_f1']:.4f}"
                if pd.notna(
                    sub["macro_f1"]
                )
                else "UNDEFINED"
            )

            print(
                f"{cohort_name}: "
                f"n={sub['n']}, "
                f"accuracy={sub['accuracy']:.4f}, "
                f"macro-F1={f1_str}, "
                f"MCC={sub['mcc']:.4f}, "
                f"macro-AUC={auc_str}"
            )

    return (
        pd.DataFrame(rows),
        signature_sizes
    )


# ============================================================================
# EXPLORATORY CROSS-COHORT KRUSKAL-WALLIS
# ============================================================================

def cross_cohort_kruskal_wallis(
    cv_results_df
):
    """
    Exploratory only.

    Five fold estimates per cohort provide low statistical power.
    Therefore this is NOT treated as a formal independent-cohort
    comparison.
    """

    per_cohort = (
        cv_results_df[
            cv_results_df["cohort"]
            != "ALL (pooled test fold)"
        ]
    )

    groups = [
        g["accuracy"]
        .dropna()
        .values
        for _, g
        in per_cohort.groupby(
            "cohort"
        )
    ]

    groups = [
        g
        for g in groups
        if len(g) > 1
    ]

    if len(groups) < 2:

        return pd.DataFrame([{
            "Test":
                "Kruskal-Wallis "
                "(cohort accuracy)",

            "Statistic":
                np.nan,

            "p_value":
                np.nan,

            "Note":
                "Insufficient data"
        }])

    h, pv = (
        stats.kruskal(
            *groups
        )
    )

    return pd.DataFrame([{
        "Test":
            "Kruskal-Wallis "
            "(cohort accuracy across folds)",

        "Statistic":
            h,

        "p_value":
            pv,

        "Note":
            "EXPLORATORY ONLY — "
            "n=5 folds/cohort; low "
            "statistical power."
    }])


# ============================================================================
# PART C — TRIPOD-AI CHECKLIST
# ============================================================================

def build_tripod_checklist(
    signature_sizes,
    n_total
):

    mean_sig_size = (
        np.mean(
            signature_sizes
        )
        if signature_sizes
        else np.nan
    )

    checklist = [

        {
            "Item":
                "Source of data",

            "Reporting":
                (
                    "Three publicly available cohorts pooled: "
                    "METABRIC-BRCA-2016, TCGA-BRCA-2012, "
                    "TCGA-BRCA-2018 "
                    f"(n={n_total} total after ComBat batch correction)."
                )
        },

        {
            "Item":
                "Cohort splitting rule",

            "Reporting":
                (
                    f"Stratified {N_FOLDS}-fold cross-validation "
                    "on the pooled, ComBat-corrected cohort, "
                    "stratified by PAM50 subtype label "
                    f"(random_state={RANDOM_STATE}). "
                    "Cohort of origin was not used as the primary "
                    "stratification variable; per-cohort performance "
                    "was reported separately."
                )
        },

        {
            "Item":
                "Inter-cohort heterogeneity",

            "Reporting":
                (
                    "Baseline clinical/pathological characteristics "
                    "were assessed on the original per-cohort "
                    "clinical files before QC filtering, merging "
                    "and batch correction. Continuous variables were "
                    "compared using Kruskal-Wallis tests and "
                    "categorical variables using chi-square tests."
                )
        },

        {
            "Item":
                "Feature-selection leakage control",

            "Reporting":
                (
                    "The 50-gene signature was re-derived independently "
                    "within the training partition of each outer CV fold "
                    "using subtype-specific DEG filtering and 90% "
                    "cumulative-variance PCA. Mean signature size across "
                    f"folds was {mean_sig_size:.1f} genes. "
                    "No feature selection was performed on held-out data."
                )
        },

        {
            "Item":
                "Model",

            "Reporting":
                (
                    "XGBoost multiclass classifier with "
                    "n_estimators=300, max_depth=4 and "
                    "learning_rate=0.05."
                )
        },

        {
            "Item":
                "External validation design",

            "Reporting":
                (
                    "Independent external validation used "
                    "the GSE96058 cohort as described in the manuscript. "
                    "This script documents the design but does not execute "
                    "external validation."
                )
        },

        {
            "Item":
                "Performance variability",

            "Reporting":
                (
                    "Performance is summarized using mean, SD and "
                    "two-sided 95% Student-t confidence intervals "
                    "calculated from the five fold-level estimates."
                )
        }
    ]

    return pd.DataFrame(
        checklist
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Load expression dataset
    # -------------------------------------------------------------------------

    if not os.path.exists(
        COMBINED_EXPR_CSV
    ):

        raise FileNotFoundError(
            f"Could not find:\n{COMBINED_EXPR_CSV}"
        )

    df = pd.read_csv(
        COMBINED_EXPR_CSV
    )

    df = df.loc[
        :,
        ~df.columns.str.contains(
            "^Unnamed"
        )
    ]

    if "BATCH" not in df.columns:

        raise KeyError(
            "No 'BATCH' column found in input file."
        )

    df["COHORT"] = (
        df["BATCH"]
        .map(BATCH_LABEL_MAP)
        .fillna(
            df["BATCH"].astype(str)
        )
    )

    print(
        "\nModeling-cohort sizes "
        "(post-QC, post-ComBat):"
    )

    print(
        df["COHORT"]
        .value_counts()
    )

    print()

    # -------------------------------------------------------------------------
    # Convert subtype and stage labels
    # -------------------------------------------------------------------------

    if (
        df["SUBTYPE"]
        .dropna()
        .isin(
            SUBTYPE_MAP.keys()
        )
        .all()
    ):

        df["SUBTYPE_LABEL"] = (
            df["SUBTYPE"]
            .map(
                SUBTYPE_MAP
            )
        )

    else:

        df["SUBTYPE_LABEL"] = (
            df["SUBTYPE"]
        )

    if (
        df["TUMOR_STAGE"]
        .dropna()
        .isin(
            STAGE_MAP.keys()
        )
        .all()
    ):

        df["STAGE_LABEL"] = (
            df["TUMOR_STAGE"]
            .map(
                STAGE_MAP
            )
        )

    else:

        df["STAGE_LABEL"] = (
            df["TUMOR_STAGE"]
        )

    gene_cols = [
        c
        for c in df.columns
        if c not in (
            META_COLS
            + [
                "COHORT",
                "SUBTYPE_LABEL",
                "STAGE_LABEL"
            ]
        )
    ]

    print(
        f"Detected {len(gene_cols)} "
        "gene-expression columns."
    )

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    repro_df = (
        reproducibility_table()
    )

    repro_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Reproducibility_Info.csv"
        ),
        index=False
    )

    # -------------------------------------------------------------------------
    # MODULE 1 + PART A
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODULE 1 + PART A: "
        "Clinical validation & cohort comparability"
    )

    print(
        "=" * 70
    )

    validation_rows = []

    (
        df_cohort_a,
        grade_summary,
        orig_counts
    ) = (
        build_source_cohort_comparability_input(
            validation_rows
        )
    )

    validation_df = (
        pd.DataFrame(
            validation_rows
        )
    )

    validation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Validation_Report.csv"
        ),
        index=False
    )

    final_counts = (
        df["COHORT"]
        .value_counts()
        .to_dict()
    )

    counts_rows = []

    for cohort_name in orig_counts:

        counts_rows.append({
            "Cohort":
                cohort_name,

            "Original_n_pre_QC":
                orig_counts.get(
                    cohort_name,
                    np.nan
                ),

            "Final_modeling_n_post_QC":
                final_counts.get(
                    cohort_name,
                    np.nan
                )
        })

    counts_df = (
        pd.DataFrame(
            counts_rows
        )
    )

    counts_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ST_Cohort_counts.csv"
        ),
        index=False
    )

    print(
        counts_df.to_string(
            index=False
        )
    )

    cohort_summary_df = (
        run_cohort_comparability(
            df_cohort_a
        )
    )

    cohort_summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ST_Cohort_comparability.csv"
        ),
        index=False
    )

    print(
        "\n"
        + cohort_summary_df.to_string(
            index=False
        )
    )

    if grade_summary is not None:

        grade_summary.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "ST_Grade_METABRIC_only.csv"
            ),
            index=False
        )

        print(
            "\nGrade "
            "(METABRIC-2016 only, descriptive):"
        )

        print(
            grade_summary.to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # PART B
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PART B: "
        "Leakage-safe stratified 5-fold CV "
        "with per-cohort breakdown"
    )

    print(
        "=" * 70
    )

    (
        cv_results_df,
        signature_sizes
    ) = (
        run_stratified_cv_with_cohort_breakdown(
            df,
            gene_cols
        )
    )

    # -------------------------------------------------------------------------
    # Save full fold-level results
    # -------------------------------------------------------------------------

    cv_results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ST_CVbyCohort_full.csv"
        ),
        index=False
    )

    # -------------------------------------------------------------------------
    # Original mean ± SD summary
    # -------------------------------------------------------------------------

    cv_summary = (
        cv_results_df
        .groupby("cohort")[
            [
                "accuracy",
                "macro_f1",
                "mcc",
                "macro_auc"
            ]
        ]
        .agg(
            ["mean", "std"]
        )
    )

    cv_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ST_CVbyCohort_summary.csv"
        )
    )

    print(
        "\nMean ± SD:"
    )

    print(
        cv_summary
    )

    # -------------------------------------------------------------------------
    # NEW — mean + SD + 95% CI
    # -------------------------------------------------------------------------

    cv_ci_summary = (
        summarize_cv_with_ci(
            cv_results_df
        )
    )

    cv_ci_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ST_CVbyCohort_95CI_summary.csv"
        ),
        index=False
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Mean + SD + 95% CI"
    )

    print(
        "=" * 70
    )

    print(
        cv_ci_summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # NEW — manuscript-ready table
    # -------------------------------------------------------------------------

    manuscript_df = (
        create_manuscript_ready_table(
            cv_ci_summary
        )
    )

    manuscript_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ST_CVbyCohort_95CI_manuscript_table.csv"
        ),
        index=False
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MANUSCRIPT-READY TABLE"
    )

    print(
        "=" * 70
    )

    print(
        manuscript_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Exploratory cohort comparison
    # -------------------------------------------------------------------------

    kw_df = (
        cross_cohort_kruskal_wallis(
            cv_results_df
        )
    )

    kw_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ST_CVbyCohort_KruskalWallis.csv"
        ),
        index=False
    )

    print(
        "\n"
        + kw_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # PART C — TRIPOD-AI
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PART C: TRIPOD-AI checklist"
    )

    print(
        "=" * 70
    )

    checklist_df = (
        build_tripod_checklist(
            signature_sizes,
            len(df)
        )
    )

    checklist_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "TRIPOD_AI_checklist.csv"
        ),
        index=False
    )

    for _, row in (
        checklist_df.iterrows()
    ):

        print(
            f"\n[{row['Item']}]\n"
            f"{row['Reporting']}"
        )

    # -------------------------------------------------------------------------
    # Analysis summary
    # -------------------------------------------------------------------------

    overall_rows = (
        cv_results_df[
            cv_results_df["cohort"]
            == "ALL (pooled test fold)"
        ]
    )

    pooled_accuracy = (
        mean_sd_ci(
            overall_rows[
                "accuracy"
            ]
        )
    )

    per_cohort_acc = (
        cv_results_df[
            cv_results_df["cohort"]
            != "ALL (pooled test fold)"
        ]
        .groupby("cohort")[
            "accuracy"
        ]
        .mean()
    )

    if len(
        per_cohort_acc
    ):

        acc_range = (
            f"{per_cohort_acc.min():.3f}-"
            f"{per_cohort_acc.max():.3f}"
        )

    else:

        acc_range = "NA"

    kw_p = (
        kw_df.iloc[0]["p_value"]
        if "p_value"
        in kw_df.columns
        else np.nan
    )

    summary_lines = [

        "COMMENT 1 — UPDATED ANALYSIS SUMMARY",

        "=" * 45,

        (
            "- Three original source cohorts compared: "
            + ", ".join(
                f"{k}={v}"
                for k, v
                in orig_counts.items()
            )
        ),

        (
            f"- Final modeling dataset after QC/merge/ComBat: "
            f"n={len(df)}"
        ),

        (
            "- Baseline clinical/pathological characteristics "
            "were compared using chi-square tests for categorical "
            "variables and Kruskal-Wallis tests for continuous variables."
        ),

        (
            f"- Stratified {N_FOLDS}-fold CV used "
            "SUBTYPE stratification with "
            f"random_state={RANDOM_STATE}."
        ),

        (
            f"- DEG/PCA feature selection was repeated "
            f"within each training fold; mean signature size="
            f"{np.mean(signature_sizes):.1f} genes."
        ),

        (
            f"- Pooled test-fold accuracy: "
            f"{pooled_accuracy['mean']:.4f} "
            f"± {pooled_accuracy['sd']:.4f}; "
            f"95% CI="
            f"{pooled_accuracy['ci_lower']:.4f}-"
            f"{pooled_accuracy['ci_upper']:.4f}."
        ),

        (
            f"- Per-cohort mean accuracy range: "
            f"{acc_range}."
        ),

        (
            f"- Exploratory cross-cohort Kruskal-Wallis "
            f"on fold-level accuracy: p={kw_p:.4g}; "
            "interpret cautiously because only five fold-level "
            "estimates are available per cohort."
        ),

        (
            "- No statistical imputation was applied to undefined "
            "AUC values. AUC remains undefined when the corresponding "
            "held-out subset does not contain sufficient class variation."
        ),

        (
            "- 95% confidence intervals are two-sided Student-t "
            "intervals calculated from the valid fold-level estimates."
        )
    ]

    summary_text = (
        "\n".join(
            summary_lines
        )
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "Analysis_summary.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            summary_text
            + "\n"
        )

    print(
        "\n"
        + summary_text
    )

    print(
        "\nAll outputs written to:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()