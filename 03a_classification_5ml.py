import os
import json
import random
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
    classification_report, roc_auc_score
)
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
RANDOM_STATE = 42
N_CV_FOLDS = 5
N_OPTUNA_FOLDS = 3
N_OPTUNA_TRIALS = 30

BASE_DIR = Path(r"D:\MultiModal_Classification")
DATA_FILE_PATH = BASE_DIR / "output" / "02_batch_correction_deg_pca" / "SUBTYPE_PCA_DEG_expression_matrix.csv"

INPUT_DIR = BASE_DIR / "output" / "03_classification" / "ml5"
FILE_PATH = DATA_FILE_PATH

MODELS_TO_RUN = ["RF", "KNN", "SVM", "LogisticRegression", "XGB"]

RUN_OPTUNA = True

# ================================================================
# REPRODUCIBILITY
# ================================================================
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

for model_name in MODELS_TO_RUN:
    (INPUT_DIR / model_name).mkdir(parents=True, exist_ok=True)

# ================================================================
# DATA
# ================================================================
df = pd.read_csv(FILE_PATH)
DROP_COLUMNS = [
    "SUBTYPE", "PATIENT_ID", "AGE", "SEX", "OS_STATUS", "OS_MONTHS", "TUMOR_STAGE"
]
missing = [c for c in DROP_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

X = df.drop(columns=DROP_COLUMNS)
y = df["SUBTYPE"]

print(f"Dataset shape: {df.shape}")
print(f"Feature matrix: {X.shape}")
print("Class distribution:")
print(y.value_counts().sort_index())

# ================================================================
# HELPERS
# ================================================================
def make_model(model_name, params):
    if model_name == "RF":
        return RandomForestClassifier(**params)
    if model_name == "KNN":
        return KNeighborsClassifier(**params)
    if model_name == "SVM":
        # probability=True is required for multiclass ROC-AUC
        params = dict(params)
        params["probability"] = True
        return SVC(**params)
    if model_name == "LogisticRegression":
        return LogisticRegression(**params)
    if model_name == "XGB":
        return XGBClassifier(**params)
    raise ValueError(model_name)


def calc_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    margin = float(tcrit * sem)
    return mean, sd, mean - margin, mean + margin


def fold_metrics(y_true, y_pred, y_prob, classes):
    y_bin = label_binarize(y_true, classes=classes)
    auc_macro = roc_auc_score(
        y_bin, y_prob, average="macro", multi_class="ovr"
    )
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Macro_AUC": auc_macro,
        "Weighted_Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted_Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def evaluate_fixed_model_cv(model_name, params, X, y, output_dir):
    """Five-fold stratified evaluation using fixed optimized hyperparameters."""
    skf = StratifiedKFold(
        n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    rows = []
    oof_y = []
    oof_prob = []
    oof_pred = []
    classes = np.sort(y.unique())

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.20, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(sss.split(X_train_full, y_train_full))
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = make_model(model_name, params)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = fold_metrics(y_test, y_test_pred, y_prob, classes)

        rows.append({
            "Model": model_name,
            "Fold": fold,
            "Train_N": len(X_train),
            "Validation_N": len(X_val),
            "Test_N": len(X_test),
            **metrics,
        })
        oof_y.extend(y_test.to_numpy())
        oof_pred.extend(y_test_pred)
        oof_prob.append(y_prob)

        print(
            f"{model_name} fold {fold}: "
            f"AUC={metrics['Macro_AUC']:.6f}, "
            f"F1={metrics['Macro_F1']:.6f}, "
            f"MCC={metrics['MCC']:.6f}, "
            f"Acc={metrics['Accuracy']:.6f}"
        )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(output_dir / "optimized_cv_fold_metrics.csv", index=False)

    summary_rows = []
    for metric in ["Macro_AUC", "Macro_F1", "MCC", "Accuracy"]:
        mean, sd, lower, upper = calc_ci(fold_df[metric].values)
        summary_rows.append({
            "Model": model_name,
            "Metric": metric,
            "Mean": mean,
            "SD": sd,
            "95_CI_Lower": lower,
            "95_CI_Upper": upper,
            "N_Folds": N_CV_FOLDS,
            "Mean_95CI": f"{mean:.4f} ({lower:.4f}–{upper:.4f})",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "optimized_cv_summary_95CI.csv", index=False)

    oof_prob = np.vstack(oof_prob)
    oof_y = np.asarray(oof_y)
    oof_pred = np.asarray(oof_pred)

    pd.DataFrame({"y_true": oof_y, "y_pred": oof_pred}).to_csv(
        output_dir / "optimized_oof_predictions.csv", index=False
    )
    np.save(output_dir / "optimized_oof_probabilities.npy", oof_prob)

    # Pooled out-of-fold classification report and confusion matrix
    report = classification_report(oof_y, oof_pred, zero_division=0)
    (output_dir / "optimized_oof_classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    cm = confusion_matrix(oof_y, oof_pred, labels=classes)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - 5-fold out-of-fold confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "optimized_oof_confusion_matrix.png", dpi=300)
    plt.close()

    # OOF ROC curve
    y_bin = label_binarize(oof_y, classes=classes)
    plt.figure(figsize=(8, 7))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], oof_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {cls} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - 5-fold out-of-fold ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "optimized_oof_roc_curve.png", dpi=300)
    plt.close()

    return fold_df, summary_df


def optimize_hyperparameters(model_name, X, y, n_trials=N_OPTUNA_TRIALS):
    """Original optimization structure, with a multiclass-safe LR solver."""
    def objective(trial):
        if model_name == "RF":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            }
        elif model_name == "KNN":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
            }
        elif model_name == "SVM":
            params = {
                "C": trial.suggest_float("C", 1e-2, 100, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "class_weight": "balanced",
                "probability": True,
                "random_state": RANDOM_STATE,
            }
        elif model_name == "LogisticRegression":
            # lbfgs is used deliberately because this is a 5-class problem.
            params = {
                "C": trial.suggest_float("C", 1e-2, 100, log=True),
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            }
        elif model_name == "XGB":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2.0),
                "tree_method": "hist",
                "eval_metric": "mlogloss",
                "random_state": RANDOM_STATE,
            }
        else:
            raise ValueError(model_name)

        model = make_model(model_name, params)
        skf = StratifiedKFold(
            n_splits=N_OPTUNA_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        scores = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            score = accuracy_score(y.iloc[te], pred)
            scores.append(score)
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_optimization",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=HyperbandPruner(),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study


def load_or_optimize(model_name):
    output_dir = INPUT_DIR / model_name
    params_file = output_dir / "best_hyperparameters.json"
    if RUN_OPTUNA:
        study = optimize_hyperparameters(model_name, X, y)
        params = study.best_params
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)
        study.trials_dataframe().to_csv(
            output_dir / "optuna_study_log.csv", index=False
        )
        return params, study.best_value
    if not params_file.exists():
        raise FileNotFoundError(
            f"RUN_OPTUNA=False but no saved parameters were found: {params_file}"
        )
    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params, None


def write_experiment_summary(results):
    summary = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "optuna": optuna.__version__,
        "random_state": RANDOM_STATE,
        "n_cv_folds": N_CV_FOLDS,
        "n_optuna_folds": N_OPTUNA_FOLDS,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "models": MODELS_TO_RUN,
        "ci_method": "Two-sided Student-t CI from five fold-level estimates",
        "ci_formula": "mean +/- t(0.975, df=4) * SD/sqrt(5)",
        "results": results,
    }
    with open(INPUT_DIR / "updated_experiment_summary_95CI.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    all_fold = []
    all_summary = []
    compact = []

    for model_name in MODELS_TO_RUN:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        params, optuna_score = load_or_optimize(model_name)
        print("Optimized parameters:", params)
        if optuna_score is not None:
            print(f"Optuna mean 3-fold accuracy: {optuna_score:.6f}")

        # Evaluate using the same optimized model configuration across 5 folds.
        fold_df, summary_df = evaluate_fixed_model_cv(
            model_name, params, X, y, INPUT_DIR / model_name
        )

        all_fold.append(fold_df)
        all_summary.append(summary_df)

        pivot = summary_df.pivot(index="Model", columns="Metric", values="Mean_95CI").reset_index()
        compact.append(pivot)

        # Final full-data fit for saved/deployment model.
        final_model = make_model(model_name, params)
        final_model.fit(X, y)
        joblib.dump(final_model, INPUT_DIR / model_name / "best_model_updated_95CI.joblib")

    all_fold_df = pd.concat(all_fold, ignore_index=True)
    all_summary_df = pd.concat(all_summary, ignore_index=True)

    all_fold_df.to_csv(INPUT_DIR / "all_models_optimized_cv_fold_metrics.csv", index=False)
    all_summary_df.to_csv(INPUT_DIR / "all_models_optimized_cv_summary_95CI.csv", index=False)

    # Reviewer-facing compact table.
    compact_rows = []
    for model_name in MODELS_TO_RUN:
        tmp = all_summary_df[all_summary_df["Model"] == model_name].set_index("Metric")
        compact_rows.append({
            "Model": model_name,
            "Macro_AUC_(95%CI)": tmp.loc["Macro_AUC", "Mean_95CI"],
            "Macro_F1_(95%CI)": tmp.loc["Macro_F1", "Mean_95CI"],
            "MCC_(95%CI)": tmp.loc["MCC", "Mean_95CI"],
            "Accuracy_(95%CI)": tmp.loc["Accuracy", "Mean_95CI"],
        })
    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(INPUT_DIR / "all_models_CV_95CI_manuscript_table.csv", index=False)

    write_experiment_summary(
        compact_df.to_dict(orient="records")
    )

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print("Fold-level metrics:", INPUT_DIR / "all_models_optimized_cv_fold_metrics.csv")
    print("95% CI summary:", INPUT_DIR / "all_models_optimized_cv_summary_95CI.csv")
    print("Manuscript table:", INPUT_DIR / "all_models_CV_95CI_manuscript_table.csv")
