import os
import json
import random
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras import regularizers

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
RANDOM_STATE = 42
N_CV_FOLDS = 5
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 32 
MAX_EVALS = 50

BASE_DIR = Path(r"D:\MultiModal_Classification")
INPUT_DIR = BASE_DIR / "output" / "02_batch_correction_deg_pca"
FILE_NAME = "SUBTYPE_PCA_DEG_expression_matrix"
LABEL_COL = "SUBTYPE"

NON_FEATURE_COLS = ["PATIENT_ID", "AGE", "SEX", "OS_STATUS", "OS_MONTHS", "TUMOR_STAGE"]

RUN_HYPEROPT = True

EXISTING_BEST_PARAMS = {
    "units1": 64,
    "units2": 64,
    "dropout": 0.5782881515748334,
    "activation": "tanh",
    "lr": 0.00340178999601731,
    "l2_reg": 0.0006133856930939435,
}

OUTPUT_DIR = BASE_DIR / "output" / "03_classification" / "dnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# REPRODUCIBILITY
# ================================================================
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

SUBTYPE_MAPPING = {
    "0": "Normal",
    "1": "LumA",
    "2": "LumB",
    "3": "Her2",
    "4": "Basal",
}

# ================================================================
# DATA LOADING
# ================================================================
file_path = INPUT_DIR / f"{FILE_NAME}.csv"
if not file_path.exists():
    raise FileNotFoundError(file_path)

data = pd.read_csv(file_path)
data = data.dropna(subset=[LABEL_COL]).reset_index(drop=True)

y_numeric = data[LABEL_COL].astype(str).values
y_bio = np.array([SUBTYPE_MAPPING.get(v, f"Unknown_{v}") for v in y_numeric])

X_df = data.drop(columns=[LABEL_COL] + [c for c in NON_FEATURE_COLS if c in data.columns])
X = X_df.values.astype(np.float32)

classes_bio = sorted(np.unique(y_bio))
label_encoder = {c: i for i, c in enumerate(classes_bio)}
reverse_encoder = {i: c for c, i in label_encoder.items()}
y = np.array([label_encoder[v] for v in y_bio], dtype=np.int64)
NUM_CLASSES = len(classes_bio)

print("=" * 80)
print("UPDATED DNN TRAINING + 5-FOLD CV + 95% CI")
print("=" * 80)
print(f"Dataset: {data.shape}")
print(f"Feature matrix: {X.shape}")
print(f"Classes: {reverse_encoder}")
print("Class counts:")
print(pd.Series(y).value_counts().sort_index())

# ================================================================
# MODEL
# ================================================================
def build_dnn(input_dim, num_classes, params):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(
            params["units1"],
            activation=params["activation"],
            kernel_regularizer=regularizers.l2(params["l2_reg"]),
        ),
        BatchNormalization(),
        Dropout(params["dropout"]),
        Dense(
            params["units2"],
            activation=params["activation"],
            kernel_regularizer=regularizers.l2(params["l2_reg"]),
        ),
        BatchNormalization(),
        Dropout(params["dropout"]),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_dnn(model, X_train, y_train, X_val, y_val, class_weights, log_path=None):
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]
    if log_path:
        callbacks.append(CSVLogger(log_path, separator=",", append=False))

    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight={int(k): float(v) for k, v in class_weights.items()},
        verbose=0,
        callbacks=callbacks,
    )


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def calc_metrics(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_AUC": roc_auc_score(
            y_bin, y_prob, average="macro", multi_class="ovr"
        ),
    }


def ci_t(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    margin = float(tcrit * se)
    return mean, sd, mean - margin, mean + margin

# ================================================================
# OPTIONAL HYPEROPT — OFF BY DEFAULT FOR EXACT REPRODUCTION
# ================================================================
def run_hyperopt(X_dev, y_dev):
    space = {
        "units1": hp.choice("units1", [64, 128, 256]),
        "units2": hp.choice("units2", [32, 64, 128]),
        "dropout": hp.uniform("dropout", 0.3, 0.6),
        "activation": hp.choice("activation", ["relu", "elu", "tanh"]),
        "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
        "l2_reg": hp.loguniform("l2_reg", np.log(1e-5), np.log(1e-3)),
    }

    def objective(params):
        skf = StratifiedKFold(
            n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        scores = []
        for tr, va in skf.split(X_dev, y_dev):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_dev[tr])
            Xva = scaler.transform(X_dev[va])
            weights = get_class_weights(y_dev[tr])
            tf.keras.backend.clear_session()
            fold_seed = RANDOM_STATE
            np.random.seed(fold_seed)
            random.seed(fold_seed)
            tf.random.set_seed(fold_seed)
            model = build_dnn(Xtr.shape[1], NUM_CLASSES, params)
            train_dnn(model, Xtr, y_dev[tr], Xva, y_dev[va], weights)
            prob = model.predict(Xva, verbose=0)
            pred = np.argmax(prob, axis=1)
            scores.append(balanced_accuracy_score(y_dev[va], pred))
            del model
            tf.keras.backend.clear_session()
        return {"loss": -float(np.mean(scores)), "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE),
    )
    units1 = [64, 128, 256][int(best["units1"])]
    units2 = [32, 64, 128][int(best["units2"])]
    activation = ["relu", "elu", "tanh"][int(best["activation"])]
    return {
        "units1": units1,
        "units2": units2,
        "dropout": float(best["dropout"]),
        "activation": activation,
        "lr": float(best["lr"]),
        "l2_reg": float(best["l2_reg"]),
    }, trials

# ================================================================
# RECREATE ORIGINAL 80:20 DEVELOPMENT/FINAL-TEST SPLIT
# ================================================================
indices = np.arange(len(X))
(
    X_dev,
    X_final_test,
    y_dev,
    y_final_test,
    idx_dev,
    idx_final_test,
) = train_test_split(
    X,
    y,
    indices,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Development N: {len(X_dev)}")
print(f"Final test N: {len(X_final_test)}")

# ================================================================
# HYPERPARAMETERS
# ================================================================
if RUN_HYPEROPT:
    best_params, trials = run_hyperopt(X_dev, y_dev)
    with open(OUTPUT_DIR / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    pd.DataFrame(trials.trials).to_json(
        OUTPUT_DIR / "hyperopt_trials.json", orient="records", indent=2
    )
else:
    best_params = dict(EXISTING_BEST_PARAMS)
    trials = None

# The original train_model() hard-coded batch_size=32. Therefore 32 is
# deliberately retained here so the updated final model remains aligned with
# the model that produced the manuscript's original test result.
with open(OUTPUT_DIR / "used_hyperparameters.json", "w", encoding="utf-8") as f:
    json.dump(
        {**best_params, "batch_size": BATCH_SIZE, "random_state": RANDOM_STATE},
        f,
        indent=2,
    )

print("\nDNN parameters used:")
print(best_params)
print("Batch size used:", BATCH_SIZE)

# ================================================================
# 5-FOLD STRATIFIED CV ON DEVELOPMENT DATA
# ================================================================
print("\n" + "=" * 80)
print("5-FOLD STRATIFIED CV — DNN")
print("=" * 80)

skf = StratifiedKFold(
    n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
)
fold_rows = []
oof_true = []
oof_pred = []
oof_prob = []

for fold, (tr, va) in enumerate(skf.split(X_dev, y_dev), start=1):
    print(f"\nFold {fold}/{N_CV_FOLDS}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_dev[tr])
    Xva = scaler.transform(X_dev[va])
    weights = get_class_weights(y_dev[tr])

    tf.keras.backend.clear_session()
    fold_seed = RANDOM_STATE + fold
    np.random.seed(fold_seed)
    random.seed(fold_seed)
    tf.random.set_seed(fold_seed)

    model = build_dnn(Xtr.shape[1], NUM_CLASSES, best_params)
    history = train_dnn(model, Xtr, y_dev[tr], Xva, y_dev[va], weights)

    prob = model.predict(Xva, verbose=0)
    pred = np.argmax(prob, axis=1)
    metrics = calc_metrics(y_dev[va], pred, prob)

    fold_rows.append({
        "Fold": fold,
        "Train_N": len(tr),
        "Validation_N": len(va),
        **metrics,
        "Epochs_Completed": len(history.history["loss"]),
    })

    oof_true.extend(y_dev[va])
    oof_pred.extend(pred)
    oof_prob.append(prob)

    print(
        f"AUC={metrics['Macro_AUC']:.6f}, "
        f"F1={metrics['Macro_F1']:.6f}, "
        f"MCC={metrics['MCC']:.6f}, "
        f"Accuracy={metrics['Accuracy']:.6f}"
    )

    del model
    tf.keras.backend.clear_session()

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUTPUT_DIR / "DNN_CV_Fold_Level_Metrics.csv", index=False)

# ================================================================
# CV SUMMARY + 95% CI
# ================================================================
summary_rows = []
for metric in ["Macro_AUC", "Macro_F1", "MCC", "Accuracy", "Balanced_Accuracy"]:
    mean, sd, lower, upper = ci_t(fold_df[metric].values)
    summary_rows.append({
        "Metric": metric,
        "Mean": mean,
        "SD": sd,
        "95_CI_Lower": lower,
        "95_CI_Upper": upper,
        "N_Folds": N_CV_FOLDS,
        "Mean_95CI": f"{mean:.4f} ({lower:.4f}–{upper:.4f})",
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "DNN_CV_95CI_Summary.csv", index=False)

reviewer_df = summary_df[
    summary_df["Metric"].isin(["Macro_AUC", "Macro_F1", "MCC"])
].copy()
reviewer_df.to_csv(
    OUTPUT_DIR / "DNN_Reviewer2_CV_AUC_F1_MCC_95CI.csv", index=False
)

# ================================================================
# OOF OUTPUTS
# ================================================================
oof_prob = np.vstack(oof_prob)
oof_true = np.asarray(oof_true)
oof_pred = np.asarray(oof_pred)

pd.DataFrame({"y_true": oof_true, "y_pred": oof_pred}).to_csv(
    OUTPUT_DIR / "DNN_CV_OOF_Predictions.csv", index=False
)
np.save(OUTPUT_DIR / "DNN_CV_OOF_Probabilities.npy", oof_prob)

# ================================================================
# OOF CONFUSION MATRIX
# ================================================================
cm = confusion_matrix(oof_true, oof_pred, labels=np.arange(NUM_CLASSES))
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=classes_bio, yticklabels=classes_bio
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DNN — 5-fold out-of-fold confusion matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DNN_CV_OOF_Confusion_Matrix.png", dpi=300)
plt.close()

# ================================================================
# OOF ROC CURVE
# ================================================================
y_bin_oof = label_binarize(oof_true, classes=np.arange(NUM_CLASSES))
plt.figure(figsize=(8, 7))
for i, name in enumerate(classes_bio):
    fpr, tpr, _ = roc_curve(y_bin_oof[:, i], oof_prob[:, i])
    class_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={class_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DNN — 5-fold out-of-fold ROC curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DNN_CV_OOF_ROC.png", dpi=300)
plt.close()

# ================================================================
# FINAL HELD-OUT TEST MODEL
# ================================================================
train_idx_local, val_idx_local = train_test_split(
    np.arange(len(X_dev)),
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_STATE,
)

X_train = X_dev[train_idx_local]
X_val = X_dev[val_idx_local]
y_train = y_dev[train_idx_local]
y_val = y_dev[val_idx_local]

final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train)
X_val_scaled = final_scaler.transform(X_val)
X_test_scaled = final_scaler.transform(X_final_test)

final_weights = get_class_weights(y_train)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
tf.keras.backend.clear_session()

final_model = build_dnn(
    X_train_scaled.shape[1], NUM_CLASSES, best_params
)
final_history = train_dnn(
    final_model,
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    final_weights,
    str(OUTPUT_DIR / "final_optimized_training_logs.csv"),
)

# Final test predictions
final_test_prob = final_model.predict(X_test_scaled, verbose=0)
final_test_pred = np.argmax(final_test_prob, axis=1)
final_test_metrics = calc_metrics(
    y_final_test,
    final_test_pred,
    final_test_prob
)

# Train/validation metrics for documentation
train_prob = final_model.predict(X_train_scaled, verbose=0)
train_pred = np.argmax(train_prob, axis=1)
val_prob = final_model.predict(X_val_scaled, verbose=0)
val_pred = np.argmax(val_prob, axis=1)

train_metrics = calc_metrics(y_train, train_pred, train_prob)
val_metrics = calc_metrics(y_val, val_pred, val_prob)

pd.DataFrame([
    {"Dataset": "Train", **train_metrics},
    {"Dataset": "Validation", **val_metrics},
    {"Dataset": "Test", **final_test_metrics},
]).to_csv(OUTPUT_DIR / "final_performance_metrics.csv", index=False)

# Save model/scaler/encoders
final_model.save(OUTPUT_DIR / "best_model.keras")
with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(final_scaler, f)
with open(OUTPUT_DIR / "label_encoder.json", "w", encoding="utf-8") as f:
    json.dump(label_encoder, f, indent=2)
with open(OUTPUT_DIR / "reverse_encoder.json", "w", encoding="utf-8") as f:
    json.dump(reverse_encoder, f, indent=2)

# Classification report and test confusion matrix
report = classification_report(
    y_final_test, final_test_pred,
    target_names=classes_bio,
    zero_division=0
)
(OUTPUT_DIR / "final_test_classification_report.txt").write_text(
    report, encoding="utf-8"
)

cm_test = confusion_matrix(
    y_final_test, final_test_pred, labels=np.arange(NUM_CLASSES)
)
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm_test, annot=True, fmt="d", cmap="Blues",
    xticklabels=classes_bio, yticklabels=classes_bio
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DNN — final held-out test confusion matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_test_confusion_matrix.png", dpi=300)
plt.close()

# ================================================================
# METADATA
# ================================================================
metadata = {
    "input_file": str(file_path),
    "random_state": RANDOM_STATE,
    "n_cv_folds": N_CV_FOLDS,
    "development_samples": int(len(X_dev)),
    "final_test_samples": int(len(X_final_test)),
    "final_test_used_in_cv": False,
    "hyperopt_rerun": RUN_HYPEROPT,
    "batch_size_used": BATCH_SIZE,
    "epochs_max": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "hyperparameters": best_params,
    "ci_method": "Two-sided Student-t CI from five fold-level estimates",
    "ci_formula": "mean +/- t(0.975, df=4) * SD/sqrt(5)",
    "cv_metrics": ["Macro_AUC", "Macro_F1", "MCC", "Accuracy", "Balanced_Accuracy"],
    "original_final_test_metrics": {
        k: float(v) for k, v in final_test_metrics.items()
    },
}
with open(OUTPUT_DIR / "DNN_analysis_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

split_df = pd.DataFrame({
    "Original_Row_Index": np.concatenate([idx_dev, idx_final_test]),
    "Split": ["Development"] * len(idx_dev) + ["Final_Test"] * len(idx_final_test),
})
split_df.to_csv(
    OUTPUT_DIR / "DNN_Development_FinalTest_Split_Indices.csv", index=False
)

# ================================================================
# FINAL OUTPUT
# ================================================================
print("\n" + "=" * 80)
print("DNN ANALYSIS COMPLETED")
print("=" * 80)
print("5-fold CV summary:")
print(reviewer_df[["Metric", "Mean_95CI"]].to_string(index=False))
print("\nFinal held-out test performance:")
print({k: round(float(v), 6) for k, v in final_test_metrics.items()})
print("\nOutput directory:")
print(OUTPUT_DIR)
