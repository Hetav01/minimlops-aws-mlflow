"""
Smoke test: same model type (LogisticRegression) trained with different
preprocessing — scaling or no scaling. Short training cycle (small max_iter).
"""
import os
import time
import numpy as np
import mlflow
from mlflow import sklearn as mlflow_sklearn
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Load .env so MLFLOW_TRACKING_URI can be set there
if find_dotenv(usecwd=True):
    load_dotenv(override=False)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not TRACKING_URI or not TRACKING_URI.strip():
    raise SystemExit(
        "MLFLOW_TRACKING_URI is required. Set it in .env or export it.\n"
        "Examples: http://localhost:5000  or  http://<your-alb-dns>"
    )
TRACKING_URI = TRACKING_URI.strip()

mlflow.set_tracking_uri(TRACKING_URI)

# (experiment_name, scaler_instance, param label for logging)
# None = no scaling (use raw features)
EXPERIMENTS = [
    ("adult-income-lr-zscore", StandardScaler(), "StandardScaler"),
    ("adult-income-lr-minmax", MinMaxScaler(), "MinMaxScaler"),
    ("adult-income-lr-robust", RobustScaler(), "RobustScaler"),
    ("adult-income-lr-unscaled", None, "none"),
]

# ── Load data ──────────────────────────────────────────────────────────────
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

print("Loading adult income dataset from UCI...")
df = pd.read_csv(url, names=columns, skipinitialspace=True)
print(f"Dataset loaded: {len(df)} rows")

df = df.replace("?", pd.NA).dropna()
FEATURES = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
X = df[FEATURES]
y = (df["income"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Same model config for every experiment (small, fast training)
MAX_ITER = 200
C = 1.0
SOLVER = "lbfgs"
CLASS_WEIGHT = "balanced"

base_run_name = f"adult-income-lr-{int(time.time())}"

for i, (experiment_name, scaler, scaler_label) in enumerate(EXPERIMENTS):
    mlflow.set_experiment(experiment_name)
    run_name = f"{base_run_name}-exp{i + 1}"

    # Preprocessing for this experiment
    if scaler is not None:
        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)
    else:
        X_train_use = np.asarray(X_train)
        X_test_use = np.asarray(X_test)

    # Train same model type, same hyperparams (short cycle)
    clf = LogisticRegression(
        max_iter=MAX_ITER,
        C=C,
        solver=SOLVER,
        class_weight=CLASS_WEIGHT,
        random_state=42,
    )
    t0 = time.perf_counter()
    clf.fit(X_train_use, y_train)
    training_time_sec = time.perf_counter() - t0

    # Inference timing (single batch)
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test_use)
    y_pred_proba = clf.predict_proba(X_test_use)[:, 1]
    inference_time_ms = (time.perf_counter() - t0) * 1000

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Coefficients as feature importance (absolute value)
    coef = np.abs(clf.coef_[0])
    feature_importance = "\n".join(
        f"{name}: {imp:.4f}" for name, imp in zip(FEATURES, coef)
    )
    report = classification_report(y_test, y_pred)
    cm_text = f"Confusion Matrix:\n{cm}\n\nTN: {tn}, FP: {fp}\nFN: {fn}, TP: {tp}"

    # System / runtime metrics
    n_iter = int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") and clf.n_iter_ is not None else 0
    coef_nnz = int(np.count_nonzero(clf.coef_))
    coef_l1 = float(np.sum(np.abs(clf.coef_)))

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("max_iter", MAX_ITER)
        mlflow.log_param("C", C)
        mlflow.log_param("solver", SOLVER)
        mlflow.log_param("class_weight", CLASS_WEIGHT)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("dataset", "adult-income")
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("scaling", scaler_label)
        mlflow.log_param("stratified_split", True)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("true_negatives", int(tn))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("true_positives", int(tp))
        # System / runtime metrics
        mlflow.log_metric("training_time_sec", round(training_time_sec, 4))
        mlflow.log_metric("inference_time_ms", round(inference_time_ms, 4))
        mlflow.log_metric("n_iter", n_iter)
        mlflow.log_metric("train_samples", len(y_train))
        mlflow.log_metric("test_samples", len(y_test))
        mlflow.log_metric("model_coef_nnz", coef_nnz)
        mlflow.log_metric("model_coef_l1", round(coef_l1, 6))

        mlflow_sklearn.log_model(clf, "model")
        mlflow.log_text(feature_importance, "feature_importances.txt")
        mlflow.log_text(str(report), "classification_report.txt")
        mlflow.log_text(cm_text, "confusion_matrix.txt")

    print(f"  {experiment_name}: scaling={scaler_label} → accuracy={accuracy:.4f}, f1={f1:.4f}")

print(f"\nDone. LogisticRegression (max_iter={MAX_ITER}), different scaling → {len(EXPERIMENTS)} experiments.")
print("Tracking URI:", TRACKING_URI)
