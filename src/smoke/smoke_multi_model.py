"""
Smoke test: train 4 different models in a single MLflow experiment.
Same data, same metrics; compare model type in one place.
"""
import os
import time
import numpy as np
import mlflow
from mlflow import sklearn as mlflow_sklearn
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

if find_dotenv(usecwd=True):
    load_dotenv(override=False)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not TRACKING_URI or not TRACKING_URI.strip():
    raise SystemExit(
        "MLFLOW_TRACKING_URI is required. Set it in .env or export it."
    )
TRACKING_URI = TRACKING_URI.strip()

EXPERIMENT_NAME = "adult-income-multi-model"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Data ───────────────────────────────────────────────────────────────────
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
print("Loading adult income dataset...")
df = pd.read_csv(url, names=columns, skipinitialspace=True)
df = df.replace("?", pd.NA).dropna()

FEATURES = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
X = df[FEATURES]
y = (df["income"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Models: (run_name_suffix, estimator, extra_params_for_mlflow) ────────────
MODELS = [
    (
        "logistic",
        LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", random_state=42),
        {"model": "LogisticRegression", "max_iter": 200, "C": 1.0},
    ),
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=50, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        {"model": "RandomForestClassifier", "n_estimators": 50, "max_depth": 10},
    ),
    (
        "sgd",
        SGDClassifier(
            max_iter=200, tol=1e-3, class_weight="balanced", random_state=42
        ),
        {"model": "SGDClassifier", "max_iter": 200},
    ),
    (
        "naive_bayes",
        GaussianNB(),
        {"model": "GaussianNB"},
    ),
]

base_run_name = f"adult-income-{int(time.time())}"

for suffix, clf, extra_params in MODELS:
    model_name = extra_params["model"]
    run_name = f"{model_name}_{base_run_name}"
    print(f"Training {model_name}...")

    t0 = time.perf_counter()
    clf.fit(X_train_s, y_train)
    training_time_sec = time.perf_counter() - t0

    if hasattr(clf, "predict_proba") and callable(getattr(clf, "predict_proba", None)):
        y_pred_proba = clf.predict_proba(X_test_s)[:, 1]
    else:
        y_pred_proba = clf.decision_function(X_test_s)

    y_pred = clf.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_test, y_pred)
    cm_text = f"Confusion Matrix:\n{cm}\n\nTN: {tn}, FP: {fp}\nFN: {fn}, TP: {tp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset", "adult-income")
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("scaling", "StandardScaler")
        for k, v in extra_params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("true_negatives", int(tn))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("true_positives", int(tp))
        mlflow.log_metric("training_time_sec", round(training_time_sec, 4))
        mlflow.log_metric("train_samples", len(y_train))
        mlflow.log_metric("test_samples", len(y_test))

        mlflow_sklearn.log_model(clf, "model")
        mlflow.log_text(str(report), "classification_report.txt")
        mlflow.log_text(cm_text, "confusion_matrix.txt")

    print(f"  {model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}, time={training_time_sec:.2f}s")

print(f"\nDone. 4 models logged to experiment '{EXPERIMENT_NAME}'.")
print("Tracking URI:", TRACKING_URI)
