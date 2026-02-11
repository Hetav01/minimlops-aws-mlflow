import os
import time
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]  # e.g. http://<ALB-DNS>
EXPERIMENT = "minimlops-smoke"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# Load adult income dataset from UCI (48,842 rows)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

print("Loading adult income dataset from UCI...")
df = pd.read_csv(url, names=columns, skipinitialspace=True)
print(f"Dataset loaded: {len(df)} rows")

# Simple preprocessing
df = df.replace("?", pd.NA).dropna()
X = df[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]]
y = (df["income"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with mlflow.start_run(run_name=f"adult-income-rf-{int(time.time())}"):
    # Log parameters
    n_estimators = 200
    max_depth = 15
    min_samples_split = 5
    min_samples_leaf = 2
    class_weight = "balanced"
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("dataset", "adult-income")
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("features", list(X.columns))
    mlflow.log_param("feature_scaling", "StandardScaler")
    mlflow.log_param("stratified_split", True)
    
    # Train model
    print("Training RandomForest model...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    import numpy as np
    proba = clf.predict_proba(X_test_scaled)
    y_pred_proba = np.asarray(proba)[:, 1]
    
    # Log comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    mlflow.log_metric("true_negatives", int(tn))
    mlflow.log_metric("false_positives", int(fp))
    mlflow.log_metric("false_negatives", int(fn))
    mlflow.log_metric("true_positives", int(tp))
    
    # Log model
    mlflow.sklearn.log_model(clf, "model")
    
    # Log feature importances as artifact
    feature_importance = "\n".join(
        f"{name}: {importance:.4f}"
        for name, importance in zip(X.columns, clf.feature_importances_)
    )
    mlflow.log_text(feature_importance, "feature_importances.txt")
    
    # Log classification report
    report = classification_report(y_test, y_pred)
    mlflow.log_text(str(report), "classification_report.txt")
    
    # Log confusion matrix
    cm_text = f"Confusion Matrix:\n{cm}\n\nTN: {tn}, FP: {fp}\nFN: {fn}, TP: {tp}"
    mlflow.log_text(cm_text, "confusion_matrix.txt")
    
    print(f"Model trained successfully!")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

print("Logged run to:", TRACKING_URI)