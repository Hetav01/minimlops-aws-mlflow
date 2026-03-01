"""
Smoke benchmark: train multiple models with baseline vs tuned variants in one
MLflow experiment, then print side-by-side metric deltas.

Upgrades in this workflow:
- Uses a local sklearn dataset (no network dependency)
- Lightweight hyperparameter search per model (RandomizedSearchCV)
- Validation-based threshold tuning to surface precision/recall tradeoffs
- Two internet-popular optional models: XGBoost and LightGBM
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd

# Type alias for array-like inputs (DataFrame, Series, or ndarray from train_test_split)
ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]
import mlflow
from mlflow import sklearn as mlflow_sklearn
from dotenv import find_dotenv, load_dotenv

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

if find_dotenv(usecwd=True):
    load_dotenv(override=False)

RANDOM_STATE = 42
EXPERIMENT_NAME = "digits-binary-multi-model"
MAX_TUNE_SAMPLES = 15000

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
if not TRACKING_URI:
    fallback_uri = f"file://{os.path.abspath('mlruns')}"
    print("MLFLOW_TRACKING_URI is not set. Falling back to local tracking:", fallback_uri)
    TRACKING_URI = fallback_uri

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


@dataclass
class ModelSpec:
    key: str
    display_name: str
    estimator: Any
    preprocessor_kind: str
    param_distributions: dict[str, list[Any]]
    n_iter: int


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def build_preprocessors(num_cols: list[str], cat_cols: list[str]) -> dict[str, ColumnTransformer]:
    linear_transformers = []
    tree_transformers = []

    if num_cols:
        linear_transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
        tree_transformers.append(("num", SimpleImputer(strategy="median"), num_cols))

    if cat_cols:
        linear_transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                cat_cols,
            )
        )
        tree_transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )

    linear_preprocessor = ColumnTransformer(transformers=linear_transformers)
    tree_preprocessor = ColumnTransformer(transformers=tree_transformers)

    return {
        "linear": linear_preprocessor,
        "tree": tree_preprocessor,
    }


def build_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(
            key="logistic_regression",
            display_name="Logistic Regression",
            estimator=LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="saga",
                random_state=RANDOM_STATE,
            ),
            preprocessor_kind="linear",
            param_distributions={
                "clf__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "clf__fit_intercept": [True, False],
            },
            n_iter=6,
        ),
        ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            preprocessor_kind="tree",
            param_distributions={
                "clf__n_estimators": [150, 250, 400],
                "clf__max_depth": [8, 14, None],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__max_features": ["sqrt", 0.5],
            },
            n_iter=6,
        ),
        ModelSpec(
            key="extra_trees",
            display_name="Extra Trees",
            estimator=ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            preprocessor_kind="tree",
            param_distributions={
                "clf__n_estimators": [150, 250, 400],
                "clf__max_depth": [8, 14, None],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__max_features": ["sqrt", 0.5],
            },
            n_iter=6,
        ),
        ModelSpec(
            key="sgd_log_loss",
            display_name="SGD (Log Loss)",
            estimator=SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=1200,
                tol=1e-3,
            ),
            preprocessor_kind="linear",
            param_distributions={
                "clf__alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                "clf__penalty": ["l2", "l1", "elasticnet"],
            },
            n_iter=6,
        ),
    ]

    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                key="xgboost",
                display_name="XGBoost",
                estimator=XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                preprocessor_kind="tree",
                param_distributions={
                    "clf__n_estimators": [180, 300, 450],
                    "clf__max_depth": [3, 5, 7],
                    "clf__learning_rate": [0.03, 0.06, 0.1],
                    "clf__subsample": [0.7, 0.9, 1.0],
                    "clf__colsample_bytree": [0.6, 0.8, 1.0],
                },
                n_iter=6,
            )
        )

    if LGBMClassifier is not None:
        specs.append(
            ModelSpec(
                key="lightgbm",
                display_name="LightGBM",
                estimator=LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                preprocessor_kind="tree",
                param_distributions={
                    "clf__n_estimators": [180, 300, 450],
                    "clf__num_leaves": [31, 63, 127],
                    "clf__learning_rate": [0.03, 0.06, 0.1],
                    "clf__subsample": [0.7, 0.9, 1.0],
                    "clf__colsample_bytree": [0.6, 0.8, 1.0],
                },
                n_iter=6,
            )
        )

    return specs


def build_pipeline(spec: ModelSpec, preprocessors: dict[str, ColumnTransformer]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", clone(preprocessors[spec.preprocessor_kind])),
            ("clf", clone(spec.estimator)),
        ]
    )


def tune_subset(X: Any, y: Any) -> tuple[pd.DataFrame, pd.Series]:
    if len(y) <= MAX_TUNE_SAMPLES:
        return pd.DataFrame(X), pd.Series(y)
    X_sub, _x_rest, y_sub, _y_rest = train_test_split(
        X,
        y,
        train_size=MAX_TUNE_SAMPLES,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame(X_sub), pd.Series(y_sub)


def predict_scores(estimator: Any, X: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError("Estimator does not expose predict_proba or decision_function")


def evaluate_metrics(y_true: Any, y_score: np.ndarray, threshold: float) -> tuple[dict[str, float], np.ndarray]:
    y_pred = (y_score >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division="warn")
    recall = recall_score(y_true, y_pred, zero_division="warn")
    f1 = f1_score(y_true, y_pred, zero_division="warn")
    roc_auc = roc_auc_score(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics, y_pred


def optimize_threshold(y_true: Any, y_score: np.ndarray) -> float:
    thresholds = np.linspace(0.2, 0.8, 61)
    best_threshold = 0.5
    best_f1 = -1.0

    for th in thresholds:
        candidate = (y_score >= th).astype(int)
        score = f1_score(y_true, candidate, zero_division=0)  # type: ignore[arg-type]
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_threshold = float(th)

    return best_threshold


def max_combinations(param_distributions: dict[str, list[Any]]) -> int:
    if not param_distributions:
        return 1
    return int(np.prod([len(v) for v in param_distributions.values()], dtype=np.int64))


def log_run(
    run_name: str,
    model_key: str,
    model_display_name: str,
    model_class_name: str,
    variant: str,
    estimator: Any,
    metrics: dict[str, float],
    train_time_sec: float,
    dataset_name: str,
    dataset_size: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    num_cols: list[str],
    cat_cols: list[str],
    best_params: dict[str, Any] | None,
    best_cv_roc_auc: float | None,
    report_text: str,
    cm_text: str,
) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_key", model_key)
        mlflow.set_tag("model_display_name", model_display_name)
        mlflow.set_tag("model_class", model_class_name)
        mlflow.set_tag("variant", variant)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("dataset_size", dataset_size)
        mlflow.log_param("numeric_features", len(num_cols))
        mlflow.log_param("categorical_features", len(cat_cols))
        mlflow.log_param("model_name", model_display_name)
        mlflow.log_param("model_class", model_class_name)
        mlflow.log_param("threshold_selection", "validation_f1" if variant == "tuned" else "fixed_0.5")

        if best_params:
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", str(value))
        if best_cv_roc_auc is not None:
            mlflow.log_metric("best_cv_roc_auc", float(best_cv_roc_auc))

        mlflow.log_metric("training_time_sec", round(train_time_sec, 4))
        mlflow.log_metric("train_samples", int(train_samples))
        mlflow.log_metric("validation_samples", int(val_samples))
        mlflow.log_metric("test_samples", int(test_samples))

        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        mlflow_sklearn.log_model(estimator, "model")
        mlflow.log_text(report_text, "classification_report.txt")
        mlflow.log_text(cm_text, "confusion_matrix.txt")


def print_results(results: list[dict[str, Any]]) -> None:
    results_df = pd.DataFrame(results)
    display_cols = [
        "model",
        "variant",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "training_time_sec",
    ]
    print("\n=== Test Metrics (Baseline vs Tuned) ===")
    by_cols: list[str] = ["model", "variant"]
    out_df = results_df[display_cols].sort_values(by=by_cols)  # type: ignore[arg-type]
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    delta_rows: list[dict[str, Any]] = []
    metric_cols = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "training_time_sec",
    ]

    for model_name in sorted(results_df["model"].unique()):
        model_rows = results_df[results_df["model"] == model_name].set_index("variant")
        if "baseline" not in model_rows.index or "tuned" not in model_rows.index:
            continue
        baseline = model_rows.loc["baseline"]
        tuned = model_rows.loc["tuned"]

        row = {"model": model_name}
        for metric in metric_cols:
            row[f"delta_{metric}"] = float(tuned[metric]) - float(baseline[metric])
        delta_rows.append(row)

    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        print("\n=== Delta (Tuned - Baseline) ===")
        print(delta_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


def load_dataset() -> tuple[pd.DataFrame, pd.Series, str]:
    digits: Any = load_digits(as_frame=True)
    X = pd.DataFrame(digits.data) if not isinstance(digits.data, pd.DataFrame) else digits.data.copy()
    y = pd.Series((digits.target == 8).astype(int))
    return X, y, "sklearn-digits-8-vs-rest"


def main() -> None:
    print("Loading dataset...")
    X, y, dataset_name = load_dataset()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    preprocessors = build_preprocessors(num_cols=num_cols, cat_cols=cat_cols)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.25,
        stratify=y_train_all,
        random_state=RANDOM_STATE,
    )

    print(f"Dataset: {dataset_name}")
    print(f"Dataset rows: {len(X)}")
    print(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"Features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    specs = build_model_specs()
    missing_optional = []
    if XGBClassifier is None:
        missing_optional.append("xgboost")
    if LGBMClassifier is None:
        missing_optional.append("lightgbm")
    if missing_optional:
        print(
            "Optional models not available:",
            ", ".join(missing_optional),
            "| Install with: pip install xgboost lightgbm",
        )

    results: list[dict[str, Any]] = []
    run_group = f"{dataset_name}-{int(time.time())}"

    for spec in specs:
        print(f"\nTraining {spec.key} baseline...")
        baseline_pipe = build_pipeline(spec, preprocessors)
        model_class_name = baseline_pipe.named_steps["clf"].__class__.__name__

        t0 = time.perf_counter()
        baseline_pipe.fit(X_train, y_train)
        baseline_time = time.perf_counter() - t0

        baseline_scores = predict_scores(baseline_pipe, X_test)
        baseline_metrics, baseline_pred = evaluate_metrics(y_test, baseline_scores, threshold=0.5)
        baseline_report = str(classification_report(y_test, baseline_pred, digits=4, zero_division="warn"))
        baseline_cm = confusion_matrix(y_test, baseline_pred)
        b_tn, b_fp, b_fn, b_tp = baseline_cm.ravel()
        baseline_cm_text = (
            f"Confusion Matrix:\n{baseline_cm}\n\n"
            f"TN: {b_tn}, FP: {b_fp}\nFN: {b_fn}, TP: {b_tp}"
        )

        baseline_run = f"{spec.display_name} | baseline | {dataset_name} | {run_group}"
        log_run(
            run_name=baseline_run,
            model_key=spec.key,
            model_display_name=spec.display_name,
            model_class_name=model_class_name,
            variant="baseline",
            estimator=baseline_pipe,
            metrics=baseline_metrics,
            train_time_sec=baseline_time,
            dataset_name=dataset_name,
            dataset_size=len(X),
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            num_cols=num_cols,
            cat_cols=cat_cols,
            best_params=None,
            best_cv_roc_auc=None,
            report_text=baseline_report,
            cm_text=baseline_cm_text,
        )

        print(f"Tuning {spec.key}...")
        X_tune, y_tune = tune_subset(X_train, y_train)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        n_iter = min(spec.n_iter, max_combinations(spec.param_distributions))

        search = RandomizedSearchCV(
            estimator=build_pipeline(spec, preprocessors),
            param_distributions=spec.param_distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            n_jobs=1,
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            verbose=0,
        )

        search.fit(X_tune, y_tune)
        best_params = search.best_params_

        tuned_pipe = build_pipeline(spec, preprocessors)
        tuned_pipe.set_params(**best_params)

        t0 = time.perf_counter()
        tuned_pipe.fit(X_train, y_train)
        tuned_time = time.perf_counter() - t0

        val_scores = predict_scores(tuned_pipe, X_val)
        tuned_threshold = optimize_threshold(y_val, val_scores)
        tuned_scores = predict_scores(tuned_pipe, X_test)
        tuned_metrics, tuned_pred = evaluate_metrics(y_test, tuned_scores, threshold=tuned_threshold)

        tuned_report = str(classification_report(y_test, tuned_pred, digits=4, zero_division="warn"))
        tuned_cm = confusion_matrix(y_test, tuned_pred)
        t_tn, t_fp, t_fn, t_tp = tuned_cm.ravel()
        tuned_cm_text = (
            f"Confusion Matrix:\n{tuned_cm}\n\n"
            f"TN: {t_tn}, FP: {t_fp}\nFN: {t_fn}, TP: {t_tp}"
        )

        tuned_run = f"{spec.display_name} | tuned | {dataset_name} | {run_group}"
        log_run(
            run_name=tuned_run,
            model_key=spec.key,
            model_display_name=spec.display_name,
            model_class_name=model_class_name,
            variant="tuned",
            estimator=tuned_pipe,
            metrics=tuned_metrics,
            train_time_sec=tuned_time,
            dataset_name=dataset_name,
            dataset_size=len(X),
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            num_cols=num_cols,
            cat_cols=cat_cols,
            best_params=best_params,
            best_cv_roc_auc=float(search.best_score_),
            report_text=tuned_report,
            cm_text=tuned_cm_text,
        )

        print(
            f"  {spec.key}: baseline f1={baseline_metrics['f1_score']:.4f}, "
            f"tuned f1={tuned_metrics['f1_score']:.4f}, "
            f"threshold={tuned_threshold:.2f}"
        )

        results.append(
            {
                "model": spec.key,
                "variant": "baseline",
                **baseline_metrics,
                "training_time_sec": round(baseline_time, 4),
            }
        )
        results.append(
            {
                "model": spec.key,
                "variant": "tuned",
                **tuned_metrics,
                "training_time_sec": round(tuned_time, 4),
            }
        )

    print_results(results)
    print(f"\nDone. Logged {len(results)} runs to experiment '{EXPERIMENT_NAME}'.")
    print("Tracking URI:", TRACKING_URI)


if __name__ == "__main__":
    main()
