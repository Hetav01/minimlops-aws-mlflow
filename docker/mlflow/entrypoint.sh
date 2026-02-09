#!/usr/bin/env bash
set -euo pipefail

BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:////mlflow/mlflow.db}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT env var is required}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"
WORKERS="${MLFLOW_WORKERS:-2}"

echo "──────────────────────────────────────"
echo " MLflow Tracking Server"
echo "  backend : ${BACKEND_STORE_URI}"
echo "  artifacts: ${ARTIFACT_ROOT}"
echo "  host:port: ${HOST}:${PORT}"
echo "  workers  : ${WORKERS}"
echo "──────────────────────────────────────"

exec mlflow server \
    --host "${HOST}" \
    --port "${PORT}" \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --workers "${WORKERS}"