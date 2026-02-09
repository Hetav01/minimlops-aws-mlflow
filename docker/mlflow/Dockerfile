FROM python:3.12

LABEL maintainer="hetav" \
      description="MLflow Tracking Server – ECS EC2 / ALB"

# Avoid prompts, keep image small
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow==3.9 boto3

# Persistent volume mount point for SQLite DB
RUN mkdir -p /mlflow
VOLUME /mlflow

COPY docker/mlflow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]