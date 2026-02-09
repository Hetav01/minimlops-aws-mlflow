## MLflow Tracking Server — Docker image (Step 2)

Minimal image for running MLflow Tracking + Model Registry on **ECS EC2 behind an ALB**.

### Build & run locally

```bash
docker build -t mlflow-tracking:latest -f docker/mlflow/Dockerfile docker/mlflow/

docker run --rm -p 5000:5000 \
  -e MLFLOW_ARTIFACT_ROOT=s3://<BUCKET>/<PROJECT_PREFIX>mlflow-artifacts/ \
  -e AWS_REGION=us-east-1 \
  mlflow-tracking:latest
```

### Required environment variables

| Variable | Example | Notes |
|----------|---------|-------|
| MLFLOW_ARTIFACT_ROOT | s3://my-bucket/project/mlflow-artifacts/ | Required. S3 path for artifacts. |
| AWS_REGION | us-east-1 | Needed by boto3 / S3 client. |

### Optional environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| MLFLOW_BACKEND_STORE_URI | sqlite:////mlflow/mlflow.db | Swap for RDS Postgres later. |
| MLFLOW_HOST | 0.0.0.0 | |
| MLFLOW_PORT | 5000 | Must match ECS task container port. |
| MLFLOW_WORKERS | 2 | Gunicorn workers. Tune to vCPU count. |

### ECS notes

- Mount an EBS volume at `/mlflow` so the SQLite DB persists across container restarts.
- The ECS task role needs `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on the artifact bucket.
- The ECS execution role only needs ECR pull + CloudWatch Logs permissions.

---

## Step 2 — MLflow on ECS (EC2) behind ALB

### Architecture (single-service)

```
Client ──▶ ALB :443/80 ──▶ ECS Service (EC2) ──▶ MLflow container :5000
│
├── SQLite @ /mlflow/mlflow.db (EBS volume)
└── S3 artifact bucket
```

### Security groups — recommended pattern

| Resource | Inbound | Outbound |
|----------|---------|----------|
| **ALB SG** | TCP 443 (or 80) from your IP / VPN CIDR | All → ECS SG |
| **ECS Instance SG** | TCP 5000 **from ALB SG only** | All (S3, ECR, CW) |

Key point: the ECS instances should **never** be directly reachable from the internet. Only the ALB forwards traffic to port 5000.

### IAM roles (least privilege)

| Role | Permissions |
|------|-------------|
| **Task role** | `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on artifact bucket + prefix |
| **Execution role** | `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `logs:CreateLogStream`, `logs:PutLogEvents` |

> Keep these separate. The execution role is used by the ECS agent; the task role is assumed by the container itself. Never give the execution role S3 access.

### CloudWatch Logs

- Use the `awslogs` log driver in your ECS task definition.
- Recommended log group: `/ecs/mlflow-tracking`
- Set retention to **14–30 days** (avoid unbounded growth).

Example task-definition snippet (JSON):

```json
"logConfiguration": {
    "logDriver": "awslogs",
    "options": {
        "awslogs-group": "/ecs/mlflow-tracking",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "mlflow"
    }
}
```

### Resource limits

| Setting | Recommended starting point | Notes |
|---------|----------------------------|-------|
| CPU | 512 (0.5 vCPU) | MLflow server is lightweight at low scale. |
| Memory | 1024 MiB | Increase if large experiment lists cause OOM. |
| Workers | 2 | Match to vCPU; set via MLFLOW_WORKERS. |

### SQLite limitations & upgrade path

SQLite works fine for single-writer, low-concurrency usage (solo or small team).

**Known limitations:**

- No concurrent writes — only one Gunicorn worker should write at a time.
- No network access — DB is local to the container; if the container dies without EBS, data is lost.
- No connection pooling.

**When to upgrade:**

- Multiple concurrent training jobs logging at the same time.
- Team size > 2–3 people using the tracking server.
- You need HA / multi-AZ.

**Upgrade target:** Amazon RDS PostgreSQL (db.t4g.micro for dev). Change `MLFLOW_BACKEND_STORE_URI` to `postgresql://user:pass@host:5432/mlflow` — no image rebuild needed.

### Health check

- MLflow exposes `GET /health` (returns OK, HTTP 200).
- ALB target group health check path: `/health`
- Docker HEALTHCHECK is already baked into the Dockerfile.
- Recommended ALB settings: interval 30 s, timeout 5 s, healthy threshold 2, unhealthy threshold 3.

---

## ECR build / tag / push commands

Replace the placeholder values with your own:

```bash
# ── Variables (set these) ─────────────────────────────────────
AWS_ACCOUNT_ID="123456789012"
AWS_REGION="us-east-1"
ECR_REPO="mlflow-tracking"
IMAGE_TAG="latest"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

# ── 1. Authenticate Docker to ECR ────────────────────────────
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URI}"

# ── 2. Build ──────────────────────────────────────────────────
docker build -t "${ECR_REPO}:${IMAGE_TAG}" \
  -f docker/mlflow/Dockerfile docker/mlflow/

# ── 3. Tag ────────────────────────────────────────────────────
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

# ── 4. Push ───────────────────────────────────────────────────
docker push "${ECR_URI}:${IMAGE_TAG}"
```

> **Tip:** If the ECR repo doesn't exist yet, create it first:
>
> ```bash
> aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}"
> ```
