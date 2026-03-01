# 🚀 MiniMLOps  
### Event-Driven ML Training + Retraining + Batch Inference on AWS  
**MLflow + SageMaker + ECS (EC2) + S3 + Lambda + SNS/SQS + EventBridge**

MiniMLOps is a small but production-flavored MLOps platform that demonstrates:

- ✅ MLflow Tracking Server on ECS (EC2) behind an ALB  
- ✅ S3-backed artifact storage  
- ✅ SageMaker training jobs logging directly to MLflow  
- ✅ Model Registry with versioning + promotion (Staging → Production)  
- ✅ Event-driven retraining via Lambda + SNS/SQS + EventBridge  
- ✅ Batch inference + evaluation + automated retrain triggers  
- ✅ Architecture extensible to LLM monitoring (LLMOps)

This project is intentionally designed to resemble a real production ML lifecycle — not just a notebook demo.

---

# 🏗 Architecture Overview

```mermaid
flowchart LR
    A[Developer / CI] --> EB[EventBridge]
    EB --> L1[Lambda Router]
    S3E[S3 ObjectCreated] --> L1
    L1 --> SNS[SNS Topic]
    SNS --> SQS[SQS Queue]

    SQS -->|train| SM[SageMaker Training Job]
    SM -->|logs| ALB[ALB DNS]
    ALB --> ML[MLflow on ECS]
    SM -->|artifacts| S3A[(S3 Artifacts)]
    ML --> S3A
    ML --> REG[Model Registry]

    SQS -->|batch infer| BI[Batch Inference Job]
    BI --> S3P[(Predictions S3)]
    S3P --> EVAL[Evaluation Job]
    EVAL -->|metric drop| L1
    EVAL -->|promote best| REG
````

---

# 📂 Repository Structure

```
.
├── docker/
├── docs/
│   ├── README.md
│   └── manual_aws_checklist.md
├── src/
│   ├── common/
│   │   ├── aws.py
│   │   └── config.py
│   ├── smoke/
│   │   ├── bootstrap_s3_prefixes.py
│   │   ├── smoke_sts.py
│   │   ├── smoke_s3.py
│   │   ├── smoke_mlflow.py
│   │   ├── smoke_same_model.py
│   │   └── smoke_multi_model.py
│   └── training/
│       ├── README.md
│       └── __init__.py
├── Makefile
├── requirements.txt
└── README.md
```

---

# 🔧 Prerequisites

* Python 3.10+
* AWS Account
* AWS CLI configured
* Permissions for:

  * ECS
  * ECR
  * S3
  * ALB
  * SageMaker
  * Lambda
  * SNS / SQS
  * EventBridge
  * CloudWatch

---

# ⚙️ Environment Setup

Create `.env`:

```bash
AWS_REGION=us-east-1
S3_BUCKET=minimlops-lewis44
PROJECT_PREFIX=mlops-demo/

MLFLOW_TRACKING_URI=http://<ALB-DNS>

ENV=dev
MODEL_NAME=minimlops-sklearn
SNS_TOPIC_NAME=minimlops-alerts
```

> ⚠️ Important: SageMaker must be able to reach your ALB DNS.
> If ALB inbound rules only allow your laptop IP, SageMaker cannot log to MLflow.

---

# 🧪 Local Validation (Smoke Tests)

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run smoke tests:

```bash
make smoke
```

Or individually:

```bash
python -m src.smoke.smoke_sts
python -m src.smoke.smoke_s3
python -m src.smoke.smoke_mlflow
```

Bootstrap required S3 prefixes:

```bash
make bootstrap-s3
```

Creates:

```
mlops-demo/
├── data/training/
├── data/inference/
├── data/ground_truth/
├── artifacts/
└── models/
```

---

# 🧠 MLflow Conventions

## Experiments

* `minimlops/training`
* `minimlops/eval`
* `minimlops/inference`

## Mandatory Run Tags

* `project=minimlops`
* `env=dev`
* `pipeline=training|eval|inference`
* `git_sha=<commit>`
* `dataset_version=<timestamp>`
* `trigger=manual|eventbridge|sns|sqs|s3`
* `model_name=minimlops-sklearn`

These ensure consistent comparison across iterations.

---

# 🚀 SageMaker Training Flow

Each SageMaker job:

1. Loads dataset from S3
2. Trains model (sklearn MVP)
3. Logs params + metrics + artifacts to MLflow
4. Registers model version in MLflow Registry

Example registration behavior:

```
minimlops-sklearn
  ├── Version 1
  ├── Version 2
  └── Version 3
```

Promotion flow:

```
None → Staging → Production
```

Artifacts are stored under MLflow run IDs in S3 (expected behavior).

---

# 🔁 Event-Driven Retraining Architecture

## Trigger Sources

* EventBridge schedule (weekly retrain)
* S3 ObjectCreated (new data)
* Metric degradation detected in evaluation
* Manual API call (future FastAPI extension)

## Lambda Router Responsibilities

* Validate payload
* Enrich with metadata (env, timestamp, tags)
* Publish to SNS
* Log failures
* Route to DLQ if necessary

## SNS + SQS Pattern

* SNS → fanout
* SQS → buffering + retries
* DLQ → reliability

---

# 📨 Example Training Trigger Payload

```json
{
  "type": "train",
  "model_name": "minimlops-sklearn",
  "s3_train_path": "s3://bucket/mlops-demo/data/training/train.csv",
  "dataset_version": "2026-02-28",
  "trigger": "eventbridge",
  "git_sha": "abc1234"
}
```

---

# 📊 Batch Inference + Evaluation Loop

1. Load Production model
2. Generate predictions
3. Write to S3
4. Join predictions with ground truth
5. Compute metrics
6. If performance drops → trigger retraining event

This closes the full MLOps loop:

```
Train → Register → Promote → Infer → Evaluate → Retrain
```

---

# 🤖 Extending to LLM Monitoring (LLMOps)

This architecture generalizes cleanly to LLM systems.

## What Changes?

Instead of retraining base weights every week, you iterate on:

* Prompt templates
* Retrieval configs (RAG)
* Embedding models
* Guardrails
* Fine-tuning datasets
* Feedback-driven tuning

## LLM Metrics to Monitor

* Hallucination rate
* Retrieval hit rate
* Answer faithfulness
* Task success rate
* Latency
* Token usage / cost
* Safety violations

Store logs in:

```
mlops-demo/llm/logs/
mlops-demo/llm/eval/
mlops-demo/llm/feedback/
```

## LLM Retrain Triggers

Lambda can trigger jobs when:

* Quality score drops below threshold
* Hallucination rate spikes
* Cost per request increases
* New documents added to RAG corpus
* User feedback crosses threshold

Jobs triggered may:

* Rebuild vector index
* Re-run offline evaluation suite
* Fine-tune smaller LLM
* Promote new prompt + retrieval configuration

Treat “prompt + retrieval config” as versioned artifacts just like model.pkl.

---

# 💰 Cost Control

Major cost contributors:

* EC2 instances in ECS cluster
* ALB hourly cost
* NAT Gateway
* SageMaker jobs while running

To pause project:

1. Scale ECS service to 0
2. Stop Auto Scaling Group
3. Optionally delete ALB
4. Ensure no SageMaker jobs are active

---

# 🛠 Troubleshooting

### SageMaker cannot log to MLflow

* Check ALB security group inbound rules
* Confirm MLFLOW_TRACKING_URI is public and reachable

### Weird artifact folder names

* Expected MLflow behavior (run IDs)

### Messages stuck in SQS

* Check DLQ
* Confirm IAM permissions to start SageMaker jobs

---

# 🎯 What This Project Demonstrates

* End-to-end ML lifecycle
* Production-style event-driven retraining
* Model versioning discipline
* Reliable queue-based orchestration
* Real AWS infrastructure integration
* LLMOps extension path
* Practical cost-awareness
* Reproducible MLOps architecture

