## Event-driven MLOps (AWS) — local scaffold

Small, focused repo for an event-driven ML platform on AWS (S3 + orchestration + registry + triggers + inference), starting with a runnable local dev workflow and AWS smoke checks.

### Step 1 (local setup + smoke tests)

1) Create venv + install deps:

```bash
make install
```

2) Fill `.env` with your values (created manually; not committed). See `.env.example` for required keys.

3) Validate AWS credentials / identity:

```bash
make smoke-sts
```

4) Validate S3 access (bucket + prefix listing):

```bash
make smoke-s3
```

5) (Optional) Create standard S3 prefixes (folder markers):

```bash
make bootstrap-s3
```

### Notes
- AWS resources are created manually by you (nothing in this repo provisions AWS yet).
