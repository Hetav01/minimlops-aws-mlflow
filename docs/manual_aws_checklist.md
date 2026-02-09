## Manual AWS checklist (placeholder)

This repo does not provision AWS resources yet. You will create these manually.

### Required for Step 1 smoke tests
- **S3 bucket**: `S3_BUCKET`
- **IAM credentials** (or SSO role/session) that can:
  - call `sts:GetCallerIdentity`
  - access the bucket (`s3:ListBucket`, `s3:GetBucketLocation`, `s3:PutObject` for bootstrap)

### Optional (later steps)
- **SNS topic**: `SNS_TOPIC_ARN`
- Eventing components (EventBridge/SQS/DLQ), pipeline execution role(s), and MLflow backing services.
