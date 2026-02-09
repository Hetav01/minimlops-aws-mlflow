## Training (placeholder)

This folder will hold training code and (later) SageMaker Pipeline definitions.

Planned direction:
- Pipeline entrypoints for train/retrain
- Artifacts written to `s3://{S3_BUCKET}/{PROJECT_PREFIX}artifacts/`
- Models written/registered under `.../models/` (and later via MLflow registry)
