from __future__ import annotations

import boto3

from .config import Settings


def client(service_name: str, *, settings: Settings):
    """
    Create a boto3 client using the configured AWS region.

    Note on retries:
    - boto3/botocore already includes standard retry behavior.
    - For Step 1, we keep defaults (no custom retry/backoff logic).
    """
    return boto3.client(service_name, region_name=settings.aws_region)


def s3_client(*, settings: Settings):
    return client("s3", settings=settings)


def sts_client(*, settings: Settings):
    return client("sts", settings=settings)

