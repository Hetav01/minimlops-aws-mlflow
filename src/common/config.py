from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Settings:
    aws_region: str
    s3_bucket: str
    project_prefix: str
    sns_topic_arn: str | None = None


def _missing(required: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for k in required:
        v = os.getenv(k)
        if v is None or not v.strip():
            missing.append(k)
    return missing


def _normalize_project_prefix(prefix: str) -> str:
    # Keep S3 keys clean and predictable.
    p = prefix.strip()
    p = p.lstrip("/")  # S3 keys never start with /
    if p and not p.endswith("/"):
        p += "/"
    return p


def load_settings(*, env_file: str | None = None) -> Settings:
    """
    Load settings from `.env` + process env, validate required keys.

    Precedence: existing process env wins over `.env`.
    """
    dotenv_path = env_file or find_dotenv(usecwd=True)
    loaded_dotenv = False
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        loaded_dotenv = True

    required = ["AWS_REGION", "S3_BUCKET", "PROJECT_PREFIX"]
    missing = _missing(required)
    if missing:
        keys = "\n".join(f"- {k}" for k in missing)
        dotenv_hint = (
            f"\n\nDotenv:\n- Loaded from: {dotenv_path}"
            if loaded_dotenv
            else "\n\nDotenv:\n- No `.env` file was found. Create one in the repo root (you can copy `.env.example`)."
        )
        raise SystemExit(
            "Missing required environment variables.\n"
            "Set them in your `.env` (recommended) or export them in your shell:\n"
            f"{keys}\n"
            "\nExpected keys for Step 1:\n"
            "- AWS_REGION\n- S3_BUCKET\n- PROJECT_PREFIX\n"
            "\nOptional (for later):\n- SNS_TOPIC_ARN\n"
            f"{dotenv_hint}\n"
        )

    project_prefix = _normalize_project_prefix(os.environ["PROJECT_PREFIX"])
    sns_topic_arn = os.getenv("SNS_TOPIC_ARN") or None

    return Settings(
        aws_region=os.environ["AWS_REGION"],
        s3_bucket=os.environ["S3_BUCKET"],
        project_prefix=project_prefix,
        sns_topic_arn=sns_topic_arn,
    )

