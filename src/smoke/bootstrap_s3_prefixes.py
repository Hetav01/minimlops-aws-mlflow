from __future__ import annotations

from botocore.exceptions import ClientError
from rich.console import Console

from src.common.aws import s3_client
from src.common.config import load_settings


def _join_prefix(base: str, suffix: str) -> str:
    b = base
    if b and not b.endswith("/"):
        b += "/"
    s = suffix.lstrip("/")
    return f"{b}{s}"


def main() -> None:
    console = Console()
    settings = load_settings()
    s3 = s3_client(settings=settings)

    bucket = settings.s3_bucket
    base = settings.project_prefix

    required = [
        _join_prefix(base, "data/training/"),
        _join_prefix(base, "data/inference/"),
        _join_prefix(base, "data/ground_truth/"),
        _join_prefix(base, "artifacts/"),
        _join_prefix(base, "models/"),
    ]

    console.print(f"[bold]Bootstrapping S3 prefixes[/bold] in s3://{bucket}/{base}")

    created = 0
    for key in required:
        # "Folder markers" are zero-byte objects with trailing slash keys.
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=b"")
            console.print(f"  - created: s3://{bucket}/{key}")
            created += 1
        except ClientError as e:
            raise SystemExit(f"Failed to create s3://{bucket}/{key}\nError: {e}") from e

    console.print(f"[green]Done.[/green] Created {created} marker objects.")


if __name__ == "__main__":
    main()
