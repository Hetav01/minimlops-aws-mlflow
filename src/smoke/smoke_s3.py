from __future__ import annotations

from botocore.exceptions import ClientError
from rich.console import Console
from rich.table import Table

from src.common.aws import s3_client
from src.common.config import load_settings


def main() -> None:
    console = Console()
    settings = load_settings()

    s3 = s3_client(settings=settings)

    bucket = settings.s3_bucket
    prefix = settings.project_prefix

    console.print(f"[bold]Bucket[/bold]: s3://{bucket}")
    console.print(f"[bold]Prefix[/bold]: s3://{bucket}/{prefix}")

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        raise SystemExit(
            f"S3 bucket check failed for s3://{bucket}.\n"
            f"Error: {e}"
        ) from e

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

    common_prefixes: list[str] = []
    object_count = 0

    for page in pages:
        object_count += int(page.get("KeyCount", 0))
        for cp in page.get("CommonPrefixes", []) or []:
            p = cp.get("Prefix")
            if p:
                common_prefixes.append(p)

    table = Table(title="S3 Prefix Listing", show_header=True, header_style="bold")
    table.add_column("Discovered prefixes (1 level under PROJECT_PREFIX)")

    if common_prefixes:
        for p in sorted(set(common_prefixes)):
            table.add_row(f"s3://{bucket}/{p}")
    else:
        table.add_row("(none found — prefix is empty or only contains objects directly under it)")

    console.print(table)
    console.print(f"[dim]Objects scanned (API KeyCount sum): {object_count}[/dim]")


if __name__ == "__main__":
    main()
