from __future__ import annotations

from rich.console import Console
from rich.table import Table

from src.common.aws import sts_client
from src.common.config import load_settings


def main() -> None:
    console = Console()
    settings = load_settings()

    sts = sts_client(settings=settings)
    ident = sts.get_caller_identity()

    table = Table(title="AWS STS Caller Identity", show_header=True, header_style="bold")
    table.add_column("Account", style="cyan")
    table.add_column("Arn", style="green")
    table.add_row(str(ident.get("Account", "")), str(ident.get("Arn", "")))

    console.print(table)


if __name__ == "__main__":
    main()
