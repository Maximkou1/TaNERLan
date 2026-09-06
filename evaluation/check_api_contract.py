from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import rich_click as click

from evaluation.core import (
    ContractError,
    normalize_base_url,
    predict_batch,
    validate_timeouts,
    wait_for_health,
)

PROBE_ITEMS = [
    {"hash": "contract-latin", "text": "Ali Toshkent shahrida ishlaydi."},
    {"hash": "contract-cyrillic", "text": "Алишер Навоий Тошкентда туғилган."},
]


def run(args: SimpleNamespace) -> None:
    """Проверяет health и predict работающего сервиса."""

    validate_timeouts(args.startup_timeout, args.request_timeout)
    base_url = normalize_base_url(args.url)
    wait_for_health(base_url, args.startup_timeout, args.request_timeout)
    print("OK  GET /healthz")

    _, entity_count = predict_batch(base_url, PROBE_ITEMS, args.request_timeout)
    print(
        f"OK  POST /api/v1/predict ({len(PROBE_ITEMS)} documents, {entity_count} returned entities)"
    )
    print("Service contract: OK")


@click.command(
    help="Check the hackathon HTTP API contract.",
    context_settings={"show_default": True},
)
@click.option("--url", default="http://localhost:8000")
@click.option("--startup-timeout", type=float, default=300.0)
@click.option("--request-timeout", type=float, default=120.0)
def main(**options: Any) -> None:
    """Запускает проверку с компактным сообщением об ошибке."""

    try:
        run(SimpleNamespace(**options))
    except (ContractError, OSError, TypeError, ValueError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
