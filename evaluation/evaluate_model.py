from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click

from evaluation.core import JsonObject, evaluate_files, print_metrics, write_metrics


def run(args: SimpleNamespace) -> JsonObject:
    """Валидирует файлы, рассчитывает метрики и при необходимости пишет JSON."""

    gold_path = args.gold.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    metrics = evaluate_files(gold_path, predictions_path)
    print_metrics(metrics)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        write_metrics(output_path, metrics)
        print(f"Metrics: {output_path}")
    return metrics


@click.command(
    help="Calculate exact-span NER metrics.",
    context_settings={"show_default": True},
)
@click.option("--gold", type=click.Path(path_type=Path), required=True)
@click.option("--predictions", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path))
def main(**options: Any) -> None:
    """Запускает scorer с компактным сообщением об ошибке."""

    try:
        run(SimpleNamespace(**options))
    except (OSError, TypeError, ValueError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
