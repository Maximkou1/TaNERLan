from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click

from evaluation.core import (
    ContractError,
    JsonObject,
    evaluate_files,
    load_gold,
    normalize_base_url,
    predict_batch,
    print_metrics,
    validate_timeouts,
    wait_for_health,
    write_metrics,
)


def _resolve_paths(args: SimpleNamespace) -> tuple[Path, Path, Path]:
    """Проверяет пути, исключая перезапись gold или predictions метриками."""

    gold_path = args.gold.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if gold_path == predictions_path:
        raise ValueError("predictions path must differ from gold path")
    if output_path in {gold_path, predictions_path}:
        raise ValueError("output path must differ from gold and predictions paths")
    return gold_path, predictions_path, output_path


def _compact_prediction(result: JsonObject) -> JsonObject:
    """Оставляет только поля, используемые exact-span scorer."""

    entities = [
        {
            "label": entity["label"],
            "start": entity["start"],
            "end": entity["end"],
        }
        for entity in result["entities"]
    ]
    return {"hash": result["hash"], "entities": entities}


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    """Записывает проверенные ответы сервиса в JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def run(args: SimpleNamespace) -> JsonObject:
    """Прогоняет gold через HTTP API, сохраняет ответы и считает метрики."""

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    validate_timeouts(args.startup_timeout, args.request_timeout)
    base_url = normalize_base_url(args.url)
    gold_path, predictions_path, output_path = _resolve_paths(args)
    gold_records, _ = load_gold(gold_path)

    wait_for_health(base_url, args.startup_timeout, args.request_timeout)
    print("OK  GET /healthz")

    predictions: list[JsonObject] = []
    entity_count = 0
    batch_count = math.ceil(len(gold_records) / args.batch_size)
    for batch_index, start in enumerate(
        range(0, len(gold_records), args.batch_size),
        start=1,
    ):
        records = gold_records[start : start + args.batch_size]
        inputs = [
            {"hash": record["hash"], "text": record["text"]} for record in records
        ]
        try:
            results, batch_entity_count = predict_batch(
                base_url,
                inputs,
                args.request_timeout,
            )
        except ContractError as error:
            raise ContractError(
                f"batch {batch_index}/{batch_count}, records {start + 1}-{start + len(records)}: "
                f"{error}"
            ) from error
        predictions.extend(_compact_prediction(result) for result in results)
        entity_count += batch_entity_count
        if batch_index % 10 == 0 or batch_index == batch_count:
            print(
                f"Predict: {start + len(records)}/{len(gold_records)} records "
                f"({batch_index}/{batch_count} batches)"
            )

    _write_jsonl(predictions_path, predictions)
    print(f"Predictions: {predictions_path} ({entity_count} entities)")

    metrics = evaluate_files(gold_path, predictions_path)
    print_metrics(metrics)
    write_metrics(output_path, metrics)
    print(f"Metrics: {output_path}")
    return metrics


@click.command(
    help="Run a service on gold JSONL and calculate exact-span NER metrics.",
    context_settings={"show_default": True},
)
@click.option("--url", default="http://localhost:8000")
@click.option("--gold", type=click.Path(path_type=Path), default=Path("data/dev.jsonl"))
@click.option(
    "--predictions",
    type=click.Path(path_type=Path),
    default=Path("artifacts/service/dev_predictions.jsonl"),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("artifacts/service/dev_metrics.json"),
)
@click.option("--batch-size", type=int, default=8)
@click.option("--startup-timeout", type=float, default=300.0)
@click.option("--request-timeout", type=float, default=120.0)
def main(**options: Any) -> None:
    """Запускает service evaluation с компактным сообщением об ошибке."""

    try:
        run(SimpleNamespace(**options))
    except (ContractError, OSError, TypeError, ValueError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
