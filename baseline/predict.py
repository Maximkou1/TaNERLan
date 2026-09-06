from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification

from .common import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_STRIDE,
    TAGS,
    JsonObject,
    ModelFeature,
    Offsets,
    decode_bio_tokens,
    load_fast_tokenizer,
    read_records,
    resolve_device,
    tokenize_windows,
    validate_window,
)

Window = tuple[int, ModelFeature, Offsets]


def _read_baseline_config(model_dir: Path) -> JsonObject:
    """Читает параметры окон, сохранённые скриптом обучения."""

    path = model_dir / "baseline_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _validate_args(args: SimpleNamespace) -> None:
    """Проверяет параметры batch и ограничения отладочной выборки."""

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("max-records must be positive")
    if args.max_length is not None and args.max_length < 1:
        raise ValueError("max-length must be positive")


def _model_labels(model: torch.nn.Module) -> dict[int, str]:
    """Извлекает и проверяет BIO-метки из model config."""

    labels = {int(index): str(label) for index, label in model.config.id2label.items()}
    expected = set(TAGS)
    if set(labels.values()) != expected or set(labels) != set(range(len(TAGS))):
        raise ValueError(f"model labels must be exactly {list(TAGS)}")
    return labels


def _build_windows(
    records: list[JsonObject],
    tokenizer: Any,
    *,
    max_length: int,
    stride: int,
) -> list[Window]:
    """Токенизирует все документы и связывает окна с индексами записей."""

    windows: list[Window] = []
    for record_index, record in enumerate(tqdm(records, desc="Tokenize", unit="doc")):
        for feature, offsets in tokenize_windows(
            tokenizer,
            record["text"],
            max_length=max_length,
            stride=stride,
        ):
            windows.append((record_index, feature, offsets))
    return windows


@torch.inference_mode()
def _predict_token_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    windows: list[Window],
    record_count: int,
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[tuple[int, int], tuple[torch.Tensor, int]]]:
    """Усредняет вероятности одинаковых токенов из перекрывающихся окон."""

    aggregated: list[dict[tuple[int, int], tuple[torch.Tensor, int]]] = [
        {} for _ in range(record_count)
    ]
    model.eval()
    for batch_start in tqdm(
        range(0, len(windows), batch_size),
        desc="Predict",
        unit="batch",
    ):
        batch_windows = windows[batch_start : batch_start + batch_size]
        batch = tokenizer.pad(
            [feature for _, feature, _ in batch_windows],
            padding=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        probabilities = torch.softmax(model(**batch).logits.float(), dim=-1).cpu()

        for row_index, (record_index, _, offsets) in enumerate(batch_windows):
            record_scores = aggregated[record_index]
            for token_index, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                key = (start, end)
                score = probabilities[row_index, token_index]
                if key in record_scores:
                    previous, count = record_scores[key]
                    record_scores[key] = (previous + score, count + 1)
                else:
                    record_scores[key] = (score.clone(), 1)
    return aggregated


def _decode_records(
    records: list[JsonObject],
    scores: list[dict[tuple[int, int], tuple[torch.Tensor, int]]],
    id2label: dict[int, str],
) -> list[JsonObject]:
    """Преобразует усреднённые token scores в JSONL-предсказания spans."""

    predictions: list[JsonObject] = []
    for record, record_scores in zip(records, scores, strict=True):
        tagged_tokens = []
        for (start, end), (score_sum, count) in sorted(record_scores.items()):
            label_id = int((score_sum / count).argmax().item())
            tagged_tokens.append((start, end, id2label[label_id]))
        predictions.append(
            {
                "hash": record["hash"],
                "entities": decode_bio_tokens(tagged_tokens),
            }
        )
    return predictions


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    """Записывает предсказания по одному JSON-объекту на строку."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def run(args: SimpleNamespace) -> Path:
    """Загружает checkpoint и строит exact-span предсказания для JSONL."""

    _validate_args(args)
    model_dir = args.model_dir.expanduser().resolve()
    config = _read_baseline_config(model_dir)
    max_length = (
        args.max_length
        if args.max_length is not None
        else int(config.get("max_length", DEFAULT_MAX_LENGTH))
    )
    stride = args.stride if args.stride is not None else int(config.get("stride", DEFAULT_STRIDE))
    device = resolve_device(args.device)
    tokenizer = load_fast_tokenizer(str(model_dir))
    validate_window(tokenizer, max_length, stride)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    id2label = _model_labels(model)

    records = read_records(
        args.input.expanduser().resolve(),
        require_entities=False,
        limit=args.max_records,
    )
    windows = _build_windows(
        records,
        tokenizer,
        max_length=max_length,
        stride=stride,
    )
    scores = _predict_token_scores(
        model,
        tokenizer,
        windows,
        len(records),
        batch_size=args.batch_size,
        device=device,
    )
    predictions = _decode_records(records, scores, id2label)
    output_path = args.output.expanduser().resolve()
    _write_jsonl(output_path, predictions)
    print(f"Device: {device}")
    print(f"Records: {len(records)}, windows: {len(windows)}")
    print(f"Predictions: {output_path}")
    return output_path


@click.command(
    help="Run the minimal NER baseline.",
    context_settings={"show_default": True},
)
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/baseline/model"),
)
@click.option("--input", type=click.Path(path_type=Path), default=Path("dev.jsonl"))
@click.option("--output", type=click.Path(path_type=Path), default=Path("predictions.jsonl"))
@click.option("--batch-size", type=int, default=16)
@click.option("--max-length", type=int)
@click.option("--stride", type=int)
@click.option("--max-records", type=int)
@click.option("--device", type=click.Choice(("auto", "cpu", "cuda")), default="auto")
def main(**options: Any) -> None:
    """Запускает CLI инференса с компактным сообщением об ошибке."""

    try:
        run(SimpleNamespace(**options))
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
