"""Run one ready-to-use NER checkpoint and write repository JSONL predictions.

No model is loaded at import time. Downloads happen only after an explicit CLI
invocation by the user.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click
import torch
from tqdm.auto import tqdm

from baseline.common import decode_bio_tokens, load_fast_tokenizer, read_records, resolve_device
from evaluation.core import JsonObject, validate_entity

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("model_config.json")
OUR_LABELS = {"ORG", "NAME", "GEO"}


def _load_config(path: Path) -> dict[str, JsonObject]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"{path}: expected models[]")
    result = {}
    for item in models:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"{path}: every model needs a name")
        result[item["name"]] = item
    return result


def _model_spec(name: str, config_path: Path) -> JsonObject:
    models = _load_config(config_path)
    if name not in models:
        raise ValueError(f"unknown model {name!r}; choose one of: {', '.join(models)}")
    return models[name]


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _validate_predictions(predictions: list[JsonObject], records: list[JsonObject]) -> None:
    expected = {record["hash"]: record["text"] for record in records}
    actual = {record.get("hash") for record in predictions}
    if actual != set(expected):
        raise ValueError("prediction hashes do not exactly match input hashes")
    for index, prediction in enumerate(predictions, start=1):
        entities = prediction.get("entities")
        if not isinstance(entities, list):
            raise ValueError(f"prediction {index}: entities must be an array")
        seen: set[tuple[str, int, int]] = set()
        for entity in entities:
            key = validate_entity(entity, len(expected[prediction["hash"]]), f"prediction {index}")
            if key in seen:
                raise ValueError(f"prediction {index}: duplicate entity {key}")
            seen.add(key)


def _mapped_label(raw: str, label_map: dict[str, str]) -> str | None:
    normalized = raw.removeprefix("B-").removeprefix("I-").upper()
    mapped = label_map.get(raw) or label_map.get(normalized) or label_map.get(raw.lower())
    return mapped if mapped in OUR_LABELS else None


def _is_subword_continuation(token: str) -> bool:
    """Return whether a tokenizer token is a continuation piece."""

    # WordPiece uses ``##``; SentencePiece-style tokenizers mark the start of
    # a new word with ``▁`` (and byte-level BPE tokenizers with ``Ġ``).
    return token.startswith("##") or not token.startswith(("▁", "Ġ"))


def _merge_adjacent_subwords(
    tokens: list[tuple[int, int, str, str]]
) -> list[tuple[int, int, str]]:
    """Merge same-label WordPiece fragments split inside one source word.

    Some token-classification checkpoints can emit a ``B-`` tag for each
    WordPiece.  Merge only when the tokenizer explicitly marks the next token
    as a continuation and its offset is strictly adjacent to the previous
    piece. This avoids merging separate entities merely because their spans
    happen to touch.
    """

    merged: list[tuple[int, int, str]] = []
    for start, end, tag, token in tokens:
        if merged:
            previous_start, previous_end, previous_tag = merged[-1]
            previous_label = previous_tag.partition("-")[2]
            label = tag.partition("-")[2]
            if (
                previous_label == label
                and previous_end == start
                and _is_subword_continuation(token)
            ):
                merged[-1] = (previous_start, end, previous_tag)
                continue
        merged.append((start, end, tag))
    return merged


@torch.inference_mode()
def _run_token_classification(records: list[JsonObject], spec: JsonObject, args: SimpleNamespace, device: torch.device) -> list[JsonObject]:
    from transformers import AutoModelForTokenClassification

    print(f"Loading tokenizer: {spec['model_id']}", flush=True)
    tokenizer = load_fast_tokenizer(spec["model_id"])
    print(f"Loading model: {spec['model_id']}", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(spec["model_id"]).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Inference: device={device}, documents={len(records)}", flush=True)
    id2label = {int(index): str(label) for index, label in model.config.id2label.items()}
    predictions: list[JsonObject] = []
    for record in tqdm(records, desc="Predict", unit="doc"):
        encoded = tokenizer(record["text"], truncation=True, max_length=args.max_length,
                            stride=args.stride, return_offsets_mapping=True,
                            return_overflowing_tokens=True)
        input_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if input_ids and isinstance(input_ids[0], int):
            input_ids, offsets = [input_ids], [offsets]
        tokens: list[tuple[int, int, str, str]] = []
        for ids, window_offsets in zip(input_ids, offsets, strict=True):
            batch = tokenizer.pad([{"input_ids": ids, "attention_mask": [1] * len(ids)}], return_tensors="pt")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits[0]
            token_strings = tokenizer.convert_ids_to_tokens(ids)
            for token_index, (start, end) in enumerate(window_offsets):
                if start == end:
                    continue
                tag = id2label[int(logits[token_index].argmax().item())]
                mapped = _mapped_label(tag, spec["label_map"])
                prefix = tag.split("-", 1)[0] if "-" in tag else "B"
                tokens.append((int(start), int(end), f"{prefix}-{mapped}" if mapped else "O", token_strings[token_index]))
        unique = {(start, end): (tag, token) for start, end, tag, token in tokens}
        ordered_tokens = [
            (start, end, tag, token)
            for (start, end), (tag, token) in sorted(unique.items())
        ]
        merged_tokens = _merge_adjacent_subwords(ordered_tokens)
        entities = decode_bio_tokens(merged_tokens)
        predictions.append({"hash": record["hash"], "entities": entities})
    return predictions


def _extract_entity_items(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if not isinstance(result, dict):
        raise ValueError("GLiNER returned an unexpected result")
    entities = result.get("entities")
    if isinstance(entities, list):
        return [item for item in entities if isinstance(item, dict)]
    if isinstance(entities, dict):
        return [dict(item, label=label) for label, values in entities.items()
                if isinstance(values, list) for item in values if isinstance(item, dict)]
    raise ValueError("GLiNER result has no entities[]/entities{}")


def _text_windows(text: str, *, max_chars: int = 700, overlap_chars: int = 140) -> list[tuple[str, int]]:
    """Create conservative word-boundary windows without loading a tokenizer."""

    if overlap_chars >= max_chars:
        raise ValueError("GLiNER overlap must be smaller than max_chars")
    if len(text) <= max_chars:
        return [(text, 0)]
    windows: list[tuple[str, int]] = []
    first = 0
    while first < len(text):
        limit = min(len(text), first + max_chars)
        end = limit
        if limit < len(text):
            boundary = text.rfind(" ", first, limit)
            if boundary > first:
                end = boundary
        windows.append((text[first:end], first))
        if end >= len(text):
            break
        next_first = max(first + 1, end - overlap_chars)
        while next_first < len(text) and text[next_first].isspace():
            next_first += 1
        first = next_first
    return windows


def _run_gliner(records: list[JsonObject], spec: JsonObject, args: SimpleNamespace, device: torch.device) -> tuple[list[JsonObject], int]:
    if spec["backend"] == "gliner":
        from gliner import GLiNER
        print(f"Loading GLiNER model: {spec['model_id']}", flush=True)
        model = GLiNER.from_pretrained(spec["model_id"]).to(str(device))
    elif spec["backend"] == "gliner2.5":
        from gliner2 import AutoExtractor
        print(f"Loading GLiNER2.5 model: {spec['model_id']}", flush=True)
        model = AutoExtractor.from_pretrained(spec["model_id"], map_location=str(device), quantize=device.type == "cuda")
    else:
        from gliner2 import GLiNER2
        print(f"Loading GLiNER2 model: {spec['model_id']}", flush=True)
        model = GLiNER2.from_pretrained(spec["model_id"]).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Inference: device={device}, documents={len(records)}", flush=True)
    predictions: list[JsonObject] = []
    invalid_span_count = 0
    for record in tqdm(records, desc="Predict", unit="doc"):
        entities: list[JsonObject] = []
        seen: set[tuple[str, int, int]] = set()
        windows = _text_windows(record["text"])
        for window_text, base_offset in windows:
            if spec["backend"] == "gliner":
                result = model.predict_entities(window_text, spec["labels"], threshold=args.threshold)
            else:
                try:
                    result = model.extract_entities(window_text, spec["labels"], include_spans=True, include_confidence=True)
                except TypeError:
                    result = model.extract_entities(window_text, spec["labels"])
            for item in _extract_entity_items(result):
                if not isinstance(item.get("start"), int) or not isinstance(item.get("end"), int):
                    invalid_span_count += 1
                    continue
                if not 0 <= item["start"] < item["end"] <= len(window_text):
                    invalid_span_count += 1
                    continue
                label = _mapped_label(str(item.get("label", item.get("type", ""))), spec["label_map"])
                if label is not None:
                    start, end = base_offset + item["start"], base_offset + item["end"]
                    if item.get("text") is not None and window_text[item["start"] : item["end"]] != item["text"]:
                        invalid_span_count += 1
                        continue
                    key = (label, start, end)
                    if key not in seen:
                        seen.add(key)
                        entities.append({"label": label, "start": start, "end": end})
        predictions.append({"hash": record["hash"], "entities": entities})
    if invalid_span_count:
        print(f"Dropped invalid GLiNER spans: {invalid_span_count}", flush=True)
    return predictions, invalid_span_count


def run(args: SimpleNamespace) -> Path:
    spec = _model_spec(args.model, args.config)
    device = resolve_device(args.device)
    print(f"Model: {spec['name']} ({spec['model_id']})", flush=True)
    print(f"Backend: {spec['backend']}; device: {device}", flush=True)
    records = read_records(args.input, require_entities=False, limit=args.max_records)
    started = time.perf_counter()
    if spec["backend"] == "token-classification":
        predictions = _run_token_classification(records, spec, args, device)
        invalid_span_count = 0
    else:
        predictions, invalid_span_count = _run_gliner(records, spec, args, device)
    _validate_predictions(predictions, records)
    _write_jsonl(args.output, predictions)
    elapsed = time.perf_counter() - started
    metadata = {"model": spec["name"], "model_id": spec["model_id"], "backend": spec["backend"], "input": str(args.input), "records": len(records), "device": str(device), "elapsed_seconds": elapsed, "records_per_second": len(records) / elapsed if elapsed else None, "invalid_spans_dropped": invalid_span_count}
    if device.type == "cuda":
        metadata["peak_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
    args.output.with_name("metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Predictions: {args.output}")
    print(f"Metadata: {args.output.with_name('metadata.json')}")
    return args.output


@click.command(help="Run one shortlisted ready-to-use NER checkpoint locally.")
@click.option("--model", required=True, help="Name from model_config.json")
@click.option("--config", type=click.Path(path_type=Path), default=CONFIG_PATH)
@click.option("--input", type=click.Path(path_type=Path), default=ROOT / "data/dev.jsonl")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--max-length", type=int, default=256, show_default=True)
@click.option("--stride", type=int, default=64, show_default=True)
@click.option("--max-records", type=int)
@click.option("--threshold", type=float, default=0.5, show_default=True)
@click.option("--device", type=click.Choice(("auto", "cpu", "cuda")), default="auto", show_default=True)
def main(**options: Any) -> None:
    try:
        run(SimpleNamespace(**options))
    except (OSError, RuntimeError, TypeError, ValueError, ImportError, json.JSONDecodeError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
