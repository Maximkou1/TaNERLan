"""Evaluate saved predictions by Latin/Cyrillic script slices."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any

from transformers import AutoTokenizer

from evaluation.core import calculate_metrics, load_gold, read_jsonl, validate_entities

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("model_config.json")
SLICES = ("Latin", "Cyrillic", "mixed")
LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")
CYRILLIC_RE = re.compile(r"[\u0400-\u052f\u2de0-\u2dff\ua640-\ua69f]")


def script_slice(text: str) -> str | None:
    """Classify a document by the scripts present in its text."""

    has_latin = bool(LATIN_RE.search(text))
    has_cyrillic = bool(CYRILLIC_RE.search(text))
    if has_latin and has_cyrillic:
        return "mixed"
    if has_latin:
        return "Latin"
    if has_cyrillic:
        return "Cyrillic"
    return None


def percentile(values: list[int], fraction: float) -> int:
    """Return a nearest-rank percentile without adding a dependency."""

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def load_specs() -> list[dict[str, Any]]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return payload["models"]


def tokenizer_model_id(spec: dict[str, Any]) -> str:
    """Resolve the tokenizer actually used by each backend."""

    # GLiNER checkpoints contain the extractor weights, while their encoder
    # tokenizer is the DeBERTa tokenizer specified in gliner_config.json.
    if spec["backend"] in {"gliner", "gliner2", "gliner2.5"}:
        return "microsoft/mdeberta-v3-base"
    return spec["model_id"]


def load_predictions(path: Path, gold: dict[str, dict[str, Any]]) -> dict[str, set[tuple[str, int, int]]]:
    result = {}
    for index, record in enumerate(read_jsonl(path, "prediction"), start=1):
        record_hash = record["hash"]
        if record_hash not in gold:
            raise ValueError(f"{path}:{index}: unknown hash {record_hash}")
        result[record_hash] = validate_entities(
            record.get("entities"),
            len(gold[record_hash]["text"]),
            f"{path}:{index}",
        )
    if set(result) != set(gold):
        raise ValueError(f"{path}: predictions do not cover exactly the gold hashes")
    return result


def token_stats(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    lengths: list[int] = []
    unk_total = 0
    unk_docs = 0
    unk_id = tokenizer.unk_token_id
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        lengths.append(len(ids))
        token_unks = sum(
            int(token_id == unk_id) if unk_id is not None else 0
            for token_id in ids
        )
        if unk_id is None:
            tokens = tokenizer.convert_ids_to_tokens(ids)
            token_unks = sum(token in {"[UNK]", "<unk>", "<UNK>"} for token in tokens)
        unk_total += token_unks
        unk_docs += int(token_unks > 0)
    return {
        "avg_tokens": sum(lengths) / len(lengths),
        "median_tokens": median(lengths),
        "p95_tokens": percentile(lengths, 0.95),
        "max_tokens": max(lengths),
        "unk_total": unk_total,
        "unk_docs": unk_docs,
        "unk_rate_tokens": unk_total / sum(lengths) if sum(lengths) else 0.0,
    }


def main() -> None:
    gold_records, gold = load_gold(ROOT / "data" / "dev.jsonl")
    slices = {record["hash"]: script_slice(record["text"]) for record in gold_records}
    result: dict[str, Any] = {
        "definition": {
            "Latin": "contains Latin letters and no Cyrillic letters",
            "Cyrillic": "contains Cyrillic letters and no Latin letters",
            "mixed": "contains both Latin and Cyrillic letters",
            "token_lengths": "includes special tokens; no truncation",
            "matching": "exact label/start/end; micro-F1",
        },
        "documents": {
            slice_name: sum(value == slice_name for value in slices.values())
            for slice_name in SLICES
        },
        "models": {},
    }

    for spec in load_specs():
        name = spec["name"]
        prediction_path = ROOT / "artifacts" / "models" / name / "dev_predictions.jsonl"
        predictions = load_predictions(prediction_path, gold)
        tokenizer_id = tokenizer_model_id(spec)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, use_fast=True, local_files_only=True
        )
        model_rows: dict[str, Any] = {}
        for slice_name in SLICES:
            hashes = [record_hash for record_hash, value in slices.items() if value == slice_name]
            subset_gold = {record_hash: gold[record_hash] for record_hash in hashes}
            subset_predictions = {record_hash: predictions[record_hash] for record_hash in hashes}
            metrics = calculate_metrics(subset_gold, subset_predictions)
            texts = [gold[record_hash]["text"] for record_hash in hashes]
            model_rows[slice_name] = {
                "records": len(hashes),
                "metrics": metrics,
                "tokenizer": tokenizer_id,
                "tokenization": token_stats(tokenizer, texts),
            }
        result["models"][name] = model_rows

    output = ROOT / "artifacts" / "script_slice_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Slice summary: {output}")
    for name, rows in result["models"].items():
        line = " ".join(
            f"{slice_name}: F1={rows[slice_name]['metrics']['micro']['f1']:.3f}, "
            f"UNK={rows[slice_name]['tokenization']['unk_total']}"
            for slice_name in SLICES
        )
        print(f"{name}: {line}")


if __name__ == "__main__":
    main()
