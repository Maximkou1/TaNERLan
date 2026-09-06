"""CLI: drop noisy-formatting records from train.jsonl and see if it helps.

A separate, one-off experiment on the plain baseline train set -- not part of
the augmentation/ ladder (which only ever *adds* records). Investigated
data/train.jsonl for concrete "dirty data" candidates first: structural
issues (duplicate hashes, invalid offsets, overlapping/duplicate entities)
are already impossible, since baseline/common.py's loader rejects them.
Semantic issues that mattered for the external corpora in augmentation/ (bare
generic nouns tagged as entities, systematic mislabeling) also weren't
present here -- lowercase-initial entities, single-character entities, and
high entity-density records all turned out to be legitimate short mentions
(hashtags, nicknames, name lists) on inspection, not annotation errors, so
none of those are filtered.

What *is* present, and is what this script removes, is stylistic noise from
the dataset's social-media origin -- three criteria, chosen after auditing
counts and overlap:

- ALL-CAPS records: >95% of alphabetic characters uppercase (min 10 letters).
  Spam/blessing-style posts ("MASHALLOH OFARIN KUZ TEGMASIN...").
- Stretched letters: 4+ identical characters in a row anywhere in the text
  (elongated words for emphasis, or decorative character runs).
- Very short records: under 20 characters -- too little context for the
  model to learn from either way.

A record is dropped if it matches *any* of the three (union, not
intersection). This does remove some records that still carry gold entities
(173 of the 400 short ones, for example) -- a real cost, weighed against the
hope that dropping noisy formatting improves what the model learns from the
rest. That's exactly the question this experiment is testing, not something
assumed going in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click

JsonObject = dict[str, Any]

_STRETCH = re.compile(r"(.)\1{3,}", re.UNICODE)
_MIN_LETTERS_FOR_CAPS_CHECK = 10
_CAPS_RATIO_THRESHOLD = 0.95
_SHORT_TEXT_THRESHOLD = 20


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < _MIN_LETTERS_FOR_CAPS_CHECK:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def _dirty_reasons(record: JsonObject) -> list[str]:
    """Returns every matching reason (a record can match more than one)."""

    text = record["text"]
    reasons = []
    if _caps_ratio(text) > _CAPS_RATIO_THRESHOLD:
        reasons.append("all_caps")
    if _STRETCH.search(text):
        reasons.append("stretched_letters")
    if len(text) < _SHORT_TEXT_THRESHOLD:
        reasons.append("short_text")
    return reasons


def _read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def run(args: SimpleNamespace) -> JsonObject:
    """Reads --input, drops records matching any dirty-data reason, writes --output."""

    input_path = args.input.expanduser().resolve()
    records = _read_jsonl(input_path)

    stats: JsonObject = {
        "input_records": len(records),
        "by_reason": {"all_caps": 0, "stretched_letters": 0, "short_text": 0},
        "dropped_records": 0,
        "dropped_entities_by_label": {"ORG": 0, "NAME": 0, "GEO": 0},
    }
    kept: list[JsonObject] = []
    for record in records:
        reasons = _dirty_reasons(record)
        if not reasons:
            kept.append(record)
            continue
        for reason in reasons:
            stats["by_reason"][reason] += 1
        stats["dropped_records"] += 1
        for entity in record["entities"]:
            stats["dropped_entities_by_label"][entity["label"]] += 1

    stats["output_records"] = len(kept)
    output_path = args.output.expanduser().resolve()
    _write_jsonl(output_path, kept)
    print(f"Input: {input_path} ({len(records)} records)")
    print(f"Output: {output_path} ({len(kept)} records)")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.stats is not None:
        stats_path = args.stats.expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Stats: {stats_path}")
    return stats


@click.command(
    help="Drop ALL-CAPS / stretched-letter / very-short records from a train.jsonl.",
    context_settings={"show_default": True},
)
@click.option("--input", type=click.Path(path_type=Path), default=Path("data/train.jsonl"))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("cleaning/data/train_clean.jsonl"),
)
@click.option(
    "--stats",
    type=click.Path(path_type=Path),
    default=Path("cleaning/data/filter_stats.json"),
)
def main(**options: Any) -> None:
    """Runs the CLI with a compact error message."""

    try:
        run(SimpleNamespace(**options))
    except (OSError, TypeError, ValueError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
