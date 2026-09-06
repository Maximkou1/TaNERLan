"""CLI: apply apostrophe/diacritic normalization to train.jsonl, in place.

Isolates the effect tested jointly in step 1 (transliteration.py already runs
this same normalization internally before script-flipping, so its result
never showed the effect of normalization on its own, or on the ~2,952 mixed-
script records that transliteration.py leaves untouched). This script applies
``tanerlan.modern_bert.tokenizer.tokenization_utils.prepare_input(text, homoglyphs=False)`` to every
record's text and nothing else: no new records are added, record count is
unchanged, and entity spans are untouched because the normalization is
length-preserving (each apostrophe-like codepoint maps to exactly one
canonical codepoint -- see transliteration.py's docstring for the full
argument).

To build the "transliteration on top of normalization" arm, point
build_dataset.py's --input at this script's --output instead of at
data/train.jsonl -- it needs no code changes, since build_dataset.py already
uses whatever --input gives it both as the pass-through copies and as the
source for transliterate_record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

JsonObject = dict[str, Any]


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
    """Reads --input, normalizes every record's text, writes --output."""

    input_path = args.input.expanduser().resolve()
    records = _read_jsonl(input_path)

    stats: JsonObject = {"input_records": len(records), "changed_records": 0}
    output: list[JsonObject] = []
    for record in records:
        normalized = prepare_input(record["text"], homoglyphs=False)
        if normalized != record["text"]:
            stats["changed_records"] += 1
        new_record = dict(record)
        new_record["text"] = normalized
        output.append(new_record)

    stats["output_records"] = len(output)
    output_path = args.output.expanduser().resolve()
    _write_jsonl(output_path, output)
    print(f"Input: {input_path} ({len(records)} records)")
    print(f"Output: {output_path} ({len(output)} records)")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.stats is not None:
        stats_path = args.stats.expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Stats: {stats_path}")
    return stats


@click.command(
    help="Normalize apostrophe/diacritic variants in a train.jsonl, in place (no new records).",
    context_settings={"show_default": True},
)
@click.option("--input", type=click.Path(path_type=Path), default=Path("data/train.jsonl"))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/train_normalized.jsonl"),
)
@click.option(
    "--stats",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/normalization_stats.json"),
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
