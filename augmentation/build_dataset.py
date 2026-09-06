"""CLI: build a train.jsonl augmented with transliterated duplicates.

For every record written purely in one script (Cyrillic-only or Latin-only),
adds one extra record with the text transliterated into the other script and
entity offsets remapped exactly (see augmentation/transliteration.py).
Records mixing both scripts, or neither, are kept exactly once, unmodified.

dev.jsonl is never touched -- only data used for training is augmented, dev
stays the original distribution the final model is judged against.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click

from augmentation.transliteration import (
    TransliterationSkipped,
    classify_script,
    transliterate_record,
)

JsonObject = dict[str, Any]

_DIRECTION_BY_SCRIPT = {"cyrillic": "cyr2lat", "latin": "lat2cyr"}
_SUFFIX_BY_SCRIPT = {"cyrillic": "translit-lat", "latin": "translit-cyr"}


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
    """Reads --input, appends transliterated duplicates, writes --output."""

    input_path = args.input.expanduser().resolve()
    records = _read_jsonl(input_path)

    stats: JsonObject = {
        "input_records": len(records),
        "by_script": {"cyrillic": 0, "latin": 0, "mixed": 0, "other": 0},
        "added_translit_lat": 0,
        "added_translit_cyr": 0,
        "skipped_unsafe_digraph_boundary": 0,
    }

    output: list[JsonObject] = list(records)
    seen_hashes = {record["hash"] for record in records}
    for record in records:
        script = classify_script(record["text"])
        stats["by_script"][script] += 1
        direction = _DIRECTION_BY_SCRIPT.get(script)
        if direction is None:
            continue
        try:
            new_text, new_entities = transliterate_record(
                record["text"], record["entities"], direction
            )
        except TransliterationSkipped:
            stats["skipped_unsafe_digraph_boundary"] += 1
            continue

        new_hash = f"{record['hash']}::{_SUFFIX_BY_SCRIPT[script]}"
        if new_hash in seen_hashes:
            raise ValueError(f"generated hash collision: {new_hash}")
        seen_hashes.add(new_hash)
        output.append({"hash": new_hash, "text": new_text, "entities": new_entities})
        stats[f"added_{_SUFFIX_BY_SCRIPT[script].replace('-', '_')}"] += 1

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
    help="Augment a train.jsonl with Cyrillic<->Latin transliterated duplicates.",
    context_settings={"show_default": True},
)
@click.option("--input", type=click.Path(path_type=Path), default=Path("data/train.jsonl"))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/train_translit.jsonl"),
)
@click.option(
    "--stats",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/transliteration_stats.json"),
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
