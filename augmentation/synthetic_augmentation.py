"""CLI: augment train.jsonl with entity-safe synthetic typo noise (step 3).

For every record, adds one noised duplicate with light character-level
corruption -- keyboard-adjacent substitution, deletion, duplication and
adjacent transposition -- entity spans stay exactly aligned because noise
outside a gold span is applied independently per character and the new text
is reassembled through the same offset-remapping technique used in
transliteration.py (a per-character output "piece" list + prefix-sum
offsets), and because deletion/duplication (the only length-changing ops)
are restricted to characters that fall outside every gold entity span.
Inside a span only same-length substitution is allowed, at a lower default
rate, so a span's start/end never has to move and its content is only
occasionally lightly misspelled (useful on its own -- it teaches the model
to still tag a slightly misspelled "Тoshkent"), never restructured.

The corruption operation set (substitute / delete / duplicate / transpose,
each applied independently at a small per-character probability) is the
standard rule-based typo-simulation approach also used by SAGE
(https://github.com/Pomelkin/sage, itself following ai-forever/sage). SAGE's
higher-fidelity corruptor (`SBSCConfig`) is a statistical model of real
spelling-error frequencies, but it is trained on Russian error corpora and
has no Uzbek equivalent to fall back on; pulling in the package would still
mean writing this same rule-based path, so it is implemented directly here
instead of adding the dependency.

No transliteration and no external corpora are involved in this step --
only `data/train.jsonl`, so its effect on dev metrics is attributable to
this technique alone.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click

JsonObject = dict[str, Any]

# Same-row QWERTY neighbors (Uzbek Latin keyboard) and same-row Cyrillic
# JCUKEN neighbors (Uzbek/Russian Cyrillic keyboard). Uzbek-specific letters
# (o'/g' as typed, ў/қ/ғ/ҳ) are not on a standard row position and are left
# out on purpose -- they fall back to a no-op substitution.
_QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_JCUKEN_ROWS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]


def _row_adjacency(rows: list[str]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for row in rows:
        for index, ch in enumerate(row):
            neighbors = [row[index - 1]] if index > 0 else []
            if index + 1 < len(row):
                neighbors.append(row[index + 1])
            adjacency[ch] = neighbors
    return adjacency


_ADJACENCY = {**_row_adjacency(_QWERTY_ROWS), **_row_adjacency(_JCUKEN_ROWS)}


def _keyboard_neighbor(ch: str, rng: random.Random) -> str:
    """Returns a same-row keyboard neighbor of ``ch``, or ``ch`` if unknown."""

    neighbors = _ADJACENCY.get(ch.lower())
    if not neighbors:
        return ch
    replacement = rng.choice(neighbors)
    return replacement.upper() if ch.isupper() else replacement


def _protected_mask(text: str, entities: list[JsonObject]) -> list[bool]:
    mask = [False] * len(text)
    for entity in entities:
        for i in range(entity["start"], entity["end"]):
            mask[i] = True
    return mask


def augment_text(
    text: str,
    entities: list[JsonObject],
    rng: random.Random,
    *,
    free_prob: float,
    entity_prob: float,
    swap_prob: float,
) -> tuple[str, list[JsonObject]]:
    """Returns noised text and exactly-remapped entities.

    ``free_prob`` gates substitute/delete/duplicate/transpose outside gold
    spans; ``entity_prob`` gates same-length substitution inside them.
    """

    protected = _protected_mask(text, entities)
    chars = list(text)

    # Adjacent transposition first, free positions only, same length so it
    # cannot disturb offsets computed afterwards.
    index = 0
    while index < len(chars) - 1:
        if (
            not protected[index]
            and not protected[index + 1]
            and chars[index].isalpha()
            and chars[index + 1].isalpha()
            and rng.random() < swap_prob
        ):
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
            index += 2  # do not chain-swap the same character twice
            continue
        index += 1

    pieces: list[str] = [""] * len(chars)
    for i, ch in enumerate(chars):
        if protected[i]:
            pieces[i] = _keyboard_neighbor(ch, rng) if ch.isalpha() and rng.random() < entity_prob else ch
            continue
        if not ch.isalpha() or rng.random() >= free_prob:
            pieces[i] = ch
            continue
        op = rng.choice(("substitute", "delete", "duplicate"))
        if op == "substitute":
            pieces[i] = _keyboard_neighbor(ch, rng)
        elif op == "delete":
            pieces[i] = ""
        else:
            pieces[i] = ch + ch

    offsets = [0] * (len(pieces) + 1)
    for i, piece in enumerate(pieces):
        offsets[i + 1] = offsets[i] + len(piece)

    new_entities = [
        {"label": e["label"], "start": offsets[e["start"]], "end": offsets[e["end"]]} for e in entities
    ]
    return "".join(pieces), new_entities


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
    """Reads --input, appends one noised duplicate per record, writes --output."""

    input_path = args.input.expanduser().resolve()
    records = _read_jsonl(input_path)
    output = list(records)
    seen_hashes = {record["hash"] for record in records}

    for record in records:
        for copy_index in range(args.copies_per_record):
            rng = random.Random(f"{args.seed}:{record['hash']}:{copy_index}")
            new_text, new_entities = augment_text(
                record["text"],
                record["entities"],
                rng,
                free_prob=args.free_prob,
                entity_prob=args.entity_prob,
                swap_prob=args.swap_prob,
            )
            new_hash = f"{record['hash']}::aug-{copy_index}"
            if new_hash in seen_hashes:
                raise ValueError(f"generated hash collision: {new_hash}")
            seen_hashes.add(new_hash)
            output.append({"hash": new_hash, "text": new_text, "entities": new_entities})

    stats = {
        "input_records": len(records),
        "copies_per_record": args.copies_per_record,
        "output_records": len(output),
    }
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
    help="Augment train.jsonl with entity-safe synthetic typo noise.",
    context_settings={"show_default": True},
)
@click.option("--input", type=click.Path(path_type=Path), default=Path("data/train.jsonl"))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/train_synthetic.jsonl"),
)
@click.option(
    "--stats",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/synthetic_augmentation_stats.json"),
)
@click.option("--copies-per-record", type=int, default=1)
@click.option("--free-prob", type=float, default=0.06, help="Per-char noise rate outside entities.")
@click.option("--entity-prob", type=float, default=0.02, help="Per-char substitution rate inside entities.")
@click.option("--swap-prob", type=float, default=0.02, help="Per-position adjacent-transposition rate.")
@click.option("--seed", type=int, default=42)
def main(**options: Any) -> None:
    """Runs the CLI with a compact error message."""

    try:
        run(SimpleNamespace(**options))
    except (OSError, TypeError, ValueError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
