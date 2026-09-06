"""CLI: augment train.jsonl with unlabeled tahrirchi corpus text (step 2).

Adds paragraph/sentence-scale chunks from `tahrirchi/uz-books-v2` (fiction,
Cyrillic + Latin) and `tahrirchi/uz-crawl` (news + Telegram posts) as extra
train records with `entities: []`. No transliteration is applied (step 1's
concern); text is used exactly as published, only split into chunks.

Important caveat, kept here rather than buried in a docstring elsewhere:
these corpora have no NER annotations. Adding a chunk with `entities: []`
tells the model "there are zero ORG/NAME/GEO mentions in this text", which is
almost certainly false for running book and news text -- it is real label
noise, not verified negatives. This step exists specifically to measure
whether that trade-off (more diverse language modeling signal for the
tokenizer/encoder vs. systematic false-negative pressure on recall) nets out
positive or negative on dev metrics; see augmentation/README.md for the
result.

Reproducing the input: this script fetches exactly one shard per split via
`huggingface_hub.hf_hub_download` (pinned filenames below), not the full
corpus -- uz-books-v2 alone is ~34B tokens across 43 Cyrillic + 26 Latin
shards, far more than a single experiment needs. One shard per split already
gives 892-368,017 source documents to sample from, more than enough for the
few thousand chunks this step adds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
import rich_click as click
from huggingface_hub import hf_hub_download

JsonObject = dict[str, Any]

# name -> (repo_id, filename). One shard per split, pinned so re-running this
# script is deterministic. See the module docstring for why not the full corpus.
SOURCES: dict[str, tuple[str, str]] = {
    "books-cyr": ("tahrirchi/uz-books-v2", "data/cyr-00000-of-00043.parquet"),
    "books-lat": ("tahrirchi/uz-books-v2", "data/lat-00000-of-00026.parquet"),
    "crawl-news": ("tahrirchi/uz-crawl", "data/news-00000-of-00007.parquet"),
    "crawl-telegram": ("tahrirchi/uz-crawl", "data/telegram_blogs-00000-of-00001.parquet"),
}

_BLANK_LINES = re.compile(r"\n\s*\n+")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _fetch_texts(repo_id: str, filename: str) -> list[str]:
    """Downloads (and caches) one parquet shard, returns its text column."""

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    table = pq.read_table(local_path, columns=["text"])
    return [str(value) for value in table.column("text").to_pylist() if value]


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Splits one paragraph into sentence-bounded pieces under ``max_chars``."""

    sentences = _SENTENCE_END.split(paragraph)
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) > max_chars and buffer:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    # A single sentence can still exceed max_chars; hard-cut as a last resort.
    return [piece[:max_chars] for piece in pieces]


def chunk_text(text: str, *, min_chars: int, max_chars: int) -> list[str]:
    """Splits a raw document into paragraph-scale chunks in ``[min, max]`` chars.

    Mirrors the length distribution of ``data/train.jsonl`` (median 151,
    p90 1091 chars) instead of using whole-book or whole-article documents.
    Consecutive short paragraphs are greedily merged up to ``max_chars``;
    paragraphs longer than ``max_chars`` are split at sentence boundaries.
    """

    paragraphs = [p.strip() for p in _BLANK_LINES.split(text.replace("\r\n", "\n"))]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split())  # collapse internal whitespace/newlines
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_long_paragraph(paragraph, max_chars))
            continue
        candidate = f"{buffer} {paragraph}".strip() if buffer else paragraph
        if len(candidate) > max_chars and buffer:
            chunks.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if len(c) >= min_chars]


def _sample_chunks(
    texts: list[str],
    *,
    count: int,
    min_chars: int,
    max_chars: int,
    seed: int,
) -> list[str]:
    """Chunks documents in a deterministic shuffled order until ``count`` is met."""

    import random

    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    chunks: list[str] = []
    seen: set[str] = set()
    for index in order:
        for chunk in chunk_text(texts[index], min_chars=min_chars, max_chars=max_chars):
            if chunk in seen:
                continue
            seen.add(chunk)
            chunks.append(chunk)
            if len(chunks) >= count:
                return chunks
    return chunks


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
    """Builds train + unlabeled-corpus-chunk records and writes them out."""

    train_path = args.train.expanduser().resolve()
    records = _read_jsonl(train_path)
    output = list(records)

    stats: JsonObject = {"input_records": len(records), "by_source": {}}
    for name, (repo_id, filename) in SOURCES.items():
        texts = _fetch_texts(repo_id, filename)
        chunks = _sample_chunks(
            texts,
            count=args.per_source,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            seed=args.seed,
        )
        stats["by_source"][name] = {"source_documents": len(texts), "chunks_added": len(chunks)}
        for index, chunk in enumerate(chunks):
            output.append({"hash": f"tahrirchi-{name}-{index:06d}", "text": chunk, "entities": []})

    stats["output_records"] = len(output)
    output_path = args.output.expanduser().resolve()
    _write_jsonl(output_path, output)
    print(f"Train: {train_path} ({len(records)} records)")
    print(f"Output: {output_path} ({len(output)} records)")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.stats is not None:
        stats_path = args.stats.expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Stats: {stats_path}")
    return stats


@click.command(
    help="Augment train.jsonl with unlabeled tahrirchi/uz-books-v2 + uz-crawl chunks.",
    context_settings={"show_default": True},
)
@click.option("--train", type=click.Path(path_type=Path), default=Path("data/train.jsonl"))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/train_external.jsonl"),
)
@click.option(
    "--stats",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/external_corpora_stats.json"),
)
@click.option("--per-source", type=int, default=1500, help="Chunks added per corpus source.")
@click.option("--min-chars", type=int, default=25)
@click.option("--max-chars", type=int, default=1200)
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
