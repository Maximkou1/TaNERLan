"""CLI: augment train.jsonl with real externally-annotated Uzbek NER corpora.

Replaces the first, abandoned attempt at step 2 (`external_corpora.py`, raw
tahrirchi text labeled `entities: []`): that approach measurably hurt dev
recall (see augmentation/README.md) because tagging real book/news text as
"zero entities" is systematic label noise, not a verified negative. This
version instead converts *already human/rule-annotated* Uzbek NER corpora
into our schema, so `entities: []` sentences are genuine negatives (the
annotators found nothing there), not an artifact of skipping annotation.

Two sources are combined:
- `orvile/named-entity-recognition-for-uzbek-language` on Kaggle
  ("courpusNER2015", 11k+ sentences, BIOES-tagged PER/ORG/LOC), fetched
  anonymously via `kagglehub` (no API key needed for this public dataset).
  Tokens are rejoined with single spaces (the source has no punctuation
  tokens to preserve) and BIOES spans decoded into our schema.
- The Mendeley "Dataset of Uzbek language NER (3000+)" gold set
  (7bxcj57xdz, CC BY 4.0, 4,176 sentences, BIO, 8 entity types), supplied
  locally as `augmentation/Uzbek_NER_Gold.tsv` (its own automated download
  returns `AccessDenied` from every unauthenticated request tried -- see
  below -- so this file has to be fetched by hand). BIO spans are decoded
  the same way; MISC/TEMPORAL/NUMERIC/WORK/MONEY tags fall outside our
  ORG/NAME/GEO schema and are left untagged rather than mapped.

Both sources map PER -> NAME, LOC -> GEO, ORG -> ORG.

Two other sources the user pointed at were evaluated and set aside for
this step, documented in augmentation/README.md rather than silently
dropped:
- `risqaliyevds/uzbek_ner` (Hugging Face): entities are given as bare mention
  strings with no offsets, and 93.8% of records list "O'zbekiston" as a GPE
  entity while only 20.6% of records even contain that string in the text --
  strong evidence of templated/hallucinated auto-labeling, not safe to train
  on without a much heavier manual audit than this step warrants.
- The other Mendeley Data dataset (7d59mk8xp5, "3000+" BIOES set): also
  well-described (human-annotated, CC BY 4.0) but its download bucket
  (prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com) returns
  AccessDenied to every unauthenticated request this script tried (direct,
  with Referer/Origin headers, via the dataset API's signed-URL endpoint) --
  it appears to require an authenticated browser session Mendeley's site
  itself holds, not just a public dataset. If you can download that zip by
  hand too, hand me the file and this module can add a loader for it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openpyxl
import rich_click as click

JsonObject = dict[str, Any]

KAGGLE_DATASET = "orvile/named-entity-recognition-for-uzbek-language"
KAGGLE_FILENAME = "courpusNER2015 (11k sentences).xlsx"
LABEL_MAP = {"PER": "NAME", "LOC": "GEO", "ORG": "ORG"}


def _fetch_kaggle_xlsx() -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    matches = list(root.rglob(KAGGLE_FILENAME))
    if not matches:
        raise ValueError(f"{KAGGLE_FILENAME!r} not found under {root}")
    return matches[0]


def _iter_sentences(xlsx_path: Path) -> list[list[tuple[str, str]]]:
    """Groups (word, tag) rows of Sheet1 by their Sentence column."""

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True)
    sheet = workbook["Sheet1"]
    sentences: list[list[tuple[str, str]]] = []
    current_key: object = object()
    current: list[tuple[str, str]] = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        sentence_key, word, tag = row[0], row[1], row[2]
        if sentence_key is None or word is None:
            continue
        if sentence_key != current_key:
            if current:
                sentences.append(current)
            current = []
            current_key = sentence_key
        # A UTF-8 apostrophe re-decoded as CP1251 and re-saved shows up as the
        # 3-byte sequence "вЂ™" in one recurring source phrase ("Xalq
        # ta'limi vazirligi"); undo it before offsets are computed.
        clean_word = str(word).replace("вЂ™", "ʼ")
        current.append((clean_word, str(tag) if tag is not None else "O"))
    if current:
        sentences.append(current)
    return sentences


def _words_to_text_and_offsets(words: list[str]) -> tuple[str, list[int]]:
    """Joins words with single spaces, returning the text and each word's start offset."""

    offsets: list[int] = []
    position = 0
    for index, word in enumerate(words):
        if index > 0:
            position += 1  # single joining space
        offsets.append(position)
        position += len(word)
    return " ".join(words), offsets


def _drop_lowercase_mentions(
    text: str, entities: list[JsonObject]
) -> tuple[list[JsonObject], int]:
    """Drops decoded entities whose mention isn't capitalized.

    Both source corpora's PER/ORG/LOC tags turned out to cover more than our
    ORG/NAME/GEO ("proper noun") definitions -- spot-checking found generic
    common nouns tagged as entities: "tashkilotlarni" (organizations),
    "talabalar" (students), "davlat" (state/government), a bare
    "Prezidentining" (the president's, no actual name) as PER. Every real
    entity in `data/train.jsonl` and LABELING_GUIDE.md's classes is a proper
    noun, which is capitalized in both Uzbek scripts, so entities whose
    mention doesn't start with an uppercase letter are dropped here as an
    (imperfect but cheap and precision-favoring) filter for that mismatch --
    counted and returned as the 2nd element so callers can report it.
    """

    kept: list[JsonObject] = []
    dropped = 0
    for entity in entities:
        mention = text[entity["start"] : entity["end"]]
        if mention[:1].isupper():
            kept.append(entity)
        else:
            dropped += 1
    return kept, dropped


def tokens_to_record(tokens: list[tuple[str, str]]) -> tuple[str, list[JsonObject], int]:
    """Joins BIOES-tagged tokens with single spaces and decodes entity spans.

    Malformed tag sequences (a bare label with no B/I/E/S prefix, a B- with
    no matching E-, ...) are skipped rather than guessed at: the token is
    just left untagged instead of risking a wrong span.
    """

    words = [word for word, _ in tokens]
    text, offsets = _words_to_text_and_offsets(words)

    entities: list[JsonObject] = []
    index = 0
    n = len(tokens)
    while index < n:
        word, tag = tokens[index]
        prefix, _, label = tag.partition("-")
        mapped = LABEL_MAP.get(label)
        if prefix == "S" and mapped:
            start = offsets[index]
            entities.append({"label": mapped, "start": start, "end": start + len(word)})
            index += 1
        elif prefix == "B" and mapped:
            start = offsets[index]
            end = start + len(word)
            cursor = index + 1
            while cursor < n and tokens[cursor][1] == f"I-{label}":
                end = offsets[cursor] + len(tokens[cursor][0])
                cursor += 1
            if cursor < n and tokens[cursor][1] == f"E-{label}":
                end = offsets[cursor] + len(tokens[cursor][0])
                entities.append({"label": mapped, "start": start, "end": end})
                index = cursor + 1
            else:
                index += 1  # no matching E-<label>: drop, do not guess a span
        else:
            index += 1

    kept, dropped_lowercase = _drop_lowercase_mentions(text, entities)
    return text, kept, dropped_lowercase


def bio_tokens_to_record(tokens: list[tuple[str, str]]) -> tuple[str, list[JsonObject], int]:
    """Joins BIO-tagged tokens with single spaces and decodes entity spans.

    Same "don't guess" policy as `tokens_to_record`, adapted for BIO (no
    explicit E-/S- tags): a `B-<label>` starts a span, immediately following
    `I-<label>` tokens extend it, and the span ends at the first token that
    isn't `I-<label>` for that same label. A stray `I-<label>` with no
    preceding `B-<label>` is left untagged rather than guessed at.
    """

    words = [word for word, _ in tokens]
    text, offsets = _words_to_text_and_offsets(words)

    entities: list[JsonObject] = []
    index = 0
    n = len(tokens)
    while index < n:
        word, tag = tokens[index]
        prefix, _, label = tag.partition("-")
        mapped = LABEL_MAP.get(label)
        if prefix == "B" and mapped:
            start = offsets[index]
            end = start + len(word)
            cursor = index + 1
            while cursor < n and tokens[cursor][1] == f"I-{label}":
                end = offsets[cursor] + len(tokens[cursor][0])
                cursor += 1
            entities.append({"label": mapped, "start": start, "end": end})
            index = cursor
        else:
            index += 1

    kept, dropped_lowercase = _drop_lowercase_mentions(text, entities)
    return text, kept, dropped_lowercase


def _iter_gold_tsv_sentences(tsv_path: Path) -> list[list[tuple[str, str]]]:
    """Groups (Token, NER_Tag) rows of the Mendeley Gold TSV by Sentence id.

    Columns: Sentence, TokenOrder, Token, NER_Tag, pos.
    """

    sentences: list[list[tuple[str, str]]] = []
    with tsv_path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter="\t")
        next(reader, None)  # header
        current_key: object = object()
        current: list[tuple[str, str]] = []
        for row in reader:
            if len(row) < 4 or not row[2]:
                continue
            sentence_key, word, tag = row[0], row[2], row[3]
            if sentence_key != current_key:
                if current:
                    sentences.append(current)
                current = []
                current_key = sentence_key
            current.append((word, tag if tag else "O"))
        if current:
            sentences.append(current)
    return sentences


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


def _empty_source_stats(source_sentences: int) -> JsonObject:
    return {
        "source_sentences": source_sentences,
        "added_records": 0,
        "added_entities_by_label": {"NAME": 0, "GEO": 0, "ORG": 0},
        "dropped_lowercase_mentions": 0,
        "sentences_with_zero_entities": 0,
    }


def _add_source(
    output: list[JsonObject],
    sentences: list[list[tuple[str, str]]],
    decode: Any,
    hash_prefix: str,
) -> JsonObject:
    """Decodes every sentence with `decode`, appends non-empty ones to `output`."""

    source_stats = _empty_source_stats(len(sentences))
    for sentence_index, tokens in enumerate(sentences):
        text, entities, dropped = decode(tokens)
        if not text.strip():
            continue
        output.append({"hash": f"{hash_prefix}-{sentence_index:06d}", "text": text, "entities": entities})
        source_stats["added_records"] += 1
        source_stats["dropped_lowercase_mentions"] += dropped
        if not entities:
            source_stats["sentences_with_zero_entities"] += 1
        for entity in entities:
            source_stats["added_entities_by_label"][entity["label"]] += 1
    return source_stats


def run(args: SimpleNamespace) -> JsonObject:
    """Reads --train, appends converted Kaggle + Mendeley Gold sentences, writes --output."""

    train_path = args.train.expanduser().resolve()
    records = _read_jsonl(train_path)
    output = list(records)

    xlsx_path = _fetch_kaggle_xlsx()
    kaggle_sentences = _iter_sentences(xlsx_path)
    kaggle_stats = _add_source(output, kaggle_sentences, tokens_to_record, "kaggle-uzner2015")

    gold_tsv_path = args.gold_tsv.expanduser().resolve() if args.gold_tsv is not None else None
    if gold_tsv_path is not None and gold_tsv_path.exists():
        gold_sentences = _iter_gold_tsv_sentences(gold_tsv_path)
        gold_stats = _add_source(output, gold_sentences, bio_tokens_to_record, "mendeley-gold")
    else:
        gold_sentences = []
        gold_stats = _empty_source_stats(0)
        if gold_tsv_path is not None:
            print(f"Mendeley Gold TSV not found at {gold_tsv_path}, skipping that source.")

    stats: JsonObject = {
        "input_records": len(records),
        "kaggle_uzner2015": kaggle_stats,
        "mendeley_gold": gold_stats,
        "added_records": kaggle_stats["added_records"] + gold_stats["added_records"],
        "output_records": len(output),
    }
    output_path = args.output.expanduser().resolve()
    _write_jsonl(output_path, output)
    print(f"Train: {train_path} ({len(records)} records)")
    print(f"Kaggle source: {xlsx_path} ({len(kaggle_sentences)} sentences)")
    if gold_sentences:
        print(f"Mendeley Gold source: {gold_tsv_path} ({len(gold_sentences)} sentences)")
    print(f"Output: {output_path} ({len(output)} records)")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.stats is not None:
        stats_path = args.stats.expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Stats: {stats_path}")
    return stats


@click.command(
    help="Augment train.jsonl with the Kaggle courpusNER2015 + Mendeley Gold Uzbek NER corpora.",
    context_settings={"show_default": True},
)
@click.option("--train", type=click.Path(path_type=Path), default=Path("data/train.jsonl"))
@click.option(
    "--gold-tsv",
    type=click.Path(path_type=Path),
    default=None,
    help="Mendeley Gold TSV (BIO), e.g. augmentation/Uzbek_NER_Gold.tsv. Omit to use Kaggle only.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/train_external_ner.jsonl"),
)
@click.option(
    "--stats",
    type=click.Path(path_type=Path),
    default=Path("augmentation/data/external_ner_stats.json"),
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
