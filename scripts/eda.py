from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import rich_click as click

JsonObject = dict[str, Any]

LATIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CYRILLIC_RANGE = (0x0400, 0x04FF)


def read_jsonl(path: Path) -> list[JsonObject]:
    """Читает JSONL-файл, пропуская пустые строки."""

    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def char_category(char: str) -> str:
    """Группирует символ в укрупнённую категорию для отчёта."""

    if char.isspace():
        return "whitespace"
    if "A" <= char <= "Z" or "a" <= char <= "z":
        return "latin"
    if CYRILLIC_RANGE[0] <= ord(char) <= CYRILLIC_RANGE[1]:
        return "cyrillic"
    if char.isdigit():
        return "digit"
    category = unicodedata.category(char)
    if category.startswith("P"):
        return "punctuation"
    if category.startswith("S"):
        return "symbol"
    return f"other ({category})"


def has_latin(text: str) -> bool:
    return any(char in LATIN_LETTERS for char in text)


def has_cyrillic(text: str) -> bool:
    return any(CYRILLIC_RANGE[0] <= ord(char) <= CYRILLIC_RANGE[1] for char in text)


def report_characters(records: list[JsonObject], top_chars: int) -> None:
    """1. Какие символы встречаются в датасете и как часто."""

    char_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for record in records:
        for char in record["text"]:
            char_counts[char] += 1
            category_counts[char_category(char)] += 1

    print("== 1. Символы ==")
    print(f"Уникальных символов: {len(char_counts)}")
    print("По категориям:")
    for category, count in category_counts.most_common():
        print(f"  {category:>20}: {count}")

    print(f"Топ-{top_chars} символов по частоте:")
    for char, count in char_counts.most_common(top_chars):
        printable = char if not char.isspace() else repr(char)
        print(f"  {printable!r:>8} (U+{ord(char):04X}, {char_category(char)}): {count}")
    print()


def report_labels(records: list[JsonObject]) -> None:
    """2. Как представлены разные теги (метки сущностей)."""

    label_counts: Counter[str] = Counter()
    label_lengths: dict[str, list[int]] = {}
    entities_per_doc: Counter[int] = Counter()
    docs_without_entities = 0

    for record in records:
        entities = record["entities"]
        entities_per_doc[len(entities)] += 1
        if not entities:
            docs_without_entities += 1
        for entity in entities:
            label = entity["label"]
            label_counts[label] += 1
            span_len = entity["end"] - entity["start"]
            label_lengths.setdefault(label, []).append(span_len)

    total_docs = len(records)
    total_entities = sum(label_counts.values())

    print("== 2. Теги ==")
    print(f"Документов: {total_docs}, документов без сущностей: {docs_without_entities}")
    print(f"Всего сущностей: {total_entities}")
    for label, count in label_counts.most_common():
        lengths = label_lengths[label]
        avg_len = sum(lengths) / len(lengths)
        share = count / total_entities * 100
        print(
            f"  {label:>6}: {count:>6} ({share:5.1f}%), "
            f"длина спана: мин={min(lengths)}, макс={max(lengths)}, средн={avg_len:.1f}"
        )

    print("Сущностей на документ (топ-10 по частоте):")
    for entity_count, doc_count in sorted(entities_per_doc.items(), key=lambda item: -item[1])[:10]:
        print(f"  {entity_count} сущностей: {doc_count} документов")
    print()


def report_script_mixing(records: list[JsonObject]) -> None:
    """3. Как часто в одном примере сочетаются кириллица и латиница."""

    only_cyrillic = only_latin = both = neither = 0
    for record in records:
        text = record["text"]
        cyr = has_cyrillic(text)
        lat = has_latin(text)
        if cyr and lat:
            both += 1
        elif cyr:
            only_cyrillic += 1
        elif lat:
            only_latin += 1
        else:
            neither += 1

    total = len(records)
    print("== 3. Смешение кириллицы и латиницы ==")
    print(f"Всего документов: {total}")
    print(f"  только кириллица : {only_cyrillic:>6} ({only_cyrillic / total * 100:5.1f}%)")
    print(f"  только латиница  : {only_latin:>6} ({only_latin / total * 100:5.1f}%)")
    print(f"  и то, и другое   : {both:>6} ({both / total * 100:5.1f}%)")
    print(f"  ни то, ни другое : {neither:>6} ({neither / total * 100:5.1f}%)")
    print()


@click.command(
    help="EDA for the Uzbek NER dataset.",
    context_settings={"show_default": True},
)
@click.argument("paths", nargs=-1, required=True, type=click.Path(path_type=Path))
@click.option("--top-chars", type=int, default=40, help="Сколько самых частых символов показывать.")
def main(paths: tuple[Path, ...], top_chars: int) -> None:
    """Печатает EDA-отчёт по одному или нескольким JSONL-файлам (train/dev)."""

    for path in paths:
        records = read_jsonl(path)
        print(f"##### {path} ({len(records)} записей) #####\n")
        report_characters(records, top_chars)
        report_labels(records)
        report_script_mixing(records)


if __name__ == "__main__":
    main()
