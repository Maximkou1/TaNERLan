"""Собирает пул упоминаний для аугментации заменой сущностей из внешних корпусов.

    python -m tanerlan.modern_bert.ner.data.build_mention_pool --output data/mention_pool.jsonl

Результат — jsonl со строками {"label", "stem", "n_words", "script", "lang", "source"};
путь указывается в yaml как data.augmentation.mention_pool_path. Пул собирается
из строк, которых нет в train, по языкам и письменностям (см. data/mention_pool.py
и data/language.py):

  uz  Mendeley "Uzbek NER Gold" (augmentation/Uzbek_NER_Gold.tsv, BIO, 4.2k предложений,
      скачивается только вручную — сайт отдаёт AccessDenied без браузерной сессии);
      Kaggle courpusNER2015 (augmentation/courpusNER2015.tsv, BIOES, 11k предложений;
      если файла нет — kagglehub); WikiANN uz (HF unimelb-nlp/wikiann, 3k предложений).
      Все три — латиница; основы копируются в кириллицу транслитом.
  ru  WikiNEuRal ru (HF Babelscape/wikineural, 92k предложений): только кириллица,
      только именительный падеж (pymorphy3: "Венеции", "Иванову" отбрасываются),
      основа встречается не реже --silver-min-count раз (разметка автоматическая);
      копии в латинице транслитом (Moskva, Ivanov).
  en  WikiNEuRal en: латиница, не реже --silver-min-count раз.

train.jsonl нужен как справочник (известные основы для отщепления окончаний,
нарицательные слова), его упоминания в пул не попадают.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import rich_click as click
from rich.console import Console
from rich.table import Table

from augmentation.transliteration import classify_script
from tanerlan.modern_bert.ner.data.mention_pool import (
    MentionPool,
    build_mention_pool,
    read_gold_tsv,
    tagged_sentences_to_records,
    write_pool_jsonl,
)
from tanerlan.modern_bert.ner.data.records import Entity, Record, read_records

console = Console()

KAGGLE_DATASET = "orvile/named-entity-recognition-for-uzbek-language"
WIKINEURAL_TAGS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
_PARENTHETICAL = re.compile(r"\s*\(.*$")
_HAS_LETTER = re.compile(r"[^\W\d_]")


# --- источники ---------------------------------------------------------------------


def read_kaggle_tsv(path: Path) -> list[Record]:
    """Kaggle courpusNER2015: колонки Sentence, Word, BIOES-Tag, перевод; "вЂ™" — битый апостроф."""
    sentences: list[list[tuple[str, str]]] = []
    with path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter="\t")
        next(reader, None)
        current_key: object = object()
        current: list[tuple[str, str]] = []
        for row in reader:
            if len(row) < 3 or not row[1]:
                continue
            if row[0] != current_key:
                if current:
                    sentences.append(current)
                current, current_key = [], row[0]
            current.append((row[1].replace("вЂ™", "ʼ"), row[2] or "O"))
        if current:
            sentences.append(current)
    return tagged_sentences_to_records(sentences, "kaggle")


def fetch_kaggle_tsv(target: Path) -> Path:
    """Скачивает xlsx через kagglehub (публичный датасет, без ключа) и сохраняет tsv рядом."""
    import kagglehub
    import openpyxl

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    matches = list(root.rglob("*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"xlsx not found under {root}")
    sheet = openpyxl.load_workbook(matches[0], read_only=True)["Sheet1"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        stream.write("Sentence\tWord\tBIOES-Tag\tEnglish-version (translation)\t\n")
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if row[0] is None or row[1] is None:
                continue
            stream.write(f"{row[0]}\t{row[1]}\t{row[2] or 'O'}\t{row[3] or ''}\t\n")
    return target


def read_hf_tagged(
    dataset: str,
    config: str | None,
    splits: tuple[str, ...],
    tag_names: list[str] | None,
) -> list[Record]:
    """WikiANN / WikiNEuRal: tokens + ner_tags -> записи; у WikiANN в упоминаниях бывают
    уточнения в скобках ("Martin ( Georgia )"), они отрезаются."""
    from datasets import load_dataset

    records: list[Record] = []
    for split in splits:
        data = load_dataset(dataset, config, split=split)
        names = tag_names or data.features["ner_tags"].feature.names
        sentences = [
            [
                (token, names[tag])
                for token, tag in zip(row["tokens"], row["ner_tags"], strict=True)
            ]
            for row in data
        ]
        for record in tagged_sentences_to_records(
            sentences, f"{dataset.split('/')[-1]}-{config or ''}-{split}"
        ):
            text = record["text"]
            entities: list[Entity] = []
            for entity in record["entities"]:
                surface = text[entity["start"] : entity["end"]]
                trimmed = _PARENTHETICAL.sub("", surface).rstrip(" .")
                if trimmed and "(" not in trimmed and ")" not in trimmed:
                    entities.append({**entity, "end": entity["start"] + len(trimmed)})
            records.append({**record, "entities": entities})
    return records


def keep_nominative(records: list[Record]) -> tuple[list[Record], int]:
    """Оставляет русские упоминания, у которых каждое слово может быть именительным
    падежом единственного числа (или аббревиатурой): "Москва", "Иванов", но не "Москве"."""
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    cache: dict[str, bool] = {}

    def nominative(word: str) -> bool:
        if word not in cache:
            if not _HAS_LETTER.search(word) or (word.isupper() and len(word) <= 6):
                cache[word] = True
            else:
                cache[word] = any(
                    {"nomn", "sing"} <= set(p.tag.grammemes) and p.score >= 0.05
                    for p in morph.parse(word)
                )
        return cache[word]

    kept: list[Record] = []
    dropped = 0
    for record in records:
        text = record["text"]
        entities: list[Entity] = []
        for entity in record["entities"]:
            surface = text[entity["start"] : entity["end"]]
            if classify_script(surface) == "cyrillic" and all(
                nominative(w) for w in re.split(r"[\s\-]+", surface)
            ):
                entities.append(entity)
            else:
                dropped += 1
        kept.append({**record, "entities": entities})
    return kept, dropped


# --- сборка -------------------------------------------------------------------------


def entity_count(records: list[Record]) -> int:
    return sum(len(r["entities"]) for r in records)


@click.command(
    help="Собирает пул упоминаний для mention replacement из внешних NER-корпусов."
)
@click.option(
    "--train",
    type=click.Path(path_type=Path, exists=True),
    default=Path("data/train.jsonl"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("data/mention_pool.jsonl"),
    show_default=True,
)
@click.option(
    "--gold-tsv",
    type=click.Path(path_type=Path),
    default=Path("augmentation/Uzbek_NER_Gold.tsv"),
    show_default=True,
)
@click.option(
    "--kaggle-tsv",
    type=click.Path(path_type=Path),
    default=Path("augmentation/courpusNER2015.tsv"),
    show_default=True,
)
@click.option("--wikiann-uz/--no-wikiann-uz", default=True, show_default=True)
@click.option("--wikineural-ru/--no-wikineural-ru", default=True, show_default=True)
@click.option("--wikineural-en/--no-wikineural-en", default=True, show_default=True)
@click.option(
    "--silver-min-count",
    type=int,
    default=2,
    show_default=True,
    help="Порог частоты для WikiANN/WikiNEuRal",
)
@click.option("--min-type-purity", type=float, default=0.9, show_default=True)
@click.option(
    "--stats",
    type=click.Path(path_type=Path),
    default=None,
    help="json со статистикой источников",
)
def main(
    train: Path,
    output: Path,
    gold_tsv: Path,
    kaggle_tsv: Path,
    wikiann_uz: bool,
    wikineural_ru: bool,
    wikineural_en: bool,
    silver_min_count: int,
    min_type_purity: float,
    stats: Path | None,
) -> None:
    reference = read_records(train)
    pool = MentionPool()
    table = Table(title="Источники пула упоминаний")
    for column in ("источник", "язык", "предложений", "упоминаний", "основ в пуле"):
        table.add_column(column)
    report: dict[str, Any] = {}

    def add(name: str, lang: str, records: list[Record], **kwargs: Any) -> None:
        part = build_mention_pool(
            records,
            reference_records=reference,
            min_type_purity=min_type_purity,
            lang=lang,
            source=name,
            **kwargs,
        )
        before = {label: len(entries) for label, entries in pool.entries.items()}
        pool.merge(part)
        added = {
            label: len(entries) - before.get(label, 0)
            for label, entries in pool.entries.items()
        }
        report[name] = {
            "lang": lang,
            "sentences": len(records),
            "mentions": entity_count(records),
            "added": added,
        }
        table.add_row(
            name, lang, str(len(records)), str(entity_count(records)), json.dumps(added)
        )

    if gold_tsv.exists():
        with console.status("Mendeley Gold"):
            add("gold", "uz", read_gold_tsv(gold_tsv), min_count=1, transliterate=True)
    else:
        console.print(
            f"[yellow]нет {gold_tsv}: Mendeley Gold пропущен (скачивается вручную)[/yellow]"
        )

    if not kaggle_tsv.exists():
        with console.status("Kaggle courpusNER2015: скачиваю через kagglehub"):
            fetch_kaggle_tsv(kaggle_tsv)
    with console.status("Kaggle courpusNER2015"):
        add(
            "kaggle", "uz", read_kaggle_tsv(kaggle_tsv), min_count=1, transliterate=True
        )

    if wikiann_uz:
        with console.status("WikiANN uz"):
            records = read_hf_tagged(
                "unimelb-nlp/wikiann", "uz", ("train", "validation", "test"), None
            )
            # ORG в WikiANN — заголовки статей ("Andronovo madaniyati", "Moʻgʻullar istilosi"), не организации
            records = [
                {**r, "entities": [e for e in r["entities"] if e["label"] != "ORG"]}
                for r in records
            ]
            add("wikiann_uz", "uz", records, min_count=1, transliterate=True)

    if wikineural_ru:
        with console.status("WikiNEuRal ru"):
            records = read_hf_tagged(
                "Babelscape/wikineural", None, ("train_ru",), WIKINEURAL_TAGS
            )
            records, dropped = keep_nominative(records)
            console.print(
                f"WikiNEuRal ru: отброшено не в именительном падеже / не кириллица: {dropped}"
            )
            add(
                "wikineural_ru",
                "ru",
                records,
                min_count=silver_min_count,
                transliterate=True,
                split_endings=False,
            )

    if wikineural_en:
        with console.status("WikiNEuRal en"):
            records = read_hf_tagged(
                "Babelscape/wikineural", None, ("train_en",), WIKINEURAL_TAGS
            )
            add(
                "wikineural_en",
                "en",
                records,
                min_count=silver_min_count,
                split_endings=False,
            )

    write_pool_jsonl(pool, output)
    console.print(table)
    summary = pool.stats()
    console.print(f"Пул: {json.dumps(summary, ensure_ascii=False)}")
    console.print(
        f"Записано {sum(len(e) for e in pool.entries.values())} основ в {output}"
    )
    if stats is not None:
        stats.parent.mkdir(parents=True, exist_ok=True)
        stats.write_text(
            json.dumps(
                {"sources": report, "pool": summary}, ensure_ascii=False, indent=2
            )
            + "\n"
        )

    langs = Counter(e.lang for entries in pool.entries.values() for e in entries)
    if langs.get("uz", 0) == 0:
        raise SystemExit("в пуле нет узбекских основ: проверьте источники")


if __name__ == "__main__":
    main()
