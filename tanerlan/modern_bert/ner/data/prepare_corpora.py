"""Скачивает внешние NER-датасеты, конвертирует в формат кейса и собирает stage1.jsonl.

Всё захардкожено: источники, маппинг классов в ORG/NAME/GEO, потолки по числу
сущностей. Указываешь только директорию.

    python fetch_external_ner.py --output data/external
    python fetch_external_ner.py --output data/external --train data/train.jsonl

Результат в директории:
    kaznerd.jsonl, wikiann_tr.jsonl, wikineural_ru.jsonl, wikineural_en.jsonl,
    wikiann_az.jsonl, wikiann_uz.jsonl      — по источнику, целиком
    stage1.jsonl                            — смесь с потолками, для первой стадии
    mixed_train.jsonl                       — оригинал ×2 + stage1, если передан --train
"""

import hashlib
import json
import random
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import rich_click as click
from datasets import Sequence, load_dataset
from rich.console import Console
from rich.table import Table

console = Console()
SEED = 1337
GROUP = 4
TARGET = ("ORG", "NAME", "GEO")
WIKIANN_MAP = {"PER": "NAME", "LOC": "GEO", "ORG": "ORG"}
WIKINEURAL_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
KAZNERD_RAW = "https://raw.githubusercontent.com/IS2AI/KazNERD/main/KazNERD/IOB2_{split}.txt"


@dataclass
class Source:
    name: str
    lang: str
    cap: int
    kind: str
    dataset: str = ""
    config: str | None = None
    splits: tuple[str, ...] = ("train",)
    mapping: dict = None
    tag_names: list[str] | None = None


SOURCES = [
    Source("kaznerd", "kk", 35_000, "kaznerd", splits=("train", "valid", "test"),
           mapping={"PERSON": "NAME", "ORGANISATION": "ORG", "GPE": "GEO", "LOCATION": "GEO", "FACILITY": "GEO"}),
    Source("wikiann_tr", "tr", 25_000, "hf", "unimelb-nlp/wikiann", "tr", ("train", "validation", "test"), WIKIANN_MAP),
    Source("wikineural_ru", "ru", 25_000, "hf", "Babelscape/wikineural", None, ("train_ru",), WIKIANN_MAP, WIKINEURAL_TAGS),
    Source("wikineural_en", "en", 10_000, "hf", "Babelscape/wikineural", None, ("train_en",), WIKIANN_MAP, WIKINEURAL_TAGS),
    Source("wikiann_az", "az", 5_000, "hf", "unimelb-nlp/wikiann", "az", ("train", "validation", "test"), WIKIANN_MAP),
    Source("wikiann_uz", "uz", 10**9, "hf", "unimelb-nlp/wikiann", "uz", ("train", "validation", "test"), WIKIANN_MAP),
]

NO_SPACE_BEFORE = set(",.;:!?)]}»”’%…" + chr(39))
NO_SPACE_AFTER = set("([{«“‘")
Sentence = tuple[list[str], list[str]]


# --- конвертация ---------------------------------------------------------------


def detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    parts, offsets, length = [], [], 0
    for index, token in enumerate(tokens):
        if index > 0 and not (token[0] in NO_SPACE_BEFORE or tokens[index - 1][-1] in NO_SPACE_AFTER):
            parts.append(" ")
            length += 1
        parts.append(token)
        offsets.append((length, length + len(token)))
        length += len(token)
    return "".join(parts), offsets


def bio_to_spans(tags: list[str]) -> Iterator[tuple[int, int, str]]:
    start = label = None
    for index, tag in enumerate(tags + ["O"]):
        prefix, _, kind = tag.partition("-")
        if prefix == "I" and kind == label and start is not None:
            continue
        if start is not None:
            yield start, index, label
            start = label = None
        if prefix in ("B", "I"):
            start, label = index, kind


APOSTROPHES = set(chr(39) + "\u2019\u02bb\u02bc")


def _touches_word(text: str, index: int, step: int) -> bool:
    """Есть ли буква или цифра между позицией и ближайшим пробелом в направлении step."""
    while 0 <= index < len(text) and not text[index].isspace():
        if text[index].isalnum():
            return True
        index += step
    return False


def clean_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Срезает пунктуацию с краёв спана; None, если спан внутрисловный или пуст.

    Silver-разметки вроде WikiANN иногда начинают сущность с суффикса ('ın)
    или захватывают скобку. Спан, начинающийся с апострофа, — суффикс.
    Слева и справа до пробела не должно быть букв, иначе спан внутри слова;
    справа допускается апостроф — так турецкий отделяет падеж: İstanbul'da.
    """
    if text[start] in APOSTROPHES:
        return None
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    if start >= end:
        return None
    if _touches_word(text, start - 1, -1):
        return None
    if end < len(text) and text[end] not in APOSTROPHES and _touches_word(text, end, 1):
        return None
    return start, end


DROPPED: dict[str, int] = {}


def to_records(sentences: Iterator[Sentence], source: Source) -> Iterator[dict]:
    buffer: list[Sentence] = []
    counter = 0
    DROPPED[source.name] = 0

    def flush() -> Iterator[dict]:
        nonlocal counter
        if not buffer:
            return
        texts, entities, base = [], [], 0
        for tokens, tags in buffer:
            text, offsets = detokenize(tokens)
            for s, e, kind in bio_to_spans(tags):
                label = source.mapping.get(kind)
                if not label:
                    continue
                span = clean_span(text, offsets[s][0], offsets[e - 1][1])
                if span is None:
                    DROPPED[source.name] += 1
                    continue
                entities.append({"label": label, "start": base + span[0], "end": base + span[1]})
            texts.append(text)
            base += len(text) + 1
        document = "\n".join(texts)
        counter += 1
        yield {"hash": hashlib.sha1(f"{source.name}:{counter}:{document}".encode()).hexdigest(),
               "text": document, "entities": entities, "source": source.name, "lang": source.lang}
        buffer.clear()

    for sentence in sentences:
        buffer.append(sentence)
        if len(buffer) >= GROUP:
            yield from flush()
    yield from flush()


def read_conll_text(text: str) -> Iterator[Sentence]:
    tokens, tags = [], []
    for line in text.splitlines():
        if not line.strip():
            if tokens:
                yield tokens, tags
                tokens, tags = [], []
            continue
        columns = line.split()
        tokens.append(columns[0])
        tags.append(columns[-1])
    if tokens:
        yield tokens, tags


def read_kaznerd(source: Source) -> Iterator[Sentence]:
    for split in source.splits:
        with urllib.request.urlopen(KAZNERD_RAW.format(split=split)) as response:
            yield from read_conll_text(response.read().decode("utf-8"))


def read_hf(source: Source) -> Iterator[Sentence]:
    for split in source.splits:
        data = load_dataset(source.dataset, source.config, split=split)
        names = source.tag_names
        if names is None:
            feature = data.features["ner_tags"]
            names = (feature.feature if isinstance(feature, Sequence) else feature).names
        for record in data:
            yield list(record["tokens"]), [names[t] for t in record["ner_tags"]]


def verify(records: list[dict]) -> None:
    for record in records:
        text = record["text"]
        for entity in record["entities"]:
            s, e = entity["start"], entity["end"]
            assert 0 <= s < e <= len(text) and not text[s].isspace() and not text[e - 1].isspace(), (record["hash"], entity)
            assert not (s > 0 and text[s - 1].isalpha()) and not (e < len(text) and text[e].isalpha()), (record["hash"], entity)


def write_jsonl(records: list[dict], path: Path, keep_meta: bool = True) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = record if keep_meta else {k: record[k] for k in ("hash", "text", "entities")}
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def cap_records(records: list[dict], cap: int, rng: random.Random) -> list[dict]:
    rng.shuffle(records)
    kept, total = [], 0
    for record in records:
        if total >= cap:
            break
        kept.append(record)
        total += len(record["entities"])
    return kept


# --- CLI ----------------------------------------------------------------------


@click.command()
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Директория для готовых файлов.")
@click.option("--train", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Оригинальный train.jsonl; если передан, дополнительно собирается mixed_train.jsonl.")
def main(output: Path, train: Path | None) -> None:
    """Скачать, сконвертировать и собрать внешние NER-данные для первой стадии обучения."""
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    stage1: list[dict] = []
    table = Table(title=str(output))
    for column in ("источник", "язык", "документов", "сущностей", "отброшено", "в stage1"):
        table.add_column(column, justify="right" if column not in ("источник", "язык") else "left")

    for source in SOURCES:
        with console.status(f"{source.name}: скачиваю и конвертирую"):
            reader = read_kaznerd if source.kind == "kaznerd" else read_hf
            records = list(to_records(reader(source), source))
            verify(records)
            write_jsonl(records, output / f"{source.name}.jsonl")
            kept = cap_records(list(records), source.cap, rng)
            stage1.extend(kept)
        table.add_row(source.name, source.lang, str(len(records)), str(sum(len(r["entities"]) for r in records)),
                      str(DROPPED[source.name]), str(sum(len(r["entities"]) for r in kept)))

    rng.shuffle(stage1)
    write_jsonl(stage1, output / "stage1.jsonl", keep_meta=False)
    table.add_row("stage1.jsonl", "", str(len(stage1)), "", "", str(sum(len(r["entities"]) for r in stage1)))

    if train is not None:
        original = [json.loads(line) for line in train.open(encoding="utf-8") if line.strip()]
        mixed = list(stage1)
        for repeat in range(2):
            for record in original:
                mixed.append({**record, "hash": record["hash"] if repeat == 0 else f"{record['hash']}-r1"})
        rng.shuffle(mixed)
        write_jsonl(mixed, output / "mixed_train.jsonl", keep_meta=False)
        table.add_row("mixed_train.jsonl", "", str(len(mixed)), "", "", str(sum(len(r["entities"]) for r in mixed)))

    console.print(table)


if __name__ == "__main__":
    main()
