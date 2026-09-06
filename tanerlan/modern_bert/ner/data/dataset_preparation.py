"""Подготовка данных для NER: токенизация с оффсетами, выравнивание BIO-меток, кэш.

Кэш лежит в ~/.cache/tanerlan/ner-datasets. Ключ учитывает содержимое
jsonl, токенизатор, max_length, схему меток и лимит: смена любого из них
пересобирает датасет, повторный запуск читает готовый arrow с диска.

В датасете, помимо input_ids/labels, остаются исходный текст, исходные
координаты сущностей и оффсеты токенов: коллатор отдаёт их в батч для
span-метрик, а аугментации работают с текстом и переводят его в токены
только для небольшого подмножества примеров.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict, cast

from datasets import Dataset, load_from_disk
from kostyl.utils import DirLock, setup_logger
from transformers import PreTrainedTokenizerBase

from augmentation.transliteration import classify_script
from tanerlan.modern_bert.ner.data.records import Entity, Record
from tanerlan.modern_bert.ner.labels import LabelSchema
from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

logger = setup_logger(fmt="only_message")

CACHE_ROOT = Path.home() / ".cache" / "tanerlan" / "ner-datasets"
IGNORE_INDEX = -100
_PREPARATION_VERSION = 1

Offsets = list[tuple[int, int]]


class EncodedRecord(TypedDict):
    input_ids: list[int]
    labels: list[int]
    offsets: Offsets
    n_tokens: int
    truncated: bool
    n_misaligned: int
    n_lost: int


def tokenizer_fingerprint(tokenizer: PreTrainedTokenizerBase) -> str:
    """Хэш содержимого токенизатора: другой словарь или нормализатор — другой кэш."""
    if getattr(tokenizer, "is_fast", False):
        payload = tokenizer.backend_tokenizer.to_str()
    else:
        payload = json.dumps(sorted(tokenizer.get_vocab().items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def trim_offsets(text: str, offsets: list[tuple[int, int]] | list[list[int]]) -> Offsets:
    """Убирает ведущие пробелы из оффсетов токенов.

    ByteLevel-претокенизатор приклеивает пробел к следующему слову ("Ġso'z"),
    и оффсет токена начинается с пробела; границы сущностей же стоят на буквах.
    Токен из одних пробелов схлопывается в пустой (start == end), как и
    спец-токены: он не получает метку и прозрачен при декодировании.
    """
    trimmed: Offsets = []
    for start, end in offsets:
        start, end = int(start), int(end)
        while start < end and text[start].isspace():
            start += 1
        trimmed.append((start, end))
    return trimmed


def align_labels(
    offsets: Offsets, entities: list[Entity], schema: LabelSchema
) -> tuple[list[int], int, int]:
    """Символьные spans -> BIO-метки токенов.

    Токен относится к сущности, если пересекается с ней; первый такой токен
    получает B-, остальные I-. Возвращает (labels, n_misaligned, n_lost):
    misaligned — граница сущности не совпала с границей токена (например,
    ".," слился в один токен), lost — сущность целиком за обрезкой max_length.
    """
    labels = [schema.o_id if start < end else IGNORE_INDEX for start, end in offsets]
    covered_end = max((end for _, end in offsets), default=0)
    n_misaligned = n_lost = 0
    n = len(offsets)
    i = 0
    for entity in entities:
        ent_start, ent_end = entity["start"], entity["end"]
        while i < n and (offsets[i][0] >= offsets[i][1] or offsets[i][1] <= ent_start):
            i += 1
        matched: list[int] = []
        j = i
        while j < n and offsets[j][0] < ent_end:
            if offsets[j][0] < offsets[j][1]:
                matched.append(j)
            j += 1
        if not matched:
            if ent_start >= covered_end:
                n_lost += 1
            else:
                n_misaligned += 1
            continue
        labels[matched[0]] = schema.begin_id(entity["label"])
        inside_id = schema.inside_id(entity["label"])
        for k in matched[1:]:
            labels[k] = inside_id
        if offsets[matched[0]][0] != ent_start or offsets[matched[-1]][1] != ent_end:
            n_misaligned += 1
        i = matched[0]
    return labels, n_misaligned, n_lost


def _is_truncated(text: str, offsets: Offsets) -> bool:
    covered_end = max((end for _, end in offsets), default=0)
    return covered_end < len(text.rstrip())


def encode_texts(
    texts: list[str],
    entities: list[list[Entity]],
    tokenizer: PreTrainedTokenizerBase,
    schema: LabelSchema,
    max_length: int,
) -> list[EncodedRecord]:
    """Нормализация (без изменения длины) -> токены с оффсетами -> BIO-метки."""
    prepared = [prepare_input(text) for text in texts]
    encoded = tokenizer(
        prepared,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    result: list[EncodedRecord] = []
    for text, record_entities, input_ids, raw_offsets in zip(
        prepared, entities, encoded["input_ids"], encoded["offset_mapping"], strict=True
    ):
        offsets = trim_offsets(text, raw_offsets)
        labels, n_misaligned, n_lost = align_labels(offsets, record_entities, schema)
        result.append(
            {
                "input_ids": list(input_ids),
                "labels": labels,
                "offsets": offsets,
                "n_tokens": len(input_ids),
                "truncated": _is_truncated(text, offsets),
                "n_misaligned": n_misaligned,
                "n_lost": n_lost,
            }
        )
    return result


def encode_record(
    text: str,
    entities: list[Entity],
    tokenizer: PreTrainedTokenizerBase,
    schema: LabelSchema,
    max_length: int,
) -> EncodedRecord:
    return encode_texts([text], [entities], tokenizer, schema, max_length)[0]


def _cache_key(
    records_path: Path,
    tokenizer_fp: str,
    schema: LabelSchema,
    max_length: int,
    limit: int | None,
) -> str:
    payload = {
        "file": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "tokenizer": tokenizer_fp,
        "tags": schema.tags,
        "max_length": max_length,
        "limit": limit,
        "normalize": "prepare_input",
        "version": _PREPARATION_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _build_dataset(
    records: list[Record],
    tokenizer: PreTrainedTokenizerBase,
    schema: LabelSchema,
    max_length: int,
    desc: str,
) -> Dataset:
    raw = Dataset.from_list(cast(list[dict[str, Any]], records))

    def encode(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        encoded = encode_texts(batch["text"], batch["entities"], tokenizer, schema, max_length)
        columns: dict[str, list[Any]] = {key: [] for key in EncodedRecord.__annotations__}
        for item in encoded:
            for key in columns:
                columns[key].append(item[key])  # ty: ignore[invalid-key]
        columns["script"] = [classify_script(text) for text in batch["text"]]
        return columns

    return raw.map(encode, batched=True, batch_size=256, desc=desc)


def prepare_dataset(
    records_path: Path,
    records: list[Record],
    tokenizer: PreTrainedTokenizerBase,
    schema: LabelSchema,
    max_length: int,
    limit: int | None,
    desc: str,
) -> Dataset:
    """Токенизированный датасет из кэша, при промахе строится и сохраняется на диск."""
    key = _cache_key(records_path, tokenizer_fingerprint(tokenizer), schema, max_length, limit)
    cache_path = CACHE_ROOT / key
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # Проверка и сборка под замком: иначе local ranks одновременно видят промах
    # и строят один и тот же кэш, читая при этом недописанную директорию.
    with DirLock(CACHE_ROOT):
        if not cache_path.exists():
            logger.info(f"Cache miss for {records_path} -> building {cache_path}")
            dataset = _build_dataset(records, tokenizer, schema, max_length, desc)
            dataset.save_to_disk(str(cache_path))
        else:
            logger.info(f"Cache hit for {records_path}: {cache_path}")
    return cast(Dataset, load_from_disk(str(cache_path)))


def entity_type_counts(records: list[Record]) -> Counter[str]:
    return Counter(entity["label"] for record in records for entity in record["entities"])


def dataset_stats(dataset: Dataset, records: list[Record]) -> dict[str, Any]:
    """Сводка по подготовленному датасету для логов и hparams."""
    n_tokens = sorted(dataset["n_tokens"])
    n = len(n_tokens)

    def percentile(p: float) -> int:
        return n_tokens[min(n - 1, int(n * p))]

    return {
        "records": n,
        "records_with_entities": sum(1 for r in records if r["entities"]),
        "entities": sum(len(r["entities"]) for r in records),
        "entities_by_type": dict(entity_type_counts(records)),
        "scripts": dict(Counter(dataset["script"])),
        "records_by_source": dict(Counter(r["source"] or "<none>" for r in records)),
        "tokens_total": sum(n_tokens),
        "tokens_p50": percentile(0.5),
        "tokens_p95": percentile(0.95),
        "tokens_p99": percentile(0.99),
        "tokens_max": n_tokens[-1],
        "truncated_records": sum(dataset["truncated"]),
        "entities_lost_by_truncation": sum(dataset["n_lost"]),
        "entities_misaligned_with_tokens": sum(dataset["n_misaligned"]),
    }
