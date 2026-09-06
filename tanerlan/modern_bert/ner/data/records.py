"""Чтение и валидация jsonl с разметкой (формат data/dataset_manifest.json)."""

import json
from pathlib import Path
from typing import Any, TypedDict


class Entity(TypedDict):
    label: str
    start: int
    end: int


class Record(TypedDict):
    hash: str
    text: str
    entities: list[Entity]
    source: str | None  # источник записи (train, kaggle_courpusner2015, ...), для взвешенного сэмплирования


def entity_key(entity: Entity) -> tuple[str, int, int]:
    return entity["label"], entity["start"], entity["end"]


def _validate_entities(raw: Any, text: str, source: str) -> list[Entity]:
    if not isinstance(raw, list):
        raise TypeError(f"{source}: entities must be an array")
    entities: list[Entity] = []
    seen: set[tuple[str, int, int]] = set()
    for index, entity in enumerate(raw):
        if not isinstance(entity, dict):
            raise TypeError(f"{source}/entities[{index}]: entity must be an object")
        label, start, end = entity.get("label"), entity.get("start"), entity.get("end")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{source}/entities[{index}]: label must be a non-empty string")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        key = (label, start, end)
        if key in seen:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        seen.add(key)
        entities.append({"label": label, "start": start, "end": end})

    entities.sort(key=lambda e: (e["start"], e["end"], e["label"]))
    for left, right in zip(entities, entities[1:], strict=False):
        if right["start"] < left["end"]:
            raise ValueError(f"{source}: overlapping entities are not supported")
    return entities


def read_records(
    path: Path,
    require_entities: bool = True,
    limit: int | None = None,
) -> list[Record]:
    """Читает jsonl; без require_entities поле entities может отсутствовать (тестовый сет)."""
    records: list[Record] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            source = f"{path}:{line_number}"
            if not isinstance(raw, dict):
                raise TypeError(f"{source}: record must be an object")
            record_hash, text = raw.get("hash"), raw.get("text")
            if not isinstance(record_hash, str) or not record_hash:
                raise ValueError(f"{source}: hash must be a non-empty string")
            if not isinstance(text, str):
                raise TypeError(f"{source}: text must be a string")
            if record_hash in seen_hashes:
                raise ValueError(f"{source}: duplicate hash {record_hash}")
            seen_hashes.add(record_hash)

            if require_entities or "entities" in raw:
                entities = _validate_entities(raw.get("entities"), text, source)
            else:
                entities = []
            record_source = raw.get("source")
            if record_source is not None and (not isinstance(record_source, str) or not record_source):
                raise ValueError(f"{source}: source must be a non-empty string")
            records.append(
                {"hash": record_hash, "text": text, "entities": entities, "source": record_source}
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
