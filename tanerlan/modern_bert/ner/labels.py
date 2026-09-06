"""Схема меток NER: типы сущностей из данных <-> BIO-теги и их id.

Типы сущностей не захардкожены: они собираются по train-выборке и
сохраняются в конфиг модели (label2id/id2label), откуда predict.py
восстанавливает схему.
"""

from collections.abc import Iterable, Mapping
from typing import Literal, override

O_TAG = "O"
Prefix = Literal["B", "I"]


class LabelSchema:
    def __init__(self, entity_types: Iterable[str]) -> None:
        types = sorted(set(entity_types))
        if not types:
            raise ValueError("Entity types are empty: nothing to train on")
        if any(t == O_TAG or "-" in t for t in types):
            raise ValueError(f"Invalid entity types: {types}")
        self.entity_types: list[str] = types
        self.tags: list[str] = [O_TAG] + [
            f"{prefix}-{entity_type}" for entity_type in types for prefix in ("B", "I")
        ]
        self.tag2id: dict[str, int] = {tag: idx for idx, tag in enumerate(self.tags)}
        self.id2tag: dict[int, str] = {idx: tag for tag, idx in self.tag2id.items()}
        self._type_index = {t: i for i, t in enumerate(types)}

    @property
    def num_tags(self) -> int:
        return len(self.tags)

    @property
    def o_id(self) -> int:
        return self.tag2id[O_TAG]

    def begin_id(self, entity_type: str) -> int:
        return self.tag2id[f"B-{entity_type}"]

    def inside_id(self, entity_type: str) -> int:
        return self.tag2id[f"I-{entity_type}"]

    def type_index(self, entity_type: str) -> int:
        return self._type_index[entity_type]

    # --- span-голова: классы 0 = нет сущности, k + 1 = entity_types[k] ---------------

    @property
    def num_span_classes(self) -> int:
        return len(self.entity_types) + 1

    @property
    def span_id2label(self) -> dict[int, str]:
        return {0: O_TAG, **{i + 1: t for i, t in enumerate(self.entity_types)}}

    @property
    def span_label2id(self) -> dict[str, int]:
        return {label: idx for idx, label in self.span_id2label.items()}

    def span_class(self, entity_type: str) -> int:
        return self._type_index[entity_type] + 1

    def split(self, tag_id: int) -> tuple[Prefix, str] | None:
        """(prefix, type) для B-/I-тега, None для O."""
        tag = self.id2tag[tag_id]
        if tag == O_TAG:
            return None
        prefix, entity_type = tag.split("-", 1)
        return prefix, entity_type  # ty: ignore[invalid-return-type]

    @classmethod
    def from_label2id(cls, label2id: Mapping[str, int]) -> "LabelSchema":
        """Восстанавливает схему из конфига модели и проверяет совпадение id.

        Понимает обе раскладки: BIO-теги (O, B-X, I-X) у token-classification модели
        и классы спанов (O, X) у span-модели.
        """
        labels = {k: int(v) for k, v in label2id.items()}
        is_bio = any(tag.startswith(("B-", "I-")) for tag in labels)
        types = {tag.split("-", 1)[1] if is_bio else tag for tag in labels if tag != O_TAG}
        schema = cls(types)
        expected = schema.tag2id if is_bio else schema.span_label2id
        if labels != expected:
            raise ValueError(f"label2id из конфига {labels} не совпадает со схемой {expected}")
        return schema

    @override
    def __repr__(self) -> str:
        return f"LabelSchema(entity_types={self.entity_types}, tags={self.tags})"
