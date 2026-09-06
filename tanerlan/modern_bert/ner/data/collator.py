"""Коллатор NER: аугментация на лету, паддинг, прокидывание текста и spans в батч."""

import random
from typing import Any, NotRequired, TypedDict, cast

import torch
from torch.utils.data import get_worker_info
from transformers import PreTrainedTokenizerBase

from tanerlan.modern_bert.ner.config import (
    AugmentationConfig,
    BioHeadConfig,
    HeadConfig,
    SpanHeadConfig,
)
from tanerlan.modern_bert.ner.data.augmentation import (
    change_case,
    choose_case,
    choose_direction,
    replace_mentions,
    transliterate_chunk,
)
from tanerlan.modern_bert.ner.data.dataset_preparation import (
    IGNORE_INDEX,
    Offsets,
    encode_record,
)
from tanerlan.modern_bert.ner.data.mention_pool import MentionPool
from tanerlan.modern_bert.ner.data.records import Entity
from tanerlan.modern_bert.ner.data.span_targets import WordSpan, build_span_targets
from tanerlan.modern_bert.ner.labels import LabelSchema


class NerBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor  # BIO-метки токенов (есть всегда: кэш датасета их хранит)
    # python-объекты Lightning не трогает при переносе на устройство
    offsets: list[Offsets]
    texts: list[str]
    entities: list[list[Entity]]
    hashes: list[str]
    augmented: list[bool]
    # только для span-головы (data/span_targets.py)
    word_index: NotRequired[torch.Tensor]  # (B, T): индекс слова токена или -1
    num_words: NotRequired[torch.Tensor]  # (B,)
    span_labels: NotRequired[torch.Tensor]  # (B, W, K): IGNORE_INDEX / 0 / класс типа
    words: NotRequired[list[list[WordSpan]]]


class NerCollator:
    """Аугментированные примеры токенизируются заново, остальные берутся из кэша.

    Генератор случайности создаётся лениво в каждом воркере DataLoader из его
    seed (base seed лоадера + worker id), поэтому воркеры не повторяют друг
    друга, а запуск воспроизводим при фиксированном сиде.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        schema: LabelSchema,
        max_length: int,
        pad_to_multiple_of: int | None,
        seed: int,
        augmentation: AugmentationConfig | None = None,
        mention_pool: MentionPool | None = None,
        head: HeadConfig = BioHeadConfig(),
    ) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has no pad token")
        self.tokenizer = tokenizer
        self.schema = schema
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.seed = seed
        self.span_max_width = head.max_span_width if isinstance(head, SpanHeadConfig) else None
        self.augmentation = (
            augmentation if augmentation is not None and augmentation.enabled else None
        )
        if self.augmentation is not None and self.augmentation.mention_replace_prob > 0.0 and mention_pool is None:
            raise ValueError("mention_replace_prob > 0 requires a mention_pool")
        self.mention_pool = mention_pool
        self._rng: random.Random | None = None

    def _get_rng(self) -> random.Random:
        if self._rng is None:
            worker_info = get_worker_info()
            self._rng = random.Random(
                self.seed if worker_info is None else worker_info.seed
            )
        return self._rng

    def _maybe_augment(self, example: dict[str, Any]) -> dict[str, Any]:
        """Замена упоминаний, транслитерация куска и смена регистра независимы и
        применяются в этом порядке; при любом изменении текста пример токенизируется заново."""
        if self.augmentation is None:
            return example
        rng = self._get_rng()
        text: str = example["text"]
        entities: list[Entity] = example["entities"]
        changed = False

        if self.mention_pool is not None and rng.random() < self.augmentation.mention_replace_prob:
            replaced = replace_mentions(text, entities, rng, self.augmentation, self.mention_pool)
            if replaced is not None:
                text, entities = replaced
                changed = True

        direction = choose_direction(text, rng, self.augmentation)
        if direction is not None:
            transliterated = transliterate_chunk(text, entities, rng, self.augmentation, direction)
            if transliterated is not None:
                text, entities = transliterated.text, transliterated.entities
                changed = True

        case_choice = choose_case(rng, self.augmentation)
        if case_choice is not None:
            case_mode, case_scope = case_choice
            spans = [(e["start"], e["end"]) for e in entities] if case_scope == "entities" else None
            if spans is None or spans:
                recased = change_case(text, case_mode, spans)
                if recased is not None:
                    text = recased  # длина сохраняется, entities те же
                    changed = True

        if not changed:
            return example
        encoded = encode_record(text, entities, self.tokenizer, self.schema, self.max_length)
        return {
            **example,
            "text": text,
            "entities": entities,
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
            "offsets": encoded["offsets"],
            "augmented": True,
        }

    def __call__(self, examples: list[dict[str, Any]]) -> NerBatch:
        examples = [self._maybe_augment(example) for example in examples]

        max_len = max(len(example["input_ids"]) for example in examples)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
        for row, example in enumerate(examples):
            length = len(example["input_ids"])
            input_ids[row, :length] = torch.tensor(
                example["input_ids"], dtype=torch.long
            )
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(example["labels"], dtype=torch.long)

        offsets = [
            [(int(start), int(end)) for start, end in example["offsets"]]
            for example in examples
        ]
        batch: NerBatch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "offsets": offsets,
            "texts": [example["text"] for example in examples],
            "entities": [example["entities"] for example in examples],
            "hashes": [example["hash"] for example in examples],
            "augmented": [
                bool(example.get("augmented", False)) for example in examples
            ],
        }
        if self.span_max_width is not None:
            batch = cast(NerBatch, {**batch, **self._span_targets(examples, offsets, max_len)})
        return batch

    def _span_targets(
        self, examples: list[dict[str, Any]], offsets: list[Offsets], max_len: int
    ) -> dict[str, Any]:
        """Слова и ленточные метки спанов; спаны строятся по (возможно аугментированным) тексту и сущностям."""
        assert self.span_max_width is not None
        targets = [
            build_span_targets(
                example["text"], example_offsets, example["entities"], self.schema, self.span_max_width
            )
            for example, example_offsets in zip(examples, offsets, strict=True)
        ]
        max_words = max(1, max(len(t.words) for t in targets))
        word_index = torch.full((len(examples), max_len), -1, dtype=torch.long)
        span_labels = torch.full(
            (len(examples), max_words, self.span_max_width), IGNORE_INDEX, dtype=torch.long
        )
        for row, target in enumerate(targets):
            word_index[row, : len(target.word_index)] = torch.tensor(target.word_index, dtype=torch.long)
            if target.words:
                span_labels[row, : len(target.words)] = torch.tensor(target.span_labels, dtype=torch.long)
        return {
            "word_index": word_index,
            "num_words": torch.tensor([len(t.words) for t in targets], dtype=torch.long),
            "span_labels": span_labels,
            "words": [t.words for t in targets],
        }
