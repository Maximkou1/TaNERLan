"""Цели для span-головы: слова из токенов и ленточная матрица меток спанов.

Слово — та же буквенно-цифровая цепочка, что в decoding.group_words, так что
BIO- и span-голова видят одну и ту же разбивку. Спан (i, j) по словам хранится
в ленте [i, k], k = j - i < max_width: для документа из W слов это W x K ячеек
вместо W x W (в train до 3874 слов на документ, а сущности не длиннее 35 слов).

span_labels[i, k]: IGNORE_INDEX — недопустимый спан (i + k >= W или паддинг),
0 — нет сущности, schema.span_class(type) — gold-спан.
"""

from dataclasses import dataclass

from tanerlan.modern_bert.ner.data.dataset_preparation import IGNORE_INDEX, Offsets
from tanerlan.modern_bert.ner.data.records import Entity
from tanerlan.modern_bert.ner.decoding import Word, group_words
from tanerlan.modern_bert.ner.labels import LabelSchema

WordSpan = tuple[int, int]  # (start, end) в символах текста


@dataclass(slots=True)
class SpanTargets:
    words: list[WordSpan]
    word_index: list[int]  # по токенам: индекс слова или -1 (спец-/пробельный токен)
    span_labels: list[list[int]]  # (W, K)
    n_misaligned: int  # граница gold-спана не совпала с границей слова
    n_lost: int  # gold-спан за обрезкой или длиннее max_width


def entity_word_span(words: list[Word], entity: Entity) -> tuple[int, int] | None:
    """(первое, последнее) слово, пересекающееся с сущностью; None — слов нет (обрезка)."""
    first = last = None
    for index, word in enumerate(words):
        if word.end <= entity["start"]:
            continue
        if word.start >= entity["end"]:
            break
        if first is None:
            first = index
        last = index
    if first is None or last is None:
        return None
    return first, last


def build_span_targets(
    text: str,
    offsets: Offsets,
    entities: list[Entity],
    schema: LabelSchema,
    max_width: int,
) -> SpanTargets:
    words = group_words(text, offsets)
    n_words = len(words)
    word_index = [-1] * len(offsets)
    for index, word in enumerate(words):
        for token in word.tokens:
            word_index[token] = index

    span_labels = [
        [0 if i + k < n_words else IGNORE_INDEX for k in range(max_width)] for i in range(n_words)
    ]
    n_misaligned = n_lost = 0
    for entity in entities:
        located = entity_word_span(words, entity)
        if located is None:
            n_lost += 1
            continue
        first, last = located
        if words[first].start != entity["start"] or words[last].end != entity["end"]:
            n_misaligned += 1
        width = last - first
        if width >= max_width:
            n_lost += 1
            continue
        span_labels[first][width] = schema.span_class(entity["label"])

    return SpanTargets(
        words=[(word.start, word.end) for word in words],
        word_index=word_index,
        span_labels=span_labels,
        n_misaligned=n_misaligned,
        n_lost=n_lost,
    )
