"""Логиты по токенам -> символьные spans в исходном тексте.

Единственный декодер для обучения (метрики), валидации и инференса. Три шага:

1. Пословная агрегация. Токены группируются в "слова" — максимальные цепочки
   буквенно-цифровых символов текста: токен продолжает слово, если начинается
   ровно там, где кончился предыдущий, и по обе стороны границы стоят буквы или
   цифры. Тип слова (O или X) — среднее по его токенам P(B-X)+P(I-X); выбор
   между B-X и I-X — по первому токену слова, потому что B в обучении стоит
   только на нём (простое усреднение размывает B у многотокенных слов и
   склеивает соседние сущности одного типа).
   Границы группируются по тексту, а не по word_ids() токенизатора: тот режет
   "C1" на "C" и "1", а в разметке границ внутри буквенно-цифровой цепочки нет
   (0 из 66k), так что слово — минимальная единица, на которой спан может
   начаться или кончиться. Спец-токены и пробельные (пустой оффсет) прозрачны.

2. Constrained Viterbi по словам с фиксированной матрицей переходов: 0 для
   допустимых, -inf для запрещённых (O -> I-X, B-X -> I-Y, I-X -> I-Y при
   X != Y, старт с I-X). Эмиссии — log усреднённых вероятностей.

3. Сборка spans: B-X открывает, I-X продолжает, O закрывает; start/end по
   первому и последнему слову; края очищаются от невидимых Cf-символов.

Обрывки слов (Brown -> ĠBro|wn -> ORG:'wn') и невалидные цепочки тегов после
этого невозможны по построению.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from tanerlan.modern_bert.ner.data.records import Entity
from tanerlan.modern_bert.ner.labels import LabelSchema
from tanerlan.modern_bert.tokenizer.tokenization_utils import trim_span_edges

EntityKey = tuple[str, int, int]
_NEG_INF = -1e30
_EPS = 1e-12


@dataclass(slots=True)
class Word:
    start: int
    end: int
    tokens: list[int]  # индексы токенов


def group_words(text: str, offsets: Sequence[tuple[int, int]]) -> list[Word]:
    """Токены -> слова (буквенно-цифровые цепочки текста); пустые оффсеты пропускаются."""
    words: list[Word] = []
    for index, (start, end) in enumerate(offsets):
        start, end = int(start), int(end)
        if start >= end:
            continue
        if (
            words
            and words[-1].end == start
            and text[start - 1].isalnum()
            and text[start].isalnum()
        ):
            words[-1].end = end
            words[-1].tokens.append(index)
        else:
            words.append(Word(start=start, end=end, tokens=[index]))
    return words


class BioTransitions:
    """Маска допустимых переходов BIO и индексы B/I по типам (не обучается)."""

    def __init__(self, schema: LabelSchema) -> None:
        n = schema.num_tags
        self.start = np.zeros(n, dtype=np.float64)
        self.transition = np.zeros((n, n), dtype=np.float64)  # [prev, cur]
        self.begin_ids = np.array([schema.begin_id(t) for t in schema.entity_types], dtype=np.int64)
        self.inside_ids = np.array([schema.inside_id(t) for t in schema.entity_types], dtype=np.int64)
        self.o_id = schema.o_id
        for cur in range(n):
            parsed = schema.split(cur)
            if parsed is None or parsed[0] == "B":
                continue
            entity_type = parsed[1]
            self.start[cur] = _NEG_INF
            for prev in range(n):
                prev_parsed = schema.split(prev)
                if prev_parsed is None or prev_parsed[1] != entity_type:
                    self.transition[prev, cur] = _NEG_INF


def word_emissions(probs: np.ndarray, words: list[Word], transitions: BioTransitions) -> np.ndarray:
    """(W, C) вероятности тегов по словам: тип — среднее по токенам, B/I — по первому токену."""
    out = np.zeros((len(words), probs.shape[1]), dtype=np.float64)
    b, i = transitions.begin_ids, transitions.inside_ids
    for row, word in enumerate(words):
        mean = probs[word.tokens].mean(axis=0)
        first = probs[word.tokens[0]]
        type_mass = mean[b] + mean[i]
        begin_share = first[b] / np.maximum(first[b] + first[i], _EPS)
        out[row, transitions.o_id] = mean[transitions.o_id]
        out[row, b] = type_mass * begin_share
        out[row, i] = type_mass * (1.0 - begin_share)
    return out


def viterbi(emissions: np.ndarray, transitions: BioTransitions) -> list[int]:
    """emissions: (W, C) log-вероятности; возвращает лучшую допустимую цепочку тегов."""
    n_words, n_tags = emissions.shape
    if n_words == 0:
        return []
    score = emissions[0] + transitions.start
    backpointers = np.empty((n_words, n_tags), dtype=np.int64)
    for i in range(1, n_words):
        candidates = score[:, None] + transitions.transition  # [prev, cur]
        backpointers[i] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + emissions[i]
    tags = [int(score.argmax())]
    for i in range(n_words - 1, 0, -1):
        tags.append(int(backpointers[i, tags[-1]]))
    tags.reverse()
    return tags


def spans_from_tags(
    tags: Sequence[int],
    words: Sequence[tuple[int, int]],
    text: str,
    schema: LabelSchema,
) -> list[Entity]:
    """Теги по словам -> spans: B-X открывает, I-X продолжает, O закрывает; края чистятся."""
    entities: list[Entity] = []
    current: Entity | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        start, end = trim_span_edges(text, current["start"], current["end"])
        if start < end:
            entities.append({"label": current["label"], "start": start, "end": end})
        current = None

    for (word_start, word_end), tag in zip(words, tags, strict=True):
        parsed = schema.split(tag)
        if parsed is None:
            flush()
            continue
        prefix, entity_type = parsed
        if prefix == "B" or current is None or current["label"] != entity_type:
            flush()
            current = {"label": entity_type, "start": word_start, "end": word_end}
        else:
            current["end"] = word_end
    flush()
    return entities


def decode_word_emissions(
    emissions: np.ndarray,
    words: Sequence[tuple[int, int]],
    text: str,
    schema: LabelSchema,
    transitions: BioTransitions,
    type_scales: Mapping[str, float] | None = None,
) -> list[Entity]:
    """emissions: (W, C) вероятности тегов по словам -> constrained Viterbi -> spans.
    type_scales ({"ORG": 1.2}) умножает эмиссии B-/I-тегов типа перед Viterbi."""
    if len(words) == 0:
        return []
    if type_scales:
        emissions = emissions.copy()
        for entity_type, scale in type_scales.items():
            emissions[:, [schema.begin_id(entity_type), schema.inside_id(entity_type)]] *= scale
    tags = viterbi(np.log(emissions + _EPS), transitions)
    return spans_from_tags(tags, words, text, schema)


def decode_probs(
    probs: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    text: str,
    schema: LabelSchema,
    transitions: BioTransitions,
    type_scales: Mapping[str, float] | None = None,
) -> list[Entity]:
    """probs: (T, C) вероятности тегов по токенам (T = len(offsets)) -> spans."""
    words = group_words(text, offsets)
    if not words:
        return []
    emissions = word_emissions(probs, words, transitions)
    return decode_word_emissions(emissions, [(w.start, w.end) for w in words], text, schema, transitions, type_scales)


def entity_keys(entities: Sequence[Entity]) -> set[EntityKey]:
    return {(e["label"], e["start"], e["end"]) for e in entities}
