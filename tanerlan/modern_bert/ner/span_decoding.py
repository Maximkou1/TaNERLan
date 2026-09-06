"""Вероятности спанов (W, K, C) -> непересекающиеся символьные spans.

greedy — кандидаты (argmax != 0) по убыванию вероятности, спан берётся, если не
пересекает уже взятые (Yu et al. 2020). dp — точный максимум суммы
log p(type) - log p(none) по множеству непересекающихся спанов (динамика по
позициям слов). Оба края спана чистятся от невидимых символов, как в BIO-декодере.
"""

from collections.abc import Mapping
from typing import Literal

import numpy as np

from tanerlan.modern_bert.ner.data.records import Entity
from tanerlan.modern_bert.ner.labels import LabelSchema
from tanerlan.modern_bert.tokenizer.tokenization_utils import trim_span_edges

SpanDecoding = Literal["greedy", "dp"]
_EPS = 1e-12


def _candidates(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(i, k, class, score) допустимых спанов, у которых argmax — не 'нет сущности'."""
    n_words, max_width, _ = probs.shape
    i_idx, k_idx = np.meshgrid(np.arange(n_words), np.arange(max_width), indexing="ij")
    valid = i_idx + k_idx < n_words
    classes = probs.argmax(axis=-1)
    keep = valid & (classes > 0)
    i_sel, k_sel = np.nonzero(keep)
    cls = classes[i_sel, k_sel]
    return i_sel, k_sel, cls, probs[i_sel, k_sel, cls]


def _greedy(probs: np.ndarray) -> list[tuple[int, int, int]]:
    i_sel, k_sel, cls, score = _candidates(probs)
    occupied = np.zeros(probs.shape[0], dtype=bool)
    chosen: list[tuple[int, int, int]] = []
    for idx in np.argsort(-score, kind="stable"):
        start, end = int(i_sel[idx]), int(i_sel[idx] + k_sel[idx])
        if occupied[start : end + 1].any():
            continue
        occupied[start : end + 1] = True
        chosen.append((start, end, int(cls[idx])))
    return chosen


def _dp(probs: np.ndarray) -> list[tuple[int, int, int]]:
    n_words = probs.shape[0]
    i_sel, k_sel, cls, score = _candidates(probs)
    gain = np.log(score + _EPS) - np.log(probs[i_sel, k_sel, 0] + _EPS)
    by_end: list[list[tuple[float, int, int]]] = [[] for _ in range(n_words)]
    for start, width, c, g in zip(i_sel, k_sel, cls, gain, strict=True):
        by_end[int(start + width)].append((float(g), int(start), int(c)))

    best = np.zeros(n_words + 1)
    choice: list[tuple[int, int] | None] = [None] * (n_words + 1)  # (start, class) спана, кончающегося на j
    for end in range(n_words):
        best[end + 1] = best[end]
        for g, start, c in by_end[end]:
            candidate = best[start] + g
            if candidate > best[end + 1]:
                best[end + 1] = candidate
                choice[end + 1] = (start, c)
    chosen: list[tuple[int, int, int]] = []
    position = n_words
    while position > 0:
        picked = choice[position]
        if picked is None:
            position -= 1
            continue
        start, c = picked
        chosen.append((start, position - 1, c))
        position = start
    chosen.reverse()
    return chosen


def decode_span_probs(
    probs: np.ndarray,
    words: list[tuple[int, int]],
    text: str,
    schema: LabelSchema,
    mode: SpanDecoding = "greedy",
    none_scale: float = 1.0,
    type_scales: Mapping[str, float] | None = None,
) -> list[Entity]:
    """probs: (W, K, C) вероятности классов спанов, words: (start, end) слов в символах.

    none_scale < 1 дисконтирует класс "нет сущности" перед выбором (порог recall/precision
    без переобучения; для ансамблей с BIO-моделями, где произведение вероятностей тегов
    занижает длинные спаны). type_scales — множители на классы отдельных типов ({"ORG": 1.2}
    поднимает recall ORG).
    """
    if len(words) == 0:
        return []
    probs = probs[: len(words)]
    if none_scale != 1.0 or type_scales:
        probs = probs.copy()
        probs[..., 0] *= none_scale
        for entity_type, scale in (type_scales or {}).items():
            probs[..., schema.span_class(entity_type)] *= scale
    spans = _greedy(probs) if mode == "greedy" else _dp(probs)
    entities: list[Entity] = []
    for first, last, c in sorted(spans):
        start, end = trim_span_edges(text, words[first][0], words[last][1])
        if start < end:
            entities.append({"label": schema.entity_types[c - 1], "start": start, "end": end})
    return entities
