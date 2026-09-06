"""Ансамбль моделей с разными токенизаторами и головами на общей сетке слов текста.

Слова у каждой модели строятся по её токенам (decoding.group_words), и у двух
токенизаторов они могут расходиться там, где токен склеивает букву с пунктуацией
("O'z"). Поэтому для ансамбля вводится каноническая разбивка только по тексту:
максимальные буквенно-цифровые цепочки плюс каждый прочий непробельный символ
отдельно (невидимые Cf-символы пропускаются). Выход каждой модели переносится на
неё по символьным оффсетам, после чего:

  все модели BIO  -> усредняются пословные вероятности тегов -> constrained Viterbi;
  иначе           -> всё приводится к ленте спанов (W, K, C) -> усредняется ->
                     декодер спанов (span_decoding.py).

BIO -> лента: вероятность спана (i, j) типа t при независимых тегах слов
  p = P_i(B-t) * prod_{k=i+1..j} P_k(I-t) * (1 - P_{j+1}(I-t)),
класс 0 (нет сущности) — остаток до единицы.
"""

import unicodedata
from collections.abc import Sequence

import numpy as np

from tanerlan.modern_bert.ner.decoding import BioTransitions, Word

WordSpan = tuple[int, int]
_EPS = 1e-12


def canonical_words(text: str) -> list[WordSpan]:
    words: list[WordSpan] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isalnum():
            if start is None:
                start = index
            continue
        if start is not None:
            words.append((start, index))
            start = None
        if not char.isspace() and unicodedata.category(char) != "Cf":
            words.append((index, index + 1))
    if start is not None:
        words.append((start, len(text)))
    return words


def char_to_word(words: Sequence[WordSpan], text_length: int) -> np.ndarray:
    """Индекс канонического слова для каждого символа, -1 вне слов."""
    mapping = np.full(text_length + 1, -1, dtype=np.int64)
    for index, (start, end) in enumerate(words):
        mapping[start:end] = index
    return mapping


def bio_word_emissions(
    probs: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    words: Sequence[WordSpan],
    text_length: int,
    transitions: BioTransitions,
) -> np.ndarray:
    """(T, C) вероятности тегов по токенам -> (W, C) по каноническим словам.

    Как decoding.word_emissions: тип — среднее по токенам слова, B/I — по первому
    токену; но если первый токен начался в предыдущем слове (токен склеил "O'z"),
    слово может только продолжать сущность (доля B = 0). Слово без токенов -> O.
    """
    mapping = char_to_word(words, text_length)
    tokens_by_word: list[list[int]] = [[] for _ in words]
    first_word_of_token: dict[int, int] = {}
    for index, (start, end) in enumerate(offsets):
        if start >= end:
            continue
        covered = np.unique(mapping[start:end])
        covered = covered[covered >= 0]
        if covered.size == 0:
            continue
        first_word_of_token[index] = int(covered[0])
        for word in covered:
            tokens_by_word[int(word)].append(index)

    out = np.zeros((len(words), probs.shape[1]), dtype=np.float64)
    b, i = transitions.begin_ids, transitions.inside_ids
    for row, tokens in enumerate(tokens_by_word):
        if not tokens:
            out[row, transitions.o_id] = 1.0
            continue
        mean = probs[tokens].mean(axis=0)
        first = probs[tokens[0]]
        type_mass = mean[b] + mean[i]
        if first_word_of_token[tokens[0]] < row:
            begin_share = np.zeros_like(type_mass)
        else:
            begin_share = first[b] / np.maximum(first[b] + first[i], _EPS)
        out[row, transitions.o_id] = mean[transitions.o_id]
        out[row, b] = type_mass * begin_share
        out[row, i] = type_mass * (1.0 - begin_share)
    return out


def none_band(n_words: int, max_width: int, num_classes: int) -> np.ndarray:
    band = np.zeros((n_words, max_width, num_classes), dtype=np.float64)
    band[..., 0] = 1.0
    return band


def emissions_to_band(emissions: np.ndarray, transitions: BioTransitions, max_width: int) -> np.ndarray:
    """(W, C_tags) вероятности тегов по словам -> (W, K, 1 + n_types) вероятности спанов."""
    n_words = emissions.shape[0]
    n_types = len(transitions.begin_ids)
    band = np.zeros((n_words, max_width, n_types + 1), dtype=np.float64)
    positions = np.arange(n_words)
    for t, (b, i) in enumerate(zip(transitions.begin_ids, transitions.inside_ids, strict=True)):
        p_begin = emissions[:, b]
        p_inside = emissions[:, i]
        cum = np.concatenate([[0.0], np.cumsum(np.log(p_inside + _EPS))])  # cum[k] = sum_{m<k} log I[m]
        after = np.append(1.0 - p_inside[1:], 1.0)  # after[j] = 1 - I[j+1], 1 на последнем слове
        for k in range(max_width):
            starts = positions[: n_words - k] if k < n_words else positions[:0]
            ends = starts + k
            inside = np.exp(cum[ends + 1] - cum[starts + 1])
            band[starts, k, t + 1] = p_begin[starts] * inside * after[ends]
    band[..., 0] = np.clip(1.0 - band[..., 1:].sum(axis=-1), 0.0, 1.0)
    # недопустимые ячейки (i + k >= W) — чистое "нет сущности"
    i_idx, k_idx = np.meshgrid(positions, np.arange(max_width), indexing="ij")
    invalid = i_idx + k_idx >= n_words
    band[invalid] = 0.0
    band[invalid, 0] = 1.0
    return band


def span_band_to_canonical(
    band: np.ndarray,
    model_words: Sequence[Word],
    words: Sequence[WordSpan],
    text_length: int,
    max_width: int,
) -> np.ndarray:
    """(Wm, Km, C) лента по словам модели -> (W, K, C) по каноническим словам; остальное — нет сущности."""
    num_classes = band.shape[-1]
    out = none_band(len(words), max_width, num_classes)
    if not model_words or not words:
        return out
    mapping = char_to_word(words, text_length)
    first = np.full(len(model_words), -1, dtype=np.int64)
    last = np.full(len(model_words), -1, dtype=np.int64)
    for index, word in enumerate(model_words):
        covered = mapping[word.start : word.end]
        covered = covered[covered >= 0]
        if covered.size:
            first[index], last[index] = int(covered[0]), int(covered[-1])
    n_model, k_model = band.shape[0], band.shape[1]
    i_idx, k_idx = np.meshgrid(np.arange(n_model), np.arange(k_model), indexing="ij")
    j_idx = np.minimum(i_idx + k_idx, n_model - 1)
    starts = first[i_idx]
    ends = last[j_idx]
    widths = ends - starts
    keep = (i_idx + k_idx < n_model) & (starts >= 0) & (ends >= 0) & (widths >= 0) & (widths < max_width)
    out[starts[keep], widths[keep]] = band[i_idx[keep], k_idx[keep]]
    return out
