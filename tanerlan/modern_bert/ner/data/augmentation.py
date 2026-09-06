"""Аугментация текста: транслитерация куска текста между кириллицей и латиницей.

В данных реально встречается смена письменности внутри одного текста, поэтому
транслитерируется непрерывный кусок из нескольких слов (иногда весь текст):
и обычные слова, и сущности внутри куска, с пересчётом координат сущностей.

cyr2lat всегда выравниваем по границам символов (каждая буква даёт 1-2
латинских, кроме выпадающего "ь"). lat2cyr схлопывает диграфы (sh, ch, oʻ,
...) в одну букву, поэтому если граница сущности попадает внутрь диграфа,
пример не аугментируется.

Таблицы и посимвольный транслит взяты из augmentation/transliteration.py.

Регистр, два уровня: весь текст или только спаны сущностей, в нижний регистр
или капс. Модель иначе выучивает "заглавная буква = сущность": пропускает
jkga/instagram/амрико со строчной, ХАМИД/НАРГИЗА в капсе и видит имена в
немецких Liebe/Freude. Смена регистра посимвольная и только там, где длина
не меняется (İ -> i̇ и ß -> SS пропускаются), поэтому координаты сущностей
остаются прежними.

Замена упоминаний: сущность меняется на другую того же типа и письменности из
пула (data/mention_pool.py: train-разметка или внешний Mendeley Gold), контекст
вокруг остаётся настоящим. Все формы одной основы в документе (Toshkent,
Toshkentda) получают одну и ту же замену, чтобы документ оставался связным.
Модель перестаёт полагаться на запомненные строки и учится читать контекст и
морфологию; приоритет длинным заменам (mention_long_bias, отдельно по типам)
компенсирует перекос данных к однословным сущностям.
"""

import random
import re
from dataclasses import dataclass
from typing import Literal

from augmentation.transliteration import (
    _build_offsets,
    _cyr2lat_pieces,
    _lat2cyr_pieces,
    classify_script,
)
from tanerlan.modern_bert.ner.config import AugmentationConfig
from tanerlan.modern_bert.ner.data.language import detect_language
from tanerlan.modern_bert.ner.data.mention_pool import (
    MentionPool,
    attach_ending,
    is_clean_mention,
)
from tanerlan.modern_bert.ner.data.records import Entity
from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

Direction = Literal["cyr2lat", "lat2cyr"]
CaseMode = Literal["lower", "upper"]
CaseScope = Literal["text", "entities"]

_WORD = re.compile(r"\S+")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass(slots=True)
class AugmentedText:
    text: str
    entities: list[Entity]
    direction: Direction
    chunk: tuple[int, int]  # координаты куска в исходном тексте


def has_cyrillic(text: str) -> bool:
    return _CYRILLIC.search(text) is not None


def has_latin(text: str) -> bool:
    return _LATIN.search(text) is not None


def transliterate_chunk(
    text: str,
    entities: list[Entity],
    rng: random.Random,
    config: AugmentationConfig,
    direction: Direction,
) -> AugmentedText | None:
    """Транслитерирует кусок текста; None, если нечего переводить или граница сущности небезопасна."""
    script_re = _CYRILLIC if direction == "cyr2lat" else _LATIN
    words = [match.span() for match in _WORD.finditer(text)]
    candidate_words = [i for i, (start, end) in enumerate(words) if script_re.search(text[start:end])]
    if not candidate_words:
        return None

    if rng.random() < config.full_text_prob:
        chunk_start, chunk_end = 0, len(text)
    else:
        max_words = max(config.min_chunk_words, int(len(words) * config.max_chunk_fraction))
        n_words = rng.randint(config.min_chunk_words, max_words)
        first = rng.choice(candidate_words)
        last = min(first + n_words, len(words)) - 1
        chunk_start, chunk_end = words[first][0], words[last][1]

    # апострофы приводятся к ʻ/ʼ (длина сохраняется): lat2cyr узнаёт oʻ/gʻ как ў/ғ
    normalized = prepare_input(text, homoglyphs=False)
    chunk = normalized[chunk_start:chunk_end]
    if direction == "cyr2lat":
        chunk_pieces = _cyr2lat_pieces(chunk)
        boundary_unsafe = [False] * (len(chunk) + 1)
    else:
        chunk_pieces, boundary_unsafe = _lat2cyr_pieces(chunk)

    for entity in entities:
        for boundary in (entity["start"], entity["end"]):
            if chunk_start <= boundary <= chunk_end and boundary_unsafe[boundary - chunk_start]:
                return None

    pieces = list(normalized)
    pieces[chunk_start:chunk_end] = chunk_pieces
    offsets = _build_offsets(pieces)
    new_text = "".join(pieces)

    new_entities: list[Entity] = []
    for entity in entities:
        new_start, new_end = offsets[entity["start"]], offsets[entity["end"]]
        if new_start >= new_end:
            # сущность из одних выпадающих символов ("ь") — не аугментируем пример
            return None
        new_entities.append({"label": entity["label"], "start": new_start, "end": new_end})
    return AugmentedText(
        text=new_text, entities=new_entities, direction=direction, chunk=(chunk_start, chunk_end)
    )


def choose_direction(text: str, rng: random.Random, config: AugmentationConfig) -> Direction | None:
    """cyr2lat с вероятностью cyr2lat_prob, если есть кириллица, иначе lat2cyr с lat2cyr_prob."""
    if has_cyrillic(text) and rng.random() < config.cyr2lat_prob:
        return "cyr2lat"
    if has_latin(text) and rng.random() < config.lat2cyr_prob:
        return "lat2cyr"
    return None


def change_case(
    text: str, mode: CaseMode, spans: list[tuple[int, int]] | None = None
) -> str | None:
    """Посимвольная смена регистра с сохранением длины; None, если текст не изменился.

    spans — если заданы, меняются только символы внутри этих отрезков.
    """
    convert = str.lower if mode == "lower" else str.upper
    chars = list(text)
    changed = False
    ranges = spans if spans is not None else [(0, len(text))]
    for start, end in ranges:
        for i in range(start, end):
            new = convert(chars[i])
            if new != chars[i] and len(new) == 1:
                chars[i] = new
                changed = True
    return "".join(chars) if changed else None


def choose_case(
    rng: random.Random, config: AugmentationConfig
) -> tuple[CaseMode, CaseScope] | None:
    """Одна выборка на документ: весь текст в lower/upper или только сущности в lower/upper."""
    draw = rng.random()
    options: list[tuple[float, CaseMode, CaseScope]] = [
        (config.lower_prob, "lower", "text"),
        (config.upper_prob, "upper", "text"),
        (config.entity_lower_prob, "lower", "entities"),
        (config.entity_upper_prob, "upper", "entities"),
    ]
    threshold = 0.0
    for prob, mode, scope in options:
        threshold += prob
        if draw < threshold:
            return mode, scope
    return None


def _match_case(source: str, replacement: str) -> str:
    if source.islower():
        return replacement.lower()
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    return replacement


def replace_mentions(
    text: str,
    entities: list[Entity],
    rng: random.Random,
    config: AugmentationConfig,
    pool: MentionPool,
) -> tuple[str, list[Entity]] | None:
    """Меняет до mention_max_per_doc разных упоминаний; None, если менять нечего."""
    if not entities:
        return None
    ordered = sorted(entities, key=lambda e: e["start"])
    # основа -> (тип, список её форм в документе): Toshkent и Toshkentda меняются на одну замену
    groups: dict[str, tuple[str, str, list[tuple[str, str]]]] = {}
    for entity in ordered:
        surface = text[entity["start"] : entity["end"]]
        if not is_clean_mention(surface):
            continue
        stem, ending = pool.split_ending(surface)
        key = (stem.lower(), entity["label"])
        group = groups.setdefault(f"{key[0]}\x00{key[1]}", (entity["label"], stem, []))
        if all(form != surface for form, _ in group[2]):
            group[2].append((surface, ending))
    if not groups:
        return None
    order = list(groups)
    rng.shuffle(order)
    lang = detect_language(text)  # замена того же языка, что документ (data/language.py)

    mapping: dict[str, str] = {}
    replaced_groups = 0
    for key in order:
        if replaced_groups >= config.mention_max_per_doc:
            break
        label, stem, forms = groups[key]
        max_words = len(stem.split()) + config.mention_max_extra_words
        candidates = [
            e for e in pool.candidates(label, classify_script(stem), lang, stem) if e.n_words <= max_words
        ]
        if not candidates:
            continue
        bias = config.long_bias(label)
        weights = [e.n_words**bias for e in candidates]
        chosen = rng.choices(candidates, weights=weights, k=1)[0]
        for surface, ending in forms:
            mapping[surface] = _match_case(stem, attach_ending(chosen.stem, ending))
        replaced_groups += 1
    if not mapping:
        return None

    pieces: list[str] = []
    new_entities: list[Entity] = []
    cursor = 0
    position = 0
    for entity in ordered:
        gap = text[cursor : entity["start"]]
        pieces.append(gap)
        position += len(gap)
        surface = text[entity["start"] : entity["end"]]
        replacement = mapping.get(surface, surface)
        pieces.append(replacement)
        new_entities.append({"label": entity["label"], "start": position, "end": position + len(replacement)})
        position += len(replacement)
        cursor = entity["end"]
    pieces.append(text[cursor:])
    return "".join(pieces), new_entities
