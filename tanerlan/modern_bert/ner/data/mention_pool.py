"""Пул упоминаний для аугментации заменой сущностей (mention replacement).

Пул — список основ по типам (ORG/NAME/GEO), каждая с письменностью (latin/cyrillic)
и языком (uz/ru/en/other, data/language.py). Замена берётся того же типа, той же
письменности, что оригинал, и того же языка, что документ; если таких нет —
той же письменности любого языка.

Источники: внешние корпуса, собранные build_mention_pool.py в jsonl (Mendeley
"Uzbek NER Gold", Kaggle courpusNER2015, WikiANN uz, WikiNEuRal ru/en) — строки,
которых модель не видит в train, лучше отучают её от запоминания имён; либо
train-разметка (build_mention_pool на записях train), если внешнего пула нет.

Разметчики всегда включают падежное окончание в спан (Toshkent / Toshkentda /
Toshkentga — разные строки одной сущности), поэтому упоминание хранится как
основа + окончание: при замене основа берётся из пула, а окончание — от
заменяемого оригинала.

Окончание отщепляется только если голая основа известна: сама встречается в
разметке (в источнике пула или в train, если он передан как справочник) —
"Toshkentda" -> ("Toshkent", "da"), но "Kanada" остаётся целиком, потому что "Kana"
никто не размечал; для многословных основ достаточно, чтобы известным было последнее
слово основы ("London universitetida" -> "London universiteti", потому что
"universiteti" завершает "Toshkent davlat universiteti"; "Saudiya Arabistoni" не
режется: "Arabisto" ничего не завершает). Однозначные -ning/-dagi/-dan отщепляются
и у неизвестной основы ("Sinvarning" -> "Sinvar"), короткие -da/-ga/-ni — нет
("Kanada", "Nevada", "Govinda"). Притяжательные -i/-si (hokimligi, vazirligi,
kutubxonasi) не отщепляются: это часть названия организации, а не падеж.
Для русских и английских источников окончания не режутся вовсе (split_endings=False).

Фильтры мусора (юзернеймы, хэштеги, обрывки, шум с одного документа):
  - только буквы, цифры, пробел, апострофы, дефис, точка, амперсанд, запятая;
  - начинается и заканчивается буквой или цифрой, есть хотя бы одна буква;
  - письменность строго кириллица или латиница;
  - основа встречается в источнике не реже min_count раз (по всем формам);
  - основа размечена этим типом не реже чем в min_type_purity долях случаев
    ("Buxoro" GEO 36 / ORG 9 -> только в GEO-пул);
  - однословная основа, которая в справочных текстах пишется со строчной
    (hokimligi, agentlik, universitet) — нарицательное, а не имя; в пул не идёт.

Узбекские источники размечены латиницей, поэтому для кириллических документов
основы транслитерируются (transliterate=True); явно английские написания
(Microsoft, World Economic Forum) пропускаются — посимвольный транслит даёт для
них мусор. Русские основы транслитерируются в латиницу (Moskva, Ivanov) — так их
и пишут в латинских узбекских текстах.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from augmentation.transliteration import (
    _cyr2lat_pieces,
    _lat2cyr_pieces,
    classify_script,
)
from tanerlan.modern_bert.ner.data.language import Language, detect_language
from tanerlan.modern_bert.ner.data.records import Entity, Record
from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

_ALLOWED = re.compile(r"^[^\W_][^\W_ʻʼ'’‘\-.&,]*(?:(?:, |[ ʻʼ'’‘\-.&]+)[^\W_]+)*$")
_APOSTROPHES = "ʻʼ'’‘"
_HAS_LETTER = re.compile(r"[^\W\d_]")
_MAX_WORDS = 8
_MAX_WORD_LEN = 25

# падежные окончания, включаемые разметчиками в спан; порядок — от длинных к коротким
_ENDINGS_LAT = ("ning", "dagi", "dan", "da", "ga", "ka", "qa", "ni")
_ENDINGS_CYR = ("нинг", "даги", "дан", "да", "га", "ка", "қа", "ни")
_DATIVE = {"ga", "ka", "qa", "га", "ка", "қа"}
# однозначные окончания: отщепляются и у неизвестной основы (Sinvarning -> Sinvar), в отличие
# от коротких da/ga/ni, которые бывают частью имени (Kanada, Nevada, Govinda)
_UNAMBIGUOUS_ENDINGS = {"ning", "dagi", "dan", "нинг", "даги", "дан"}
_MIN_STEM_LETTERS = 3
_MIN_UNKNOWN_STEM_LETTERS = 4

# буквосочетания, которых нет в узбекской латинице: имя английское, транслит его исказит
_FOREIGN_LATIN = re.compile(r"w|c(?!h)|th|ph|ee|oo|ck", re.IGNORECASE)
# однословная основа считается нарицательной, если со строчной встречается не реже
_COMMON_WORD_MIN_COUNT = 3
_WORD = re.compile(r"[^\W\d_][\w'’‘ʻʼ\-]*")

GOLD_LABEL_MAP = {"PER": "NAME", "LOC": "GEO", "ORG": "ORG"}


def is_clean_mention(surface: str) -> bool:
    """Строка похожа на нормальное имя собственное, а не на мусор."""
    if len(surface) < 2 or surface != surface.strip():
        return False
    if not _ALLOWED.match(surface) or not _HAS_LETTER.search(surface):
        return False
    words = surface.split()
    if len(words) > _MAX_WORDS or any(len(w) > _MAX_WORD_LEN for w in words):
        return False
    return classify_script(surface) in ("cyrillic", "latin")


def _candidate_splits(surface: str) -> list[tuple[str, str]]:
    endings = _ENDINGS_CYR if classify_script(surface) == "cyrillic" else _ENDINGS_LAT
    lowered = surface.lower()
    out: list[tuple[str, str]] = []
    for ending in endings:
        if lowered.endswith(ending):
            stem = surface[: -len(ending)]
            ending = surface[-len(ending) :]
            # латинская конвенция для чужих имён: Telegram'da, Watch'ning — апостроф уходит в окончание
            if stem and stem[-1] in _APOSTROPHES:
                stem, ending = stem[:-1], stem[-1] + ending
            if (
                len(_HAS_LETTER.findall(stem.split()[-1] if stem.split() else ""))
                >= _MIN_STEM_LETTERS
            ):
                out.append((stem, ending))
    return out


def attach_ending(stem: str, ending: str) -> str:
    """Приклеивает окончание оригинала к новой основе; дательный согласуется с последней буквой."""
    if not ending:
        return stem
    apostrophe = ""
    if ending[0] in _APOSTROPHES:
        apostrophe, ending = ending[0], ending[1:]
    if ending.lower() in _DATIVE:
        # после k/q дательный ассимилируется (eshikka, qishloqqa); с апострофом имя чужое: Facebook'ga
        last = "" if apostrophe else stem[-1].lower()
        cyr = ending.lower() in ("га", "ка", "қа")
        if cyr:
            base = "ка" if last == "к" else "қа" if last == "қ" else "га"
        else:
            base = "ka" if last == "k" else "qa" if last == "q" else "ga"
        ending = base.upper() if ending.isupper() else base
    return stem + apostrophe + ending


def transliterate_stem(stem: str, target: str) -> str | None:
    """Копия основы в другой письменности; None, если транслит ненадёжен."""
    normalized = prepare_input(stem, homoglyphs=False)
    if target == "cyrillic":
        if _FOREIGN_LATIN.search(normalized):
            return None
        pieces, _ = _lat2cyr_pieces(normalized)
    else:
        pieces = _cyr2lat_pieces(normalized)
    result = "".join(pieces)
    if classify_script(result) != target or not is_clean_mention(result):
        return None
    return result


def _is_plural(word: str) -> bool:
    """Однословное "Yaponlar"/"Японлар" — жители, а не место."""
    return word.lower().endswith(("lar", "лар"))


def common_words(records: Iterable[Record]) -> set[str]:
    """Слова, которые в текстах регулярно пишутся со строчной буквы (нарицательные)."""
    counts: Counter[str] = Counter()
    for record in records:
        for match in _WORD.finditer(record["text"]):
            word = match.group()
            if word[0].islower():
                counts[word.lower()] += 1
    return {word for word, count in counts.items() if count >= _COMMON_WORD_MIN_COUNT}


@dataclass(slots=True)
class MentionEntry:
    stem: str  # каноническое написание основы (самое частое в источнике)
    n_words: int
    script: str
    lang: str = "uz"
    source: str = ""


@dataclass(slots=True)
class MentionPool:
    entries: dict[str, list[MentionEntry]] = field(
        default_factory=dict
    )  # тип -> упоминания
    known_stems: set[str] = field(
        default_factory=set
    )  # lower(stem) всех чистых упоминаний
    known_last_words: set[str] = field(
        default_factory=set
    )  # lower(последнее слово) известных основ

    def is_known_stem(self, stem: str) -> bool:
        lowered = stem.lower()
        if lowered in self.known_stems:
            return True
        words = lowered.split()
        return len(words) > 1 and words[-1] in self.known_last_words

    def split_ending(self, surface: str) -> tuple[str, str]:
        """(основа, окончание); окончание только если основа известна или окончание однозначное."""
        for stem, ending in _candidate_splits(surface):
            if self.is_known_stem(stem):
                return stem, ending
            last_word = stem.split()[-1]
            if (
                ending.lstrip(_APOSTROPHES).lower() in _UNAMBIGUOUS_ENDINGS
                and len(_HAS_LETTER.findall(last_word)) >= _MIN_UNKNOWN_STEM_LETTERS
            ):
                return stem, ending
        return surface, ""

    def candidates(
        self, label: str, script: str, lang: str, exclude_stem: str
    ) -> list[MentionEntry]:
        """Той же письменности и языка; если таких нет — той же письменности любого языка."""
        excluded = exclude_stem.lower()
        same_script = [
            e
            for e in self.entries.get(label, [])
            if e.script == script and e.stem.lower() != excluded
        ]
        same_lang = [e for e in same_script if e.lang == lang]
        return same_lang or same_script

    def add(self, label: str, entry: MentionEntry) -> bool:
        entries = self.entries.setdefault(label, [])
        if any(
            e.stem.lower() == entry.stem.lower() and e.script == entry.script
            for e in entries
        ):
            return False
        entries.append(entry)
        self.known_stems.add(entry.stem.lower())
        if entry.n_words > 1:
            self.known_last_words.add(entry.stem.lower().split()[-1])
        return True

    def merge(self, other: "MentionPool") -> None:
        for label, entries in other.entries.items():
            for entry in entries:
                self.add(label, entry)
        self.known_stems |= other.known_stems
        self.known_last_words |= other.known_last_words

    def learn_stems(self, records: Iterable[Record]) -> None:
        """Упоминания records (обычно train) становятся известными основами: нужно, чтобы
        у заменяемых оригиналов отщеплялись окончания, даже если пул собран из других корпусов."""
        for surface in _clean_surfaces(records):
            lowered = surface.lower()
            self.known_stems.add(lowered)
            if " " in lowered:
                self.known_last_words.add(lowered.split()[-1])

    def stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for label, entries in self.entries.items():
            counts: Counter[str] = Counter(f"{e.lang}-{e.script[:3]}" for e in entries)
            out[label] = {"total": len(entries), **dict(sorted(counts.items()))}
        return out


def _clean_surfaces(records: Iterable[Record]) -> Counter[str]:
    surfaces: Counter[str] = Counter()
    for record in records:
        text = record["text"]
        for entity in record["entities"]:
            surface = text[entity["start"] : entity["end"]]
            if is_clean_mention(surface):
                surfaces[surface] += 1
    return surfaces


def build_mention_pool(
    records: list[Record],
    min_count: int = 2,
    min_type_purity: float = 0.8,
    reference_records: list[Record] | None = None,
    transliterate: bool = False,
    lang: Language | None = None,
    split_endings: bool = True,
    source: str = "",
) -> MentionPool:
    """records — источник упоминаний; reference_records (обычно train) — справочник:
    их упоминания расширяют список известных основ (чтобы отщеплять окончания у
    строк из маленького источника), а их тексты дают список нарицательных слов.
    lang — язык всех записей источника; None -> определяется по каждой записи."""
    reference = reference_records or []
    surfaces = _clean_surfaces(records)
    known = {s.lower() for s in surfaces}
    known |= {s.lower() for s in _clean_surfaces(reference)}
    generic = common_words(reference) if reference else set()
    pool = MentionPool(
        known_stems=known, known_last_words={s.split()[-1] for s in known if " " in s}
    )

    # основа -> (тип -> число упоминаний), основа -> написания, основа -> языки
    by_stem_label: dict[str, Counter[str]] = defaultdict(Counter)
    spellings: dict[str, Counter[str]] = defaultdict(Counter)
    langs: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        text = record["text"]
        record_lang = lang if lang is not None else detect_language(text)
        for entity in record["entities"]:
            surface = text[entity["start"] : entity["end"]]
            if not is_clean_mention(surface):
                continue
            stem, _ = pool.split_ending(surface) if split_endings else (surface, "")
            key = stem.lower()
            by_stem_label[key][entity["label"]] += 1
            spellings[key][stem] += 1
            langs[key][record_lang] += 1

    for key, label_counts in by_stem_label.items():
        total = sum(label_counts.values())
        if total < min_count:
            continue
        canonical = spellings[key].most_common(1)[0][0]
        words = canonical.split()
        if len(words) == 1 and _is_plural(canonical):
            continue
        if len(words) <= 2 and all(w.lower() in generic for w in words):
            continue  # "Boshliq", "Davlat rahbari", "Oliy Kengashi" — нарицательные, не имена
        script = classify_script(canonical)
        stem_lang = langs[key].most_common(1)[0][0]
        for label, count in label_counts.items():
            if count / total < min_type_purity:
                continue
            stem_text = canonical
            if label == "NAME":
                # "Prezident Shavkat Mirziyoyev", "Janob Zou" — титул в пул не идёт
                while (
                    len(stem_text.split()) > 1
                    and stem_text.split()[0].lower() in generic
                ):
                    stem_text = stem_text.split(maxsplit=1)[1]
            variants = [(stem_text, script)]
            if transliterate:
                other = "cyrillic" if script == "latin" else "latin"
                copy = transliterate_stem(stem_text, other)
                if copy is not None:
                    variants.append((copy, other))
            for stem, stem_script in variants:
                pool.add(
                    label,
                    MentionEntry(
                        stem=stem,
                        n_words=len(stem.split()),
                        script=stem_script,
                        lang=stem_lang,
                        source=source,
                    ),
                )
    for entries in pool.entries.values():
        entries.sort(key=lambda e: e.stem)
    return pool


def write_pool_jsonl(pool: MentionPool, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for label in sorted(pool.entries):
            for entry in pool.entries[label]:
                stream.write(
                    json.dumps({"label": label, **asdict(entry)}, ensure_ascii=False)
                    + "\n"
                )


def read_pool_jsonl(path: Path) -> MentionPool:
    """Пул, собранный build_mention_pool.py: строки {"label", "stem", "n_words", "script", "lang", "source"}."""
    pool = MentionPool()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.pop("label")
            pool.add(label, MentionEntry(**row))
    for entries in pool.entries.values():
        entries.sort(key=lambda e: e.stem)
    return pool


def read_gold_tsv(path: Path) -> list[Record]:
    """Mendeley "Uzbek NER Gold" (колонки Sentence, TokenOrder, Token, NER_Tag, pos; BIO).

    Токены склеиваются пробелами, PER/LOC/ORG -> NAME/GEO/ORG, остальные типы
    (MISC, TEMPORAL, ...) не размечаются. Апострофы приводятся к ʻ/ʼ как в train
    (длина сохраняется). Спан со строчной буквы — нарицательное, пропускается.
    """
    sentences: list[list[tuple[str, str]]] = []
    with path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter="\t")
        next(reader, None)
        current_key: object = object()
        current: list[tuple[str, str]] = []
        for row in reader:
            if len(row) < 4 or not row[2]:
                continue
            if row[0] != current_key:
                if current:
                    sentences.append(current)
                current, current_key = [], row[0]
            current.append((row[2], row[3] or "O"))
        if current:
            sentences.append(current)
    return tagged_sentences_to_records(sentences, "gold")


def tagged_sentences_to_records(
    sentences: list[list[tuple[str, str]]],
    hash_prefix: str,
    label_map: dict[str, str] = GOLD_LABEL_MAP,
) -> list[Record]:
    """(token, tag) в BIO или BIOES -> записи формата кейса: токены через пробел, спан
    по B/S и следующим I/E того же типа; спан со строчной буквы пропускается."""
    records: list[Record] = []
    for index, tokens in enumerate(sentences):
        text = prepare_input(" ".join(word for word, _ in tokens), homoglyphs=False)
        starts: list[int] = []
        position = 0
        for word, _ in tokens:
            starts.append(position)
            position += len(word) + 1
        entities: list[Entity] = []
        cursor = 0
        while cursor < len(tokens):
            prefix, _, raw_label = tokens[cursor][1].partition("-")
            label = label_map.get(raw_label)
            if prefix not in ("B", "S") or label is None:
                cursor += 1
                continue
            end_index = cursor
            while (
                prefix == "B"
                and end_index + 1 < len(tokens)
                and tokens[end_index + 1][1]
                in (
                    f"I-{raw_label}",
                    f"E-{raw_label}",
                )
            ):
                end_index += 1
                if tokens[end_index][1] == f"E-{raw_label}":
                    break
            start, end = starts[cursor], starts[end_index] + len(tokens[end_index][0])
            if text[start].isupper():
                entities.append({"label": label, "start": start, "end": end})
            cursor = end_index + 1
        if text.strip():
            records.append(
                {
                    "hash": f"{hash_prefix}-{index:06d}",
                    "text": text,
                    "entities": entities,
                }
            )
    return records
