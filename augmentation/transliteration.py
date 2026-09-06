"""Uzbek Cyrillic <-> Latin transliteration with exact entity-offset remapping.

Used to augment training data by adding a script-flipped duplicate of every
record written in a single script (pure Cyrillic or pure Latin). Records that
mix both scripts (commonly Russian code-switching) are left alone: applying
one language's transliteration table to the other script's letters would
corrupt whichever language it does not belong to.

Apostrophe handling reuses ``tanerlan.modern_bert.tokenizer.tokenization_utils.prepare_input``
(``homoglyphs=False``), the normalization already used elsewhere in this repo
before tokenization. It rewrites the mix of apostrophe-like codepoints
(``'`` ``` `` ``’`` ``‘`` ``ʼ`` ...) found in ``data/train.jsonl`` into exactly
two canonical Unicode Lm characters depending on role:

- U+02BB MODIFIER LETTER TURNED COMMA (``ʻ``) right after o/O/g/G, where it is
  part of the letter (``oʻ`` = ``ў``, ``gʻ`` = ``ғ``);
- U+02BC MODIFIER LETTER APOSTROPHE (``ʼ``) between two other letters, the
  glottal stop (``ъ``, e.g. ``sanʼat``);
- left untouched when it is not adjacent to a letter on the left, i.e. an
  actual quotation mark.

This is length-preserving, so it never disturbs entity offsets on its own.

On the "letters with diacritics" question: Uzbek Latin has no precomposed
character for oʻ/gʻ. ``len("oʻ") == 2`` and ``unicodedata.normalize("NFC",
"oʻ") == "oʻ"`` -- it is always two separate codepoints (base letter + Lm
modifier), never merged by NFC. A scan of data/train.jsonl also found zero
combining diacritics (Unicode category Mn) attached to any Cyrillic or Latin
letter -- the only combining marks present belong to unrelated scripts
(Arabic vowel points, Thai, Devanagari) inside quoted foreign text. So
"normalizing diacritic letters to one format" reduces entirely to the
apostrophe normalization above; there is no separate combining-mark pass to
write.

Offset remapping: every Cyrillic letter maps to 1 or 2 Latin characters
(never zero, except the dropped soft sign "ь"), so Cyrillic -> Latin is
always alignment-safe -- an original character boundary always lands on a
boundary of the transliterated text. Latin -> Cyrillic collapses digraphs
("sh", "ch", "ts", "yo", "yu", "ya", "ye", "oʻ", "gʻ") into a single Cyrillic
letter, so a gold entity boundary that happens to fall *inside* a digraph
(e.g. between the "s" and the "h" of "sh") cannot be represented after
conversion. ``transliterate_record`` raises ``TransliterationSkipped`` for
that record instead of silently shifting the span; the caller counts and
skips it. This is rare in practice because label boundaries follow word/
morpheme edges (see LABELING_GUIDE.md), essentially never mid-digraph.

Neither direction is a linguistically perfect round trip -- Cyrillic е/э both
map to Latin "e" (merged, matching the real Uzbek Latin alphabet), and the
reverse greedy match is a heuristic that can occasionally mis-split adjacent
letters that happen to spell a digraph (e.g. literal "т"+"с" vs. "ц"). That is
an accepted approximation for augmentation purposes: the goal is plausible
script-flipped text with exactly correct entity spans, not a certified
transliteration standard.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

Direction = Literal["cyr2lat", "lat2cyr"]

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")

# Lowercase Cyrillic letter -> lowercase Latin string (0, 1 or 2 chars).
# Core Uzbek Cyrillic alphabet per the official 1995/2021 Cyrillic-Latin
# correspondence table. "е" is approximated as plain "e" here (the "ye" form
# used word-initially / after a vowel is handled separately in
# _cyr2lat_pieces, which has the neighbouring-character context this table
# does not). "ц"->"ts", "щ"->"sh" and "ы"->"i" are the standard approximations
# for letters that exist mainly in Russian loanwords.
CYR_TO_LAT: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": "ʼ", "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "oʻ", "қ": "q", "ғ": "gʻ", "ҳ": "h",
}
# Word-initial or after a vowel/ъ/ь, "е"/"Е" is "ye"/"Ye" instead of "e"/"E".
_VOWELS_OR_SIGNS_BEFORE_YE = set("аеёиоуыэюяъь")

# Greedy 2-character Latin digraphs -> lowercase Cyrillic letter. Checked
# before the single-character table below. Longest (2-char) patterns first.
LAT_TO_CYR_MULTI: dict[str, str] = {
    "yo": "ё", "yu": "ю", "ya": "я", "ye": "е",
    "sh": "ш", "ch": "ч", "ts": "ц",
    "oʻ": "ў", "gʻ": "ғ",
}
# Single Latin letter -> lowercase Cyrillic letter. "c" and "w" do not belong
# to the Uzbek Latin alphabet (foreign names/brands only); mapped to the
# closest Uzbek letter as a best-effort fallback.
LAT_TO_CYR_SINGLE: dict[str, str] = {
    "a": "а", "b": "б", "c": "ц", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "ҳ", "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "х", "y": "й", "z": "з",
    "ʻ": "ъ", "ʼ": "ъ",
}


class TransliterationSkipped(Exception):
    """Raised when a gold entity boundary cannot be safely remapped."""


def classify_script(text: str) -> Literal["cyrillic", "latin", "mixed", "other"]:
    """Classifies a record by which alphabet(s) its letters belong to."""

    has_cyrillic = bool(_CYRILLIC.search(text))
    has_latin = bool(_LATIN.search(text))
    if has_cyrillic and has_latin:
        return "mixed"
    if has_cyrillic:
        return "cyrillic"
    if has_latin:
        return "latin"
    return "other"


def _cyr2lat_pieces(text: str) -> list[str]:
    """Maps every character to its Latin output piece(s), original casing."""

    pieces: list[str] = []
    for index, ch in enumerate(text):
        low = ch.lower()
        mapped = CYR_TO_LAT.get(low)
        if mapped is None:
            pieces.append(ch)
            continue
        if low == "е":
            previous = text[index - 1].lower() if index > 0 else ""
            if index == 0 or previous in _VOWELS_OR_SIGNS_BEFORE_YE or not previous.isalpha():
                mapped = "ye"
        if ch.isupper() and mapped:
            mapped = mapped[0].upper() + mapped[1:]
        pieces.append(mapped)
    _uppercase_allcaps_words(text, pieces)
    return pieces


def _lat2cyr_pieces(text: str) -> tuple[list[str], list[bool]]:
    """Maps every character to its Cyrillic output piece; flags unsafe splits.

    ``boundary_unsafe[i]`` is True when position ``i`` (between original
    characters ``i - 1`` and ``i``) falls inside a source digraph that
    collapsed into a single Cyrillic letter, so no gold span may start or end
    there.
    """

    n = len(text)
    pieces: list[str] = [""] * n
    boundary_unsafe = [False] * (n + 1)
    index = 0
    while index < n:
        two = text[index : index + 2].lower() if index + 1 < n else None
        cyr = LAT_TO_CYR_MULTI.get(two) if two else None
        if cyr is not None:
            pieces[index] = cyr.upper() if text[index].isupper() else cyr
            pieces[index + 1] = ""
            boundary_unsafe[index + 1] = True
            index += 2
            continue
        ch = text[index]
        mapped = LAT_TO_CYR_SINGLE.get(ch.lower())
        pieces[index] = ch if mapped is None else (mapped.upper() if ch.isupper() else mapped)
        index += 1
    _uppercase_allcaps_words(text, pieces)
    return pieces, boundary_unsafe


def _uppercase_allcaps_words(source_text: str, pieces: list[str]) -> None:
    """Upper-cases output pieces of source words written fully in caps.

    Per-character casing above already produces the right result for
    lowercase and Title-case words (a digraph like "sh"/"ш" only needs its
    first letter capitalized). An ALL-CAPS source word needs every piece
    upper-cased, e.g. Cyrillic "ШАҲАР" -> Latin "SHAHAR", not "ShAhAr".
    """

    for match in _WORD.finditer(source_text):
        word = match.group()
        if word.isupper():
            start, end = match.span()
            for i in range(start, end):
                pieces[i] = pieces[i].upper()


def _build_offsets(pieces: list[str]) -> list[int]:
    offsets = [0] * (len(pieces) + 1)
    for i, piece in enumerate(pieces):
        offsets[i + 1] = offsets[i] + len(piece)
    return offsets


def transliterate_record(
    text: str,
    entities: list[dict],
    direction: Direction,
) -> tuple[str, list[dict]]:
    """Transliterates ``text`` and remaps ``entities`` (label/start/end dicts).

    Raises ``TransliterationSkipped`` if any entity boundary cannot be
    represented after conversion (only possible for ``lat2cyr``, when a
    boundary falls inside a collapsed digraph).
    """

    normalized = prepare_input(text, homoglyphs=False)
    if direction == "cyr2lat":
        pieces = _cyr2lat_pieces(normalized)
        boundary_unsafe = [False] * (len(normalized) + 1)
    else:
        pieces, boundary_unsafe = _lat2cyr_pieces(normalized)

    offsets = _build_offsets(pieces)
    new_text = "".join(pieces)
    new_entities: list[dict] = []
    for entity in entities:
        start, end = entity["start"], entity["end"]
        if boundary_unsafe[start] or boundary_unsafe[end]:
            raise TransliterationSkipped(
                f"entity {entity['label']}[{start}:{end}] falls inside a merged digraph"
            )
        new_start, new_end = offsets[start], offsets[end]
        assert 0 <= new_start < new_end <= len(new_text)  # noqa: S101 - internal invariant
        new_entities.append({"label": entity["label"], "start": new_start, "end": new_end})
    return new_text, new_entities


if __name__ == "__main__":
    # Lightweight self-check, mirroring the convention in
    # tanerlan/modern_bert/tokenizer/tokenization_utils.py rather than pulling in a test framework.
    cases: list[tuple[str, Direction, list[tuple[int, int, str]], str, list[tuple[int, int, str]]]] = [
        (
            "Тошкент вилояти",
            "cyr2lat",
            [(0, 7, "GEO")],
            "Toshkent viloyati",
            [(0, 8, "GEO")],
        ),
        (
            "ШАҲАР ҳокимligi",
            "cyr2lat",
            [(0, 5, "GEO")],
            "SHAHAR hokimligi",
            [(0, 6, "GEO")],
        ),
        (
            "O'zbekiston Respublikasi",
            "lat2cyr",
            [(0, 11, "GEO")],
            "Ўзбекистон Республикаси",
            [(0, 10, "GEO")],
        ),
        (
            "Farg'ona shahrida Aziz bilan",
            "lat2cyr",
            [(0, 8, "GEO"), (18, 22, "NAME")],
            "Фарғона шаҳрида Азиз билан",
            [(0, 7, "GEO"), (16, 20, "NAME")],
        ),
    ]
    for source_text, direction_, spans, expected_text, expected_spans in cases:
        entities_in = [{"label": label, "start": s, "end": e} for s, e, label in spans]
        got_text, got_entities = transliterate_record(source_text, entities_in, direction_)
        assert got_text == expected_text, f"{source_text!r}: {got_text!r} != {expected_text!r}"
        expected_entities = [{"label": label, "start": s, "end": e} for s, e, label in expected_spans]
        assert got_entities == expected_entities, f"{got_entities} != {expected_entities}"
        for entity in got_entities:
            mention = got_text[entity["start"] : entity["end"]]
            print(f"{source_text!r} -> {got_text!r}: {entity['label']}={mention!r}")

    try:
        transliterate_record("shahar", [{"label": "GEO", "start": 1, "end": 6}], "lat2cyr")
    except TransliterationSkipped:
        print("unsafe mid-digraph boundary correctly rejected")
    else:
        raise AssertionError("expected TransliterationSkipped")

    print("all self-checks passed")
