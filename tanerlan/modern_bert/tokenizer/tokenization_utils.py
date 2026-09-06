import re
import unicodedata

MODIFIER_TURNED_COMMA = "\u02bb"
MODIFIER_APOSTROPHE = "\u02bc"

APOSTROPHE_VARIANTS = "\u0027\u0060\u00b4\u2018\u2019\u02b9\u02bb\u02bc\u2032"

_APOSTROPHE = re.compile(f"[{APOSTROPHE_VARIANTS}]")
_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

CYRILLIC_TO_LATIN = str.maketrans("аеорсхуАЕОРСХУ", "aeopcxyAEOPCXY")
LATIN_TO_CYRILLIC = str.maketrans("aeopcxyAEOPCXY", "аеорсхуАЕОРСХУ")


def _normalize_apostrophes(text: str) -> str:
    """Unify all apostrophe variants into Lm-category modifier letters.

    After o/g the apostrophe becomes U+02BB, which is part of the letter:
    oʻ = ў, gʻ = ғ. Between two other letters it becomes U+02BC, the glottal
    stop = ъ. An apostrophe with no letter on the left is left untouched:
    that is a quotation mark. Both target characters belong to \\p{L}, so the
    ByteLevel pre-tokenization regex does not split the word.
    """

    def replace(match: re.Match[str]) -> str:
        index = match.start()
        before = text[index - 1] if index > 0 else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        if not before.isalpha():
            return match.group(0)
        if before in "oOgG":
            return MODIFIER_TURNED_COMMA
        if after.isalpha():
            return MODIFIER_APOSTROPHE
        return match.group(0)

    return _APOSTROPHE.sub(replace, text)


def _script_profile(text: str) -> tuple[float, float]:
    """Return the fractions of Latin and Cyrillic among alphabetic characters."""
    latin = len(_LATIN.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    total = latin + cyrillic
    if total == 0:
        return 0.0, 0.0
    return latin / total, cyrillic / total


def _fix_homoglyphs(text: str) -> str:
    """Convert homoglyphs to the majority script within each word.

    Tоshkent with a Cyrillic о → Toshkent. This is a heuristic: enable it only
    after measuring the share of mixed-script words in the organizers' data.
    """

    def convert(match: re.Match[str]) -> str:
        word = match.group(0)
        latin_frac, cyrillic_frac = _script_profile(word)
        if latin_frac > cyrillic_frac:
            return word.translate(CYRILLIC_TO_LATIN)
        if cyrillic_frac > latin_frac:
            return word.translate(LATIN_TO_CYRILLIC)
        return word

    return _WORD.sub(convert, text)


def prepare_input(text: str, homoglyphs: bool = True) -> str:
    """Normalize text before the tokenizer. Guarantees the length is preserved."""
    normalized = _normalize_apostrophes(text)
    if homoglyphs:
        normalized = _fix_homoglyphs(normalized)
    if len(normalized) != len(text):
        raise ValueError("Normalized text has different length")
    return normalized


def trim_span_edges(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim invisible Cf-category characters from the edges of a predicted span.

    The tokenizer's normalizer removes Cf characters (ZWSP, ZWJ, etc.), but the
    offset mapping attributes a removed character to the following token, so a
    decoded span may start on a ZWSP instead of a letter. This function shrinks
    the span until both edges point at visible characters.

    Args:
        text: The full original document text (the same string the span
            offsets refer to), not the span substring itself.
        start: Span start, a character offset into ``text``.
        end: Span end, an exclusive character offset into ``text``.

    Returns:
        The adjusted ``(start, end)`` pair. If the span consists entirely of
        Cf characters, it collapses to an empty span (``start == end``).
    """
    while start < end and unicodedata.category(text[start]) == "Cf":
        start += 1
    while end > start and unicodedata.category(text[end - 1]) == "Cf":
        end -= 1
    return start, end


# if __name__ == "__main__":
#     cases = {
#         "o'zbekistonda": "oʻzbekistonda",
#         "o\u2019zbekistonda": "oʻzbekistonda",
#         "O`zbekiston": "Oʻzbekiston",
#         "G'ijduvon": "Gʻijduvon",
#         "bog'": "bogʻ",
#         "ma'no": "maʼno",
#         "san'at": "sanʼat",
#         "'Spendrups'": "'Spendrups'",
#         "«Kun.uz»": "«Kun.uz»",
#     }
#     for source, expected in cases.items():
#         got = prepare_input(source)
#         assert got == expected, f"{source!r}: ожидалось {expected!r}, получено {got!r}"
#         assert len(got) == len(source)

#     assert prepare_input("T\u043eshkent", homoglyphs=True) == "Toshkent"
#     assert prepare_input(prepare_input("o'zbek")) == prepare_input("o'zbek")

#     raw = "va \u200bToshkent shahri"
#     assert trim_span_edges(raw, 3, 12) == (4, 12)
#     assert raw[4:12] == "Toshkent"

#     print("все проверки пройдены")
