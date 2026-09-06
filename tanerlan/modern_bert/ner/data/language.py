"""Эвристическое определение языка текста: uz / ru / en / other.

Нужно аугментации заменой упоминаний (data/augmentation.py): замена берётся из
пула того же языка, что документ, и той же письменности, что оригинал.
Узбекский бывает в двух письменностях, поэтому письменность считается отдельно
(augmentation.transliteration.classify_script), а язык — по буквам, которых нет
в чужом алфавите, и по служебным словам:

  кириллица: ў қ ғ ҳ — только узбекские, ы щ — только русские; иначе служебные
             слова (ва, билан, учун против и, в, на, что); буквы чужих кириллиц
             (ә ң ө ұ ү і һ ...) без узбекских и русских признаков — other;
  латиница:  буквы с диакритикой (ç ş ğ ö ü ı é ...) — не узбекский и не английский;
             иначе служебные слова (va, bilan, uchun против the, and, of).

Короткий текст без признаков считается узбекским: корпус узбекский.
"""

import re
from typing import Literal

Language = Literal["uz", "ru", "en", "other"]

_LETTERS = re.compile(r"[^\W\d_]+")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")
# без IGNORECASE: с ним "ı" и "ſ" в классе символов матчат обычные i и s
_LATIN_FOREIGN = re.compile(r"[çşğöüıéèêáàâíóòúñãõäßæøåœÇŞĞÖÜİÉÈÊÁÀÂÍÓÒÚÑÃÕÄÆØÅŒ]")
_UZ_CYRILLIC = re.compile(r"[ўқғҳЎҚҒҲ]")
_RU_CYRILLIC = re.compile(r"[ыщЫЩ]")
# буквы, которых нет ни в узбекской, ни в русской кириллице: казахский, якутский, украинский, сербский, ...
_CYRILLIC_FOREIGN = re.compile(r"[әңөұүіһҕҥїєґђћјљњџӓӧӱӹӥӣӯӑӗәӘҢӨҰҮІҺҔҤЇЄҐЂЋЈЉЊЏ]")
_APOSTROPHES = re.compile(r"[ʻʼ'’‘`]")

_UZ_LATIN_STOPWORDS = frozenset(
    [
        "va",
        "bilan",
        "uchun",
        "bu",
        "ham",
        "deb",
        "edi",
        "bolib",
        "bolgan",
        "uning",
        "shu",
        "esa",
        "kerak",
        "yoki",
        "lekin",
        "bir",
        "biz",
        "men",
        "ular",
        "yana",
        "keyin",
        "haqida",
        "boladi",
        "bolsa",
        "emas",
        "hamda",
        "qilib",
        "qilgan",
        "ular",
        "bolishi",
        "ammo",
        "yil",
    ]
)
_UZ_CYRILLIC_STOPWORDS = frozenset(
    [
        "ва",
        "билан",
        "учун",
        "бу",
        "ҳам",
        "деб",
        "эди",
        "бўлиб",
        "бўлган",
        "унинг",
        "шу",
        "эса",
        "керак",
        "ёки",
        "лекин",
        "бир",
        "биз",
        "мен",
        "улар",
        "яна",
        "кейин",
        "ҳақида",
        "бўлади",
        "бўлса",
        "эмас",
        "ҳамда",
        "қилиб",
        "қилган",
        "бўлиши",
        "аммо",
        "йил",
    ]
)
_RU_STOPWORDS = frozenset(
    [
        "и",
        "в",
        "на",
        "не",
        "что",
        "с",
        "по",
        "как",
        "это",
        "для",
        "он",
        "она",
        "они",
        "из",
        "к",
        "у",
        "от",
        "за",
        "но",
        "а",
        "же",
        "все",
        "так",
        "его",
        "был",
        "была",
        "только",
        "или",
        "если",
        "когда",
        "уже",
        "во",
        "до",
        "также",
        "при",
        "этом",
        "их",
        "мы",
        "вы",
        "который",
        "которые",
    ]
)
_EN_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "for",
        "with",
        "on",
        "that",
        "this",
        "it",
        "as",
        "are",
        "was",
        "be",
        "by",
        "at",
        "from",
        "or",
        "an",
        "not",
        "have",
        "has",
        "you",
        "we",
        "they",
        "will",
        "his",
        "her",
        "their",
        "been",
        "which",
        "who",
        "what",
        "when",
        "there",
    ]
)


def _words(text: str) -> list[str]:
    return [
        _APOSTROPHES.sub("", w).lower()
        for w in _LETTERS.findall(_APOSTROPHES.sub("", text))
    ]


def detect_language(text: str) -> Language:
    n_cyrillic = len(_CYRILLIC.findall(text))
    n_latin = len(_LATIN.findall(text))
    n_letters = sum(len(w) for w in _LETTERS.findall(text))
    if n_letters == 0:
        return "uz"
    if max(n_cyrillic, n_latin) < n_letters * 0.5:
        return "other"  # тайский, корейский, арабский, ...
    words = _words(text)

    if n_cyrillic >= n_latin:
        uz_words = sum(w in _UZ_CYRILLIC_STOPWORDS for w in words)
        ru_words = sum(w in _RU_STOPWORDS for w in words)
        if _CYRILLIC_FOREIGN.search(text) and uz_words == 0 and ru_words == 0:
            return (
                "other"  # казахский, якутский, украинский, ... (қ ғ есть и в казахском)
            )
        uz_score = 2 * len(_UZ_CYRILLIC.findall(text)) + uz_words
        ru_score = 2 * len(_RU_CYRILLIC.findall(text)) + ru_words
        if ru_score > uz_score:
            return "ru"
        return "uz"

    uz_score = sum(w in _UZ_LATIN_STOPWORDS for w in words)
    en_score = sum(w in _EN_STOPWORDS for w in words)
    if en_score > uz_score:
        return "en"
    if uz_score == 0 and _LATIN_FOREIGN.search(text):
        return "other"  # турецкий, азербайджанский, немецкий, испанский, ...
    return "uz"
