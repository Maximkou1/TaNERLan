"""LLM-as-NER: харнесс для строгой span-метрики.

ГЛАВНОЕ ИНЖЕНЕРНОЕ РЕШЕНИЕ: у LLM НИКОГДА не спрашиваем start/end.
Модели не умеют считать символы -- они выдают правдоподобные, но неверные
числа, и при strict-метрике это гарантированный ноль. Просим ПОВЕРХНОСТНЫЕ
СТРОКИ, а координаты находим сами поиском по тексту (align_surfaces).

Второе решение: конвенция границ передаётся в промпт как ЖЁСТКОЕ правило,
выведенное из аудита train (см. build_system_prompt), а не как пожелание.
"""
from __future__ import annotations
import json, re, unicodedata
from collections import Counter

LABELS = ("GEO", "NAME", "ORG")


# --------------------------------------------------------------- промпт
def build_system_prompt(convention: dict) -> str:
    """convention -- факты из аудита train, а не догадки.

    Ключи: suffix_included (bool), quote_included (bool),
           geotail_included (bool), title_included (bool).
    """
    rules = []
    if convention["suffix_included"]:
        rules.append(
            "- Uzbek case/possessive suffixes are PART of the entity. "
            "If the text says 'Toshkentda', output exactly 'Toshkentda' "
            "(NOT 'Toshkent'). Same for -ning, -ga, -ni, -dan, -dagi, -si "
            "and Cyrillic -да, -нинг, -га, -ни, -дан.")
    else:
        rules.append("- Strip Uzbek case suffixes: 'Toshkentda' -> 'Toshkent'.")
    if convention["geotail_included"]:
        rules.append(
            "- Administrative tails are PART of GEO: 'Farg'ona viloyati', "
            "'Andijon tumani', 'Toshkent shahri' -- include the tail word.")
    if not convention["quote_included"]:
        rules.append("- Do NOT include quotation marks: «Kun.uz» -> Kun.uz")
    else:
        rules.append("- Keep quotation marks if they wrap the name.")
    if not convention["title_included"]:
        rules.append("- Do NOT include titles in NAME: "
                     "'Prezident Mirziyoyev' -> 'Mirziyoyev'.")

    return f"""You extract named entities from Uzbek text (Latin or Cyrillic script, often mixed).

LABELS (exactly three):
- GEO  = geographic/administrative places: countries, regions, cities, districts, streets, canals
- NAME = person names, nicknames, stage names
- ORG  = organizations: companies, brands, media, ministries, agencies, banks, sports clubs, apps, social networks

BOUNDARY RULES (these are strict; the scorer requires character-exact match):
{chr(10).join(rules)}
- Copy the entity substring EXACTLY as it appears in the text, byte for byte,
  including the original apostrophe character (ʻ vs ' vs ‘) and capitalization.
- Do not translate, transliterate, normalize or fix spelling.
- If the same entity appears several times, list it once per occurrence.
- Text with no entities -> empty list.

OUTPUT: JSON only, no prose, no markdown fence:
{{"entities": [{{"text": "<exact substring>", "label": "GEO|NAME|ORG"}}]}}"""


def build_user_prompt(text: str, shots: list[tuple[str, list]] | None = None) -> str:
    parts = []
    for stext, sents in (shots or []):
        payload = {"entities": [{"text": s[0], "label": s[1]} for s in sents]}
        parts.append(f"TEXT:\n{stext}\n\nJSON:\n{json.dumps(payload, ensure_ascii=False)}")
    parts.append(f"TEXT:\n{text}\n\nJSON:")
    return "\n\n---\n\n".join(parts)


# ------------------------------------------------- строки -> координаты
def _norm_ap(s: str) -> str:
    """Апострофы к одному виду -- только для СРАВНЕНИЯ, длина сохраняется."""
    return s.translate({ord(c): "'" for c in "\u2018\u2019\u02bb\u02bc\u2032\u0060\u00b4"})


def align_surfaces(text: str, items: list[dict]) -> list[dict]:
    """Поверхностные строки от LLM -> символьные спаны в text.

    Стратегии по убыванию строгости:
      1) точное совпадение;
      2) совпадение с унифицированным апострофом (LLM часто печатает ' вместо ʻ);
      3) регистронезависимое.
    Каждое вхождение занимается один раз (LLM просили перечислять повторы).
    Строки, не найденные в тексте -> отбрасываются (это галлюцинация, и при
    strict-метрике она может только испортить precision).
    """
    tl, tn, tlow = text, _norm_ap(text), _norm_ap(text).lower()
    used: list[tuple[int, int]] = []
    out = []

    def free(s, e):
        return not any(not (e <= a or s >= b) for a, b in used)

    def find(hay, needle):
        start = 0
        while True:
            i = hay.find(needle, start)
            if i < 0:
                return -1
            if free(i, i + len(needle)):
                return i
            start = i + 1

    for it in items:
        surf = (it.get("text") or "").strip()
        lab = (it.get("label") or "").upper()
        if not surf or lab not in LABELS:
            continue
        i = find(tl, surf)
        if i < 0:
            i = find(tn, _norm_ap(surf))
        if i < 0:
            i = find(tlow, _norm_ap(surf).lower())
        if i < 0:
            continue                      # не нашли -> галлюцинация, дропаем
        s, e = i, i + len(surf)
        used.append((s, e))
        out.append({"start": s, "end": e, "type": lab})
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def parse_json_lenient(raw: str) -> list[dict]:
    """LLM охотно оборачивает JSON в ```json ... ``` и добавляет пролог."""
    if not raw:
        return []
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    body = m.group(1) if m else raw
    i, j = body.find("{"), body.rfind("}")
    if i < 0 or j < 0:
        return []
    try:
        d = json.loads(body[i:j + 1])
    except json.JSONDecodeError:
        # частая поломка: обрыв по max_tokens -> откатываемся к последней
        # ЗАВЕРШЁННОЙ сущности (body[i:j+1], j = последняя "}") и дописываем
        # закрытие массива/объекта; остальные суффиксы -- на случай обрыва
        # посреди самого значения (недописанная строка и т.п.).
        trimmed = body[i:j + 1].rstrip().rstrip(",")
        for frag, suffix in ((trimmed, "]}"), (body[i:], "]}"),
                              (body[i:], "\"}]}"), (body[i:], "}]}")):
            try:
                d = json.loads(frag + suffix); break
            except json.JSONDecodeError:
                continue
        else:
            return []
    ents = d.get("entities", d if isinstance(d, list) else [])
    return [e for e in ents if isinstance(e, dict)]


def predict_one(text: str, call_fn, system: str, shots=None) -> list[dict]:
    raw = call_fn(system=system, user=build_user_prompt(text, shots))
    return align_surfaces(text, parse_json_lenient(raw))


# ------------------------------------------------------- отбор few-shot
def pick_shots(records, k=4, seed=0):
    """Примеры, покрывающие все три класса, обе графики и пустой случай."""
    import random
    rng = random.Random(seed)
    from .boundary_kit.audit import script_of
    want = [("latin", {"GEO","NAME","ORG"}), ("cyrillic", {"GEO","ORG"}),
            ("latin", {"NAME"}), ("cyrillic", set())]
    shots = []
    for scr, need in want[:k]:
        pool = [r for r in records
                if script_of(r["text"]) == scr and 60 < len(r["text"]) < 400
                and {e["label"] for e in r["entities"]} == need]
        if pool:
            r = rng.choice(pool)
            shots.append((r["text"], [(r["text"][e["start"]:e["end"]], e["label"])
                                      for e in r["entities"]]))
    return shots
