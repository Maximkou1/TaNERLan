"""Script classification used to slice metrics and pick few-shot examples."""
from collections import Counter


def _char_script(c: str) -> str | None:
    o = ord(c)
    if 0x0400 <= o <= 0x04FF:
        return "cyr"
    if (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F):
        return "lat"
    return None


def script_of(text: str) -> str:
    """cyrillic / latin / mixed / other, matching the train-set audit buckets."""
    counts = Counter(_char_script(c) for c in text)
    lat, cyr = counts.get("lat", 0), counts.get("cyr", 0)
    if lat == 0 and cyr == 0:
        return "other"
    if lat > 0 and cyr > 0:
        return "mixed"
    return "latin" if lat else "cyrillic"
