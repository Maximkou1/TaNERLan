"""Small local playground with a contract-compatible mock NER API."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import RootModel


ROOT = Path(__file__).parent
app = FastAPI(title="TaNERlan playground mock")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class Batch(RootModel[list[dict[str, Any]]]):
    pass


def validate_batch(payload: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not payload:
        raise HTTPException(400, "request must be a non-empty JSON array")
    result: list[dict[str, str]] = []
    hashes: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise HTTPException(400, f"request[{index}] must be an object")
        record_hash, text = item.get("hash"), item.get("text")
        if not isinstance(record_hash, str) or not record_hash:
            raise HTTPException(400, f"request[{index}].hash must be a non-empty string")
        if record_hash in hashes:
            raise HTTPException(400, f"request[{index}].hash must be unique")
        if not isinstance(text, str) or not text:
            raise HTTPException(400, f"request[{index}].text must be a non-empty string")
        hashes.add(record_hash)
        result.append({"hash": record_hash, "text": text})
    return result


# Deliberately simple deterministic data for UI development, not a NER model.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ORG", re.compile(r"\b(?:AI Talent Hub|OpenAI|Google|Microsoft|Yandex|Uzbekneftegaz)\b", re.I)),
    ("GEO", re.compile(r"\b(?:Ташкент(?:е|а)?|Самарканд(?:е|а)?|Бухар(?:е|ы)|Петербург(?:е|а)?|Москва|Узбекистан(?:е|а)?)\b", re.I)),
    ("NAME", re.compile(r"\b(?:Дмитрий Брекоткин|Алишер Навои|Иван Иванов|Ali|Toshkent)\b", re.I)),
)


def mock_entities(text: str) -> list[dict[str, int | str]]:
    candidates = [
        {"label": label, "start": match.start(), "end": match.end()}
        for label, pattern in RULES
        for match in pattern.finditer(text)
    ]
    # An actual API must not return duplicate/overlapping boundaries. Keeping this
    # invariant here lets the front-end be developed against realistic output.
    entities: list[dict[str, int | str]] = []
    for entity in sorted(candidates, key=lambda item: (int(item["start"]), -int(item["end"]))):
        if not entities or int(entity["start"]) >= int(entities[-1]["end"]):
            entities.append(entity)
    return entities


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/predict")
def predict(batch: Batch) -> dict[str, list[dict[str, Any]]]:
    records = validate_batch(batch.root)
    return {
        "data": [
            {"hash": item["hash"], "entities": mock_entities(item["text"])}
            for item in records
        ]
    }
