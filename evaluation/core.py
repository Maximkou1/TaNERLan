"""Общая библиотека: типы, HTTP-контракт сервиса, JSONL I/O и exact-span scorer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LABELS = ("ORG", "NAME", "GEO")
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonObject = dict[str, Any]
EntityKey = tuple[str, int, int]


class ContractError(ValueError):
    """Ошибка обязательного HTTP-контракта."""


# --------------------------------------------------------------------------
# Валидация сущностей (общая для scorer и HTTP-контракта)
# --------------------------------------------------------------------------


def validate_entity(
    entity: Any,
    text_length: int,
    source: str,
    error: type[ValueError] = ValueError,
) -> EntityKey:
    """Проверяет обязательные поля и координаты одной сущности."""

    if not isinstance(entity, dict):
        raise error(f"{source}: entity must be an object")
    label = entity.get("label")
    start = entity.get("start")
    end = entity.get("end")
    if label not in LABELS:
        raise error(f"{source}: label must be one of {sorted(LABELS)}")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= text_length
    ):
        raise error(f"{source}: invalid character offsets")
    return label, start, end


def validate_entities(
    raw: Any,
    text_length: int,
    source: str,
    error: type[ValueError] = ValueError,
) -> set[EntityKey]:
    """Проверяет список сущностей и возвращает множество exact-span ключей."""

    if not isinstance(raw, list):
        raise error(f"{source}: entities must be an array")
    entities: set[EntityKey] = set()
    for index, entity in enumerate(raw):
        key = validate_entity(entity, text_length, f"{source}/entities[{index}]", error)
        if key in entities:
            raise error(f"{source}/entities[{index}]: duplicate entity")
        entities.add(key)
    return entities


# --------------------------------------------------------------------------
# HTTP-контракт сервиса
# --------------------------------------------------------------------------


def validate_timeouts(startup_timeout: float, request_timeout: float) -> None:
    """Проверяет таймауты запуска и отдельного HTTP-запроса."""

    if startup_timeout <= 0:
        raise ContractError("startup-timeout must be positive")
    if request_timeout <= 0:
        raise ContractError("request-timeout must be positive")


def normalize_base_url(url: str) -> str:
    """Нормализует и проверяет адрес HTTP-сервиса."""

    base_url = url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ContractError("url must start with http:// or https://")
    return base_url


def _decode_json(body: bytes, source: str) -> JsonValue:
    """Декодирует UTF-8 JSON с понятным сообщением об ошибке."""

    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{source}: response is not valid UTF-8 JSON: {error}") from error


def _request_json(
    url: str,
    *,
    method: str,
    timeout: float,
    payload: JsonValue = None,
) -> tuple[int, str, JsonValue | None]:
    """Выполняет HTTP-запрос и возвращает статус, content type и JSON."""

    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = Request(  # noqa: S310 - схема URL проверена в normalize_base_url
        url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL задан участником
            status = response.status
            content_type = response.headers.get_content_type()
            response_body = response.read()
    except HTTPError as error:
        status = error.code
        content_type = error.headers.get_content_type()
        response_body = error.read()
    except URLError as error:
        raise ContractError(f"{method} {url}: connection failed: {error.reason}") from error
    except TimeoutError as error:
        raise ContractError(f"{method} {url}: request timed out") from error

    decoded = _decode_json(response_body, f"{method} {url}") if response_body else None
    return status, content_type, decoded


def _validate_health(payload: JsonValue | None) -> None:
    """Проверяет успешный ответ `/healthz`."""

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ContractError('GET /healthz: expected JSON object {"status":"ok"}')


def wait_for_health(base_url: str, startup_timeout: float, request_timeout: float) -> None:
    """Ожидает доступности сервиса, допуская connection refused и HTTP 503."""

    url = f"{base_url}/healthz"
    deadline = time.monotonic() + startup_timeout
    last_error = "service did not answer"

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ContractError(f"GET /healthz did not become ready: {last_error}")
        try:
            status, content_type, payload = _request_json(
                url,
                method="GET",
                timeout=min(request_timeout, remaining),
            )
        except ContractError as error:
            last_error = str(error)
        else:
            if status == 200:
                if content_type != "application/json":
                    raise ContractError(
                        f"GET /healthz: expected application/json, got {content_type!r}"
                    )
                _validate_health(payload)
                return
            if status != 503:
                raise ContractError(f"GET /healthz: expected 200 or 503, got {status}")
            last_error = "HTTP 503 Service Unavailable"
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def _validate_predict(
    payload: JsonValue | None,
    inputs: list[dict[str, str]],
) -> tuple[list[JsonObject], int]:
    """Проверяет envelope, порядок документов и exact-span поля."""

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ContractError("POST /api/v1/predict: expected JSON object with data[]")
    results = payload["data"]
    if len(results) != len(inputs):
        raise ContractError("POST /api/v1/predict: data length differs from request batch length")

    entity_count = 0
    for index, (result, item) in enumerate(zip(results, inputs, strict=True)):
        source = f"POST /api/v1/predict/data[{index}]"
        if not isinstance(result, dict):
            raise ContractError(f"{source}: result must be an object")
        if result.get("hash") != item["hash"]:
            raise ContractError(f"{source}: hash or result order differs from request")
        entities = validate_entities(
            result.get("entities"),
            len(item["text"]),
            source,
            ContractError,
        )
        entity_count += len(entities)
    return results, entity_count


def predict_batch(
    base_url: str,
    inputs: list[dict[str, str]],
    request_timeout: float,
) -> tuple[list[JsonObject], int]:
    """Отправляет один батч и возвращает проверенные результаты и число сущностей."""

    if not inputs:
        raise ContractError("predict batch must not be empty")
    status, content_type, payload = _request_json(
        f"{base_url}/api/v1/predict",
        method="POST",
        timeout=request_timeout,
        payload=inputs,
    )
    if status != 200:
        raise ContractError(f"POST /api/v1/predict: expected 200, got {status}")
    if content_type != "application/json":
        raise ContractError(
            f"POST /api/v1/predict: expected application/json, got {content_type!r}"
        )
    return _validate_predict(payload, inputs)


# --------------------------------------------------------------------------
# JSONL I/O
# --------------------------------------------------------------------------


def read_jsonl(path: Path, kind: str) -> list[JsonObject]:
    """Читает непустой JSONL и проверяет уникальность hash."""

    records: list[JsonObject] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            record_hash = record.get("hash")
            if not isinstance(record_hash, str) or not record_hash:
                raise ValueError(f"{path}:{line_number}: hash must be a non-empty string")
            if record_hash in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record_hash}")
            seen_hashes.add(record_hash)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no {kind} records")
    return records


def write_metrics(path: Path, metrics: JsonObject) -> None:
    """Записывает JSON-отчёт с метриками."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Exact-span scorer
# --------------------------------------------------------------------------


def _gold_by_hash(records: list[JsonObject], path: Path) -> dict[str, JsonObject]:
    """Проверяет gold-тексты и индексирует их по hash."""

    result: dict[str, JsonObject] = {}
    for index, record in enumerate(records, start=1):
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{path}:{index}: gold text must be a string")
        entities = validate_entities(record.get("entities"), len(text), f"{path}:{index}")
        result[record["hash"]] = {"text": text, "entities": entities}
    return result


def load_gold(path: Path) -> tuple[list[JsonObject], dict[str, JsonObject]]:
    """Читает и проверяет gold, возвращая записи и индекс по hash."""

    records = read_jsonl(path, "gold")
    return records, _gold_by_hash(records, path)


def _predictions_by_hash(
    records: list[JsonObject],
    path: Path,
    gold: dict[str, JsonObject],
) -> dict[str, set[EntityKey]]:
    """Проверяет соответствие hash и координат предсказаний gold-текстам."""

    predicted_hashes = {record["hash"] for record in records}
    gold_hashes = set(gold)
    missing = sorted(gold_hashes - predicted_hashes)
    extra = sorted(predicted_hashes - gold_hashes)
    if missing or extra:
        raise ValueError(
            "gold/prediction hashes differ: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )

    result: dict[str, set[EntityKey]] = {}
    for index, record in enumerate(records, start=1):
        record_hash = record["hash"]
        gold_record = gold[record_hash]
        if "text" in record and record["text"] != gold_record["text"]:
            raise ValueError(f"{path}:{index}: prediction text differs from gold for {record_hash}")
        result[record_hash] = validate_entities(
            record.get("entities"),
            len(gold_record["text"]),
            f"{path}:{index}",
        )
    return result


def _metric_values(tp: int, fp: int, fn: int) -> JsonObject:
    """Вычисляет Precision, Recall и F1 из TP, FP и FN."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold": tp + fn,
        "predicted": tp + fp,
    }


def calculate_metrics(
    gold: dict[str, JsonObject],
    predictions: dict[str, set[EntityKey]],
) -> JsonObject:
    """Считает exact-span метрики по классам, micro и macro."""

    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for record_hash, gold_record in gold.items():
        gold_entities = gold_record["entities"]
        predicted_entities = predictions[record_hash]
        for label in LABELS:
            gold_label = {entity for entity in gold_entities if entity[0] == label}
            predicted_label = {entity for entity in predicted_entities if entity[0] == label}
            counts[label]["tp"] += len(gold_label & predicted_label)
            counts[label]["fp"] += len(predicted_label - gold_label)
            counts[label]["fn"] += len(gold_label - predicted_label)

    by_label = {
        label: _metric_values(values["tp"], values["fp"], values["fn"])
        for label, values in counts.items()
    }
    micro = _metric_values(
        sum(values["tp"] for values in counts.values()),
        sum(values["fp"] for values in counts.values()),
        sum(values["fn"] for values in counts.values()),
    )
    macro = {
        metric: sum(by_label[label][metric] for label in LABELS) / len(LABELS)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "schema_version": 1,
        "matching": "same hash and exact label/start/end",
        "records": len(gold),
        "by_label": by_label,
        "micro": micro,
        "macro": macro,
    }


def evaluate_files(gold_path: Path, predictions_path: Path) -> JsonObject:
    """Валидирует два JSONL-файла и рассчитывает exact-span метрики."""

    _, gold = load_gold(gold_path)
    predictions = _predictions_by_hash(
        read_jsonl(predictions_path, "prediction"),
        predictions_path,
        gold,
    )
    return calculate_metrics(gold, predictions)


def print_metrics(metrics: JsonObject) -> None:
    """Печатает компактную таблицу основных метрик."""

    header = (
        f"{'scope':<8} {'precision':>10} {'recall':>10} {'f1':>10} {'tp':>8} {'fp':>8} {'fn':>8}"
    )
    print(header)
    print("-" * len(header))
    for label in LABELS:
        values = metrics["by_label"][label]
        print(
            f"{label:<8} {values['precision']:>10.4f} {values['recall']:>10.4f} "
            f"{values['f1']:>10.4f} {values['tp']:>8} {values['fp']:>8} {values['fn']:>8}"
        )
    micro = metrics["micro"]
    print(
        f"{'micro':<8} {micro['precision']:>10.4f} {micro['recall']:>10.4f} "
        f"{micro['f1']:>10.4f} {micro['tp']:>8} {micro['fp']:>8} {micro['fn']:>8}"
    )
    macro = metrics["macro"]
    print(
        f"{'macro':<8} {macro['precision']:>10.4f} {macro['recall']:>10.4f} "
        f"{macro['f1']:>10.4f} {'-':>8} {'-':>8} {'-':>8}"
    )
