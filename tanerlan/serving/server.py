"""HTTP-сервис NER на LitServe по контракту API.md.

    python -m tanerlan.serving.server --models-root models \\
        --batch-size 8 --batch-timeout 0.05 --host 0.0.0.0 --port 8000

Модели — все поддиректории --models-root с config.json (одна или ансамбль, порядок по
имени; веса ансамбля — --weights в том же порядке). В Docker это volume /app/models.
--model-dir добавляет модель по пути или по имени репозитория HF Hub (скачивается
huggingface_hub в --models-root). Директория с model.xml/model.onnx (export_to_onnx.py)
исполняется через OpenVINO, с HF-весами — через torch (--backend).

Эндпоинты:
  GET  /healthz          -> {"status": "ok"} после загрузки моделей, 503 пока грузятся;
  POST /api/v1/predict   -> [{"hash", "text"}, ...] -> {"data": [{"hash", "entities"}, ...]}.

Инференс — NerPredictor (tanerlan/modern_bert/ner/predict.py): одна модель или
ансамбль, нормализация текста, токенизация, forward, декодер, координаты в
исходном тексте. Запрос целиком — один элемент очереди LitServe; --batch-size
задаёт, сколько одновременных запросов склеивается в один forward, а
--batch-timeout — сколько ждать их накопления. Размер батча самого forward —
--predict-batch-size. Невалидный запрос отбрасывается middleware с 4xx ещё до
очереди, чтобы не ронять склеенные с ним чужие запросы.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, override

import litserve as ls
import rich_click as click
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from litserve.utils import WorkerSetupStatus
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tanerlan.modern_bert.ner.data.records import Entity
from tanerlan.modern_bert.ner.predict import Backend, NerPredictor, parse_type_scales
from tanerlan.modern_bert.ner.span_decoding import SpanDecoding

logger = logging.getLogger("tanerlan.serving")

API_PATH = "/api/v1/predict"
HEALTH_PATH = "/healthz"
_MODEL_CONFIG_FILE = "config.json"


# --- запрос -----------------------------------------------------------------------------


@dataclass(slots=True)
class PredictRequest:
    hashes: list[str]
    texts: list[str]


class RequestValidationError(ValueError):
    pass


def parse_request(payload: Any) -> PredictRequest:
    """Проверяет тело POST /api/v1/predict по контракту: непустой массив {hash, text} с уникальными hash."""
    if not isinstance(payload, list):
        raise RequestValidationError("request body must be a JSON array")
    if not payload:
        raise RequestValidationError("request body must be a non-empty JSON array")
    hashes: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RequestValidationError(f"item {index}: must be an object")
        doc_hash, text = item.get("hash"), item.get("text")
        if not isinstance(doc_hash, str) or not doc_hash:
            raise RequestValidationError(f"item {index}: 'hash' must be a non-empty string")
        if not isinstance(text, str):
            raise RequestValidationError(f"item {index}: 'text' must be a string")
        if doc_hash in seen:
            raise RequestValidationError(f"item {index}: duplicate hash {doc_hash!r}")
        seen.add(doc_hash)
        hashes.append(doc_hash)
        texts.append(text)
    return PredictRequest(hashes=hashes, texts=texts)


class RequestValidationMiddleware:
    """Чистый ASGI-middleware: валидирует JSON тела POST {api_path} и отвечает 400 сам.

    LitServe при склейке батча отдаёт ошибку decode_request всем запросам батча, а
    невалидный JSON у него превращается в 500, поэтому проверка стоит до очереди.
    """

    def __init__(self, app: ASGIApp, api_path: str = API_PATH) -> None:
        self.app = app
        self.api_path = api_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != self.api_path:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)

        try:
            parse_request(json.loads(body.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            await self._reject(send, f"request body is not valid UTF-8 JSON: {error}")
            return
        except RequestValidationError as error:
            await self._reject(send, str(error))
            return

        # тело проверено как JSON: выставляем Content-Type, иначе LitServe без заголовка разбирает его как форму
        scope["headers"] = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-type"] + [
            (b"content-type", b"application/json")
        ]
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(send: Send, detail: str) -> None:
        response = JSONResponse(status_code=400, content={"detail": detail})
        await response({"type": "http"}, _noop_receive, send)


async def _noop_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


# --- LitAPI -----------------------------------------------------------------------------


@dataclass(slots=True)
class PredictorSettings:
    model_dirs: list[str]
    weights: list[float] | None
    dtype: str
    predict_batch_size: int
    max_batch_tokens: int
    max_length: int | None
    span_decoding: SpanDecoding
    none_scale: float
    type_scales: dict[str, float]
    rope_scaling_threshold: float | None
    rope_scaling_factor: float
    backend: Backend = "auto"
    openvino_device: str = "CPU"
    openvino_threads: int | None = None
    use_tensorrt: bool = False
    log_level: str = "info"


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)


class NerLitAPI(ls.LitAPI):
    """Один запрос (массив документов) — один элемент очереди; батч LitServe — список запросов."""

    def __init__(self, settings: PredictorSettings, max_batch_size: int, batch_timeout: float) -> None:
        super().__init__(max_batch_size=max_batch_size, batch_timeout=batch_timeout, api_path=API_PATH)
        self.settings = settings
        self.predictor: NerPredictor | None = None

    @override
    def setup(self, device: str) -> None:
        s = self.settings
        configure_logging(s.log_level)  # воркеры LitServe — spawn, конфиг логирования главного процесса не наследуется
        self.predictor = NerPredictor.from_pretrained(
            s.model_dirs,
            device=device,
            dtype=s.dtype,
            weights=s.weights,
            batch_size=s.predict_batch_size,
            max_batch_tokens=s.max_batch_tokens,
            max_length=s.max_length,
            span_decoding=s.span_decoding,
            none_scale=s.none_scale,
            type_scales=s.type_scales,
            rope_scaling_threshold=s.rope_scaling_threshold,
            rope_scaling_factor=s.rope_scaling_factor,
            backend=s.backend,
            openvino_device=s.openvino_device,
            openvino_threads=s.openvino_threads,
            use_tensorrt=s.use_tensorrt,
        )
        logger.info(f"NerPredictor ready on {device}: {len(s.model_dirs)} model(s), labels={self.predictor.schema.entity_types}")

    @override
    def decode_request(self, request: Request, context: dict[str, Any] | None = None, **kwargs: Any) -> PredictRequest:
        """Аннотация Request нужна LitServe: по ней он читает тело как JSON (иначе ждёт query-параметр);
        фактически сюда приходит уже разобранный JSON. hash'и уезжают в context: encode_response запроса не видит."""
        payload = cast(Any, request)
        try:
            parsed = parse_request(payload)
        except RequestValidationError as error:  # middleware уже проверил; на всякий случай
            raise HTTPException(status_code=400, detail=str(error)) from error
        if context is not None:
            context["hashes"] = parsed.hashes
        return parsed

    @override
    def batch(self, inputs: list[PredictRequest]) -> list[PredictRequest]:
        return inputs

    @override
    def predict(self, x: PredictRequest | list[PredictRequest], **kwargs: Any) -> Any:
        """Без склейки (max_batch_size=1) приходит один запрос, со склейкой — список; forward один на всех."""
        if self.predictor is None:
            raise RuntimeError("predictor is not set up")
        requests = x if isinstance(x, list) else [x]
        texts = [text for request in requests for text in request.texts]
        logger.debug(f"predict: {len(requests)} request(s) merged, {len(texts)} document(s)")
        entities = self.predictor.predict(texts)
        outputs: list[list[list[Entity]]] = []
        offset = 0
        for request in requests:
            outputs.append(entities[offset : offset + len(request.texts)])
            offset += len(request.texts)
        return outputs if isinstance(x, list) else outputs[0]

    @override
    def unbatch(self, output: list[list[list[Entity]]]) -> list[list[list[Entity]]]:
        return output

    @override
    def encode_response(self, output: list[list[Entity]], context: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]: 
        hashes = (context or {}).get("hashes")
        if hashes is None or len(hashes) != len(output):
            raise HTTPException(status_code=500, detail="response/request length mismatch")
        return {
            "data": [
                {"hash": doc_hash, "entities": [dict(e) for e in entities]}
                for doc_hash, entities in zip(hashes, output, strict=True)
            ]
        }


# --- сервер -----------------------------------------------------------------------------


def resolve_model(name_or_path: str, models_root: Path | None) -> str:
    """Локальная директория как есть; иначе репозиторий HF Hub: скачивается huggingface_hub
    (в models_root/<owner--repo>, если задан, иначе в кэш хаба) и используется локальная копия."""
    if Path(name_or_path).is_dir():
        return name_or_path
    from huggingface_hub import snapshot_download

    local_dir = models_root / name_or_path.replace("/", "--") if models_root is not None else None
    logger.info(f"{name_or_path} is not a local directory, downloading from HF Hub" + (f" to {local_dir}" if local_dir else ""))
    return snapshot_download(repo_id=name_or_path, local_dir=str(local_dir) if local_dir else None)


def discover_models(models_root: Path | None, model_dirs: Sequence[str]) -> list[str]:
    """Все поддиректории --models-root с config.json (по имени) плюс явные --model-dir
    (директории или репозитории HF Hub)."""
    found: list[str] = []
    if models_root is not None and models_root.is_dir():
        found.extend(str(p) for p in sorted(models_root.iterdir()) if (p / _MODEL_CONFIG_FILE).is_file())
    found.extend(resolve_model(entry, models_root) for entry in model_dirs)
    if not found:
        raise click.UsageError(
            f"no models: put HF model directories under {models_root} (see models/README.md) or pass --model-dir"
        )
    return found


def build_server(
    settings: PredictorSettings,
    max_batch_size: int,
    batch_timeout: float,
    accelerator: Literal["auto", "cpu", "cuda"],
    devices: int | Literal["auto"],
    workers_per_device: int,
    timeout: float,
) -> ls.LitServer:
    api = NerLitAPI(settings, max_batch_size=max_batch_size, batch_timeout=batch_timeout)
    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        timeout=timeout,
        middlewares=[(RequestValidationMiddleware, {"api_path": API_PATH})],
        model_metadata={"models": settings.model_dirs, "weights": settings.weights},
    )

    @server.app.get(HEALTH_PATH)
    async def healthz() -> JSONResponse:
        statuses = server.workers_setup_status
        ready = bool(statuses) and all(v == WorkerSetupStatus.READY for v in statuses.values())
        if ready:
            return JSONResponse(status_code=200, content={"status": "ok"})
        return JSONResponse(status_code=503, content={"status": "loading"})

    return server


@click.command(context_settings={"show_default": True})
@click.option("--models-root", type=click.Path(file_okay=False, path_type=Path), default=Path("models"), help="Грузятся все поддиректории с config.json (ансамбль в алфавитном порядке).")
@click.option("--model-dir", "model_dirs", multiple=True, help="Дополнительно: HF-директория или имя на HF Hub (для локальных проверок).")
@click.option("--weights", default=None, help="Веса моделей ансамбля через запятую.")
@click.option("--batch-size", type=int, default=8, help="Сколько одновременных запросов склеивать в один forward (LitServe max_batch_size).")
@click.option("--batch-timeout", type=float, default=0.05, help="Время накопления батча запросов, с.")
@click.option("--predict-batch-size", type=int, default=16, help="Размер батча forward внутри NerPredictor.")
@click.option("--max-batch-tokens", type=int, default=8192, help="Бюджет батча forward: документов x длина с паддингом.")
@click.option("--host", default="0.0.0.0")
@click.option("--port", type=int, default=8000)
@click.option("--accelerator", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
@click.option("--devices", default="1", help="Число устройств или auto.")
@click.option("--workers-per-device", type=int, default=1)
@click.option("--dtype", type=click.Choice(["auto", "float32", "bfloat16", "float16"]), default="auto")
@click.option("--max-length", type=int, default=None, help="Предел токенов на документ; по умолчанию 4 x контекст.")
@click.option("--span-decoding", type=click.Choice(["greedy", "dp"]), default="greedy")
@click.option("--none-scale", type=float, default=1.0, help="Дисконт класса 'нет сущности' в span-декодере.")
@click.option("--type-scale", "type_scales_raw", multiple=True, help="Множитель на тип при декодировании, напр. ORG=1.2; повторяемо.")
@click.option("--rope-scaling-threshold", type=float, default=2.0, help="dynamic-NTK RoPE включается на батч, если документ длиннее threshold x контекст; 0 = никогда.")
@click.option("--rope-scaling-factor", type=float, default=1.0, help="Множитель dynamic-NTK (factor).")
@click.option("--backend", type=click.Choice(["auto", "torch", "openvino", "onnxruntime"]), default="auto", help="auto: экспорт (model.xml/model.onnx) -> OpenVINO на CPU, ONNX Runtime на CUDA; HF-веса -> torch.")
@click.option("--tensorrt/--no-tensorrt", "use_tensorrt", default=False, help="ONNX Runtime: TensorRT EP (fp16) перед CUDA EP.")
@click.option("--openvino-device", default="CPU", help="Устройство OpenVINO (CPU, GPU, AUTO).")
@click.option("--openvino-threads", type=int, default=None, help="INFERENCE_NUM_THREADS для OpenVINO CPU; по умолчанию решает OpenVINO.")
@click.option("--timeout", type=float, default=120.0, help="Таймаут запроса в очереди LitServe, с.")
@click.option("--log-level", default="info")
def main(
    model_dirs: tuple[str, ...],
    models_root: Path | None,
    weights: str | None,
    batch_size: int,
    batch_timeout: float,
    predict_batch_size: int,
    max_batch_tokens: int,
    host: str,
    port: int,
    accelerator: Literal["auto", "cpu", "cuda"],
    devices: str,
    workers_per_device: int,
    dtype: str,
    max_length: int | None,
    span_decoding: Literal["greedy", "dp"],
    none_scale: float,
    type_scales_raw: tuple[str, ...],
    rope_scaling_threshold: float,
    rope_scaling_factor: float,
    backend: Literal["auto", "torch", "openvino", "onnxruntime"],
    use_tensorrt: bool,
    openvino_device: str,
    openvino_threads: int | None,
    timeout: float,
    log_level: str,
) -> None:
    """NER-сервис по контракту API.md: GET /healthz, POST /api/v1/predict."""
    configure_logging(log_level)
    models = discover_models(models_root, model_dirs)
    parsed_weights = [float(w) for w in weights.split(",")] if weights else None
    if parsed_weights is not None and len(parsed_weights) != len(models):
        raise click.BadParameter(f"{len(parsed_weights)} weights for {len(models)} models", param_hint="--weights")
    settings = PredictorSettings(
        model_dirs=models,
        weights=parsed_weights,
        dtype=dtype,
        predict_batch_size=predict_batch_size,
        max_batch_tokens=max_batch_tokens,
        max_length=max_length,
        span_decoding=span_decoding,
        none_scale=none_scale,
        type_scales=parse_type_scales(type_scales_raw),
        rope_scaling_threshold=rope_scaling_threshold if rope_scaling_threshold > 0 else None,
        rope_scaling_factor=rope_scaling_factor,
        backend=backend,
        openvino_device=openvino_device,
        openvino_threads=openvino_threads,
        use_tensorrt=use_tensorrt,
        log_level=log_level,
    )
    logger.info(f"Models: {models}; weights={parsed_weights}; batch={batch_size} x {batch_timeout}s; forward batch={predict_batch_size}")
    server = build_server(
        settings,
        max_batch_size=batch_size,
        batch_timeout=batch_timeout,
        accelerator=accelerator,
        devices="auto" if devices == "auto" else int(devices),
        workers_per_device=workers_per_device,
        timeout=timeout,
    )
    server.run(host=host, port=port, log_level=log_level, generate_client_file=False)


if __name__ == "__main__":
    main()
