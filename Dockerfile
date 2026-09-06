# NER-сервис (API.md): GET /healthz, POST /api/v1/predict на 0.0.0.0:8000.
#
#   docker build -t ner-uz-solution .
#   docker run --rm --gpus all -p 8000:8000 -v "$PWD/models:/app/models:ro" ner-uz-solution   # GPU
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" ner-uz-solution              # CPU, медленно
#
# Модели пока монтируются volume'ом в /app/models (см. models/README.md); сервис грузит
# все поддиректории с config.json как ансамбль: HF-веса — через torch, экспорт
# export_to_onnx.py (model.xml/model.onnx) — через OpenVINO на CPU. Во время работы
# контейнер ничего не скачивает (HF_HUB_OFFLINE). Зависимости ставятся uv по uv.lock:
# основная группа (torch cu130, transformers, kostyl, ...) + группа infer (litserve,
# openvino, onnxruntime-gpu) + группа flash (прибитая сборка flash-attn под torch 2.13 /
# cu130 / py3.13). Группа tensorrt (~2 ГБ) в образ не входит: по замерам TensorRT не
# быстрее torch bf16 + flash-attention, см. tanerlan/modern_bert/ner/README.md.
# Torch-колёса несут CUDA-рантайм с собой, поэтому базовый образ — обычный python.

# ---------- этап 1: зависимости ----------
FROM python:3.13-slim-bookworm AS deps

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock ./
# только зависимости (без самого проекта — он копируется как исходники ниже)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --group infer --group flash

# ---------- этап 2: рантайм ----------
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY --from=deps /opt/venv /opt/venv

# код: пакет tanerlan (predict.py, модели, декодеры, сервер) и augmentation (транслитерация/скрипт)
COPY augmentation/__init__.py augmentation/transliteration.py ./augmentation/
COPY tanerlan ./tanerlan
COPY API.md ./
# точка монтирования весов: docker run -v "$PWD/models:/app/models:ro"
RUN mkdir -p /app/models
VOLUME ["/app/models"]

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=10 \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2); assert r.status == 200"

# параметры ансамбля из замеров на dev: равные веса, дисконт класса 'нет сущности' 0.8
CMD ["python", "-m", "tanerlan.serving.server", \
     "--models-root", "/app/models", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--batch-size", "8", "--batch-timeout", "0.05", "--predict-batch-size", "16", \
     "--none-scale", "0.8"]
