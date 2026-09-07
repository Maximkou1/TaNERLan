# NER service (see README.md): GET /healthz, POST /api/v1/predict on 0.0.0.0:8000.
#
#   docker build -t ner-uz-solution .
#   docker run --rm --gpus all -p 8000:8000 -v "$PWD/models:/app/models:ro" ner-uz-solution   # GPU
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" ner-uz-solution              # CPU, slow
#
# Models are mounted as a volume at /app/models (see README.md); the service
# loads every subdirectory containing a config.json as an ensemble: HF
# weights run through torch, an export_to_onnx.py export (model.xml/
# model.onnx) runs through OpenVINO on CPU. The container downloads nothing
# at runtime (HF_HUB_OFFLINE). Dependencies are installed by uv from
# uv.lock: the main group (torch cu130, transformers, ...) plus the infer
# group (litserve, openvino, onnxruntime-gpu) plus the flash group (a flash-
# attn build pinned to torch 2.13 / cu130 / py3.13). The tensorrt group
# (~2GB) is not included in the image: per measurements, TensorRT is not
# faster than torch bf16 + flash-attention here. Torch wheels bundle their
# own CUDA runtime, so the base image is a plain python image.

# ---------- stage 1: dependencies ----------
FROM python:3.13-slim-bookworm AS deps

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock ./
# dependencies only (the project itself is copied as source below)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --group infer --group flash

# ---------- stage 2: runtime ----------
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

# code: the tanerlan package (predict.py, models, decoders, server) and the
# transliteration helper it imports from augmentation
COPY augmentation/__init__.py augmentation/transliteration.py ./augmentation/
COPY tanerlan ./tanerlan
# mount point for weights: docker run -v "$PWD/models:/app/models:ro"
RUN mkdir -p /app/models
VOLUME ["/app/models"]

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=10 \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2); assert r.status == 200"

# ensemble defaults from measurements on dev: equal weights, 'no-entity' class discounted 0.8
CMD ["python", "-m", "tanerlan.serving.server", \
     "--models-root", "/app/models", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--batch-size", "8", "--batch-timeout", "0.05", "--predict-batch-size", "16", \
     "--none-scale", "0.8"]
