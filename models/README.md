# Модели для сервиса

Сюда кладутся HF-директории моделей (`config.json`, `model.safetensors`,
токенизатор). Директория монтируется в контейнер как volume, сервис грузит все
поддиректории с `config.json` как ансамбль (в алфавитном порядке имён):

```bash
cp -r artifacts/experiments/BesTmmBertBioNer/hf_model                          models/1-mmbert-bio
cp -r artifacts/experiments/NeRmmBERTBaseUZSpanBiaffineCNN/<run>/hf_model       models/2-mmbert-span
cp -r artifacts/experiments/NeRModernBertUZNewDecoder/<run>/hf_model            models/3-modernbert-uz-bio

docker run --rm --gpus all -p 8000:8000 -v "$PWD/models:/app/models:ro" ner-uz-solution
```

Вместо HF-весов можно положить директорию экспорта `export_to_onnx.py`
(`model.xml/.bin`, `model.onnx`, `model.ort.onnx` + `config.json` +
токенизатор): на CPU сервис исполнит её через OpenVINO (в 1.6–1.9 раза
быстрее torch на CPU), на CUDA — через ONNX Runtime. Для CUDA выгоднее
оставить HF-веса: torch bf16 + flash-attention там быстрее всех вариантов.
Одной модели тоже достаточно. Веса и прочие параметры ансамбля задаются в
`CMD` Dockerfile (`--weights`, `--none-scale`, ...), см.
`python -m tanerlan.serving.server --help`. Сами веса в git не попадают.
