# TaNERLan

Named entity recognition for Uzbek text (Latin and Cyrillic script, including
mixed-script and code-switched documents). Extracts three entity types with
exact character-span boundaries:

| Label  | Meaning                                                    |
|--------|-------------------------------------------------------------|
| `ORG`  | organizations, companies, brands, media, agencies, clubs   |
| `NAME` | person names and unambiguous aliases                        |
| `GEO`  | countries, regions, cities, districts, named places          |

Metric: exact-span micro-F1 — an entity counts only if label, `start`, and
`end` all match the gold annotation exactly.

## Two model approaches

1. **Encoder ensemble** (`tanerlan/modern_bert/`) — token-classification
   (BIO + Viterbi decoding) or span-classification (biaffine/CNN scoring)
   heads on top of a pretrained encoder (mmBERT-base, or ModernBERT after
   domain-adapted MLM pretraining with a retrained tokenizer). Multiple
   fine-tuned checkpoints can be ensembled at inference time.
2. **LLM prompting** (`tanerlan/llm/`) — zero-shot extraction via an
   OpenAI-compatible chat endpoint (tested against vLLM). The prompt encodes
   the dataset's boundary conventions (case-suffix inclusion, administrative/
   institutional tail words, apostrophe normalization) as explicit rules;
   the model returns surface strings, and `align_surfaces` locates their
   exact character spans in the source text (an LLM cannot reliably count
   characters, so it is never asked for offsets directly).

## Repository layout

```
baseline/                 minimal token-classification baseline (train/predict)
evaluation/                exact-span micro-F1 scorer, shared by both approaches
tanerlan/
  modern_bert/
    mlm/                    domain-adaptive masked-LM pretraining
    ner/                    NER fine-tuning: data pipeline, BIO/span heads,
                             decoders, training loop, ensemble inference,
                             ONNX/OpenVINO export
    tokenizer/              custom tokenizer + Uzbek text normalization
  llm/                      prompt-tuned LLM-as-NER harness
  serving/                  HTTP service (LitServe) implementing the API below
augmentation/transliteration.py   Cyrillic<->Latin transliteration
                                  (imported by tanerlan/modern_bert/ner/data)
configs/                  training configs (yaml) for each model variant
data/
  train.jsonl / dev.jsonl  labeled Uzbek NER data
  mention_pool.jsonl       entity-replacement pool used by NER augmentation
```

## Data format

One JSON object per line:

```json
{"hash": "...", "text": "...", "entities": [{"label": "GEO", "start": 0, "end": 8}]}
```

## Training

Baseline:

```bash
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/baseline
python -m baseline.predict --model-dir artifacts/baseline/model --input data/dev.jsonl --output artifacts/baseline/dev_predictions.jsonl
```

Encoder ensemble member (mmBERT-base, BIO head by default; see
`configs/ner-mmbert-span.yaml` for the span head, `configs/ner-modern-bert-uz.yaml`
for ModernBERT):

```bash
export PYTHONPATH=$(pwd)
python -m tanerlan.modern_bert.ner.data.build_mention_pool --output data/mention_pool.jsonl
python tanerlan/modern_bert/ner/train.py -c configs/ner-mmbert.yaml -e artifacts/experiments

python tanerlan/modern_bert/ner/predict.py \
    --test-path data/dev.jsonl \
    --model-dir artifacts/experiments/<experiment>/<run>/hf_model \
    --model-dir artifacts/experiments/<other-experiment>/<run>/hf_model \
    --output artifacts/predictions/dev.jsonl
```

LLM prompting:

```bash
python -m tanerlan.llm.run_api --url <vllm-endpoint> --model <model-id> \
    --input data/dev.jsonl --output artifacts/llm/dev_predictions.jsonl
```

Evaluation (either approach, same metric):

```bash
python -m evaluation.evaluate_model --gold data/dev.jsonl --predictions <predictions.jsonl>
```

## Service

```bash
docker build -t ner-uz-solution .
docker run --rm --gpus all -p 8000:8000 -v "$PWD/models:/app/models:ro" ner-uz-solution   # GPU
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" ner-uz-solution              # CPU, slow
```

Model weights are not committed to this repository; mount a directory of one
or more HF checkpoints (or an `export_to_onnx.py` export) at `/app/models`.
Every subdirectory containing a `config.json` is loaded and ensembled
(alphabetical order); see `tanerlan/modern_bert/ner/predict.py`. The
container never downloads anything at runtime (`HF_HUB_OFFLINE=1`).

**API contract** — `GET /healthz` returns `{"status": "ok"}`; `POST /api/v1/predict`
accepts a batch of `{"hash": "...", "text": "..."}` objects and returns, for
each, `{"hash": "...", "entities": [{"label": "GEO", "start": 0, "end": 8}]}`
in the same order.

## Environment

Dependencies are pinned in `pyproject.toml` / `uv.lock` (managed with
[uv](https://docs.astral.sh/uv/)): `uv sync`. See `.env.example` for the
LLM-serving configuration variables (vLLM image/model, API key, rule set).
