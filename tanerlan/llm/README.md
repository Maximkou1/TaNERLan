# LLM-as-NER (промпт-подход, без дообучения)

Извлечение сущностей запросом к чат-модели (`GET /v1/chat/completions`,
OpenAI-совместимый API, тестировалось на vLLM) вместо token-classification.
Модель просят вернуть поверхностные строки, координаты находятся поиском по
исходному тексту -- LLM не считают символы, и просить у них `start`/`end`
напрямую гарантированно портит strict-span метрику.

Лучший замеренный результат (`cyankiwi/Qwen3.8-27B-AWQ-INT4`, zero-shot,
подобранный промпт, без дообучения) на полном dev (1500 док.,
`evaluation.evaluate_model`): **P=0.796 R=0.743 F1=0.768**. Подробный журнал
подбора промпта (9 попыток, что сработало и что нет) -- в
[PROMPT_TUNING.md](PROMPT_TUNING.md).

## Файлы

- `boundary_kit/audit.py` -- классификация графики текста (cyrillic/latin/
  mixed/other), используется и метрикой, и отбором few-shot примеров.
- `boundary_audit_train.md` -- аудит конвенций границ спанов в train (какие
  суффиксы/хвосты входят в сущность, варианты апострофа и т.п.) -- на его
  основе построены жёсткие правила промпта в `llm_ner.py`.
- `llm_ner.py` -- общий харнесс: сборка промпта из конвенций
  (`build_system_prompt`), выравнивание строк модели в char-спаны текста
  (`align_surfaces`, с толерантностью к вариантам апострофа), терпимый к
  обрывам JSON-парсер, отбор few-shot примеров.
- `run_api.py` -- раннер против OpenAI-совместимого чат-эндпоинта.
  Дополнительные правила промпта (`EXTRA_RULES`) подключаются через
  `--rules`, дефолт -- лучшая найденная комбинация.
- `scorer.py` / `eval_predictions.py` -- strict exact-span micro-F1 с
  разбивкой ошибок (boundary_only/type_only/spurious/missed) и по графике
  документа -- для отладки промпта во время итерации. Для официальной цифры
  используйте `evaluation.evaluate_model` (не валидирует так строго входные
  файлы, зато терпим к предсказаниям на подмножестве gold).
- `llm_candidates.csv` -- шортлист моделей ≤10B с оценкой покрытия
  узбекского языка (проверено по HF API, не по памяти).

## Запуск

Из корня репозитория:

```bash
python -m tanerlan.llm.run_api \
  --url http://<host>:<port> \
  --model <model-id> \
  --input data/dev.jsonl \
  --output artifacts/llm/dev_predictions.jsonl \
  --concurrency 4
```

Затем официальная метрика:

```bash
python -m evaluation.evaluate_model \
  --gold data/dev.jsonl \
  --predictions artifacts/llm/dev_predictions.jsonl \
  --output artifacts/llm/dev_metrics.json
```

Во время подбора промпта удобнее `eval_predictions` -- он не требует
предсказаний на всём gold и печатает разбор ошибок:

```bash
python -m tanerlan.llm.eval_predictions \
  --gold data/dev.jsonl \
  --predictions artifacts/llm/dev_predictions.jsonl \
  --show-errors 20
```

## Известные ограничения

- Сервер инференса (vLLM + AWQ-квантизация) не полностью детерминирован даже
  при `temperature=0` и заметно деградирует под чужой нагрузкой -- см.
  раздел "Инфраструктурные находки" в `PROMPT_TUNING.md` перед тем, как
  делать выводы по единственному прогону.
- Правила промпта подбирались по разбору ошибок на dev-подвыборке (не
  train) -- см. соответствующую оговорку там же; чистой out-of-sample
  оценки это не даёт, только полный прогон на 1500 dev как частичная
  проверка обобщения за пределы подвыборки.

## HTTP-сервис на LitServe

Сервис-адаптер принимает выданный контракт `/healthz` и `/api/v1/predict`, а
текущий backend отправляет запросы в удалённый OpenAI-compatible vLLM.

```bash
python -m tanerlan.serving.service
```

Для remote-режима обязательны `VLLM_BASE_URL` и `VLLM_MODEL`; скрытого адреса
или модели по умолчанию нет. Остальные настройки: `NER_BACKEND=remote-vllm`,
`VLLM_API_KEY`, `VLLM_TIMEOUT`, `VLLM_RETRIES`, `VLLM_CONCURRENCY`,
`VLLM_HEALTH_TIMEOUT`, `VLLM_HEALTH_CACHE_TTL`, `LLM_MAX_BATCH_ITEMS`,
`LLM_MAX_TOKENS`, `LLM_THINKING`, `LLM_RULES` и `PORT`.

`VLLM_TIMEOUT` — тайм-аут одной попытки upstream. При настройках по умолчанию
максимальное время ответа рассчитывается как `ceil(LLM_MAX_BATCH_ITEMS /
VLLM_CONCURRENCY) * (VLLM_TIMEOUT * VLLM_RETRIES + backoff) + 5s` и передаётся
в LitServe как его request timeout. Поэтому лимит батча и retry-политика не могут
молча расходиться; итоговый бюджет выводится в структурированном сообщении
конфигурации. При значениях по умолчанию это 2897 секунд (около 48 минут).

Слой LitServe не зависит от способа инференса. Для автономного режима нужно
реализовать backend с тем же интерфейсом `NERBackend` и выбрать его через
`NER_BACKEND=local`; публичный HTTP API менять не потребуется.

## Docker deployment

`docker-compose.yml` запускает два контейнера: API и локальный vLLM. API ждёт
прохождения healthcheck vLLM, поэтому не начнёт принимать запросы до загрузки
модели. Используемые образы и Python runtime закреплены версиями в
`Dockerfile.api` и `.env.example`; обновлять их нужно отдельным протестированным
изменением.

```bash
docker compose --env-file .env.production up -d --build
docker compose ps
```

Порт vLLM не публикуется наружу. Для проверки готовности используйте
`http://<host>:8000/healthz`; технические метрики доступны на `/metrics`.

Для локальной smoke-проверки на GPU с примерно 4 GB VRAM используйте отдельный
override с `Qwen/Qwen3-0.6B`. Он проверяет инфраструктуру, но не воспроизводит
качество production-модели 27B:

```bash
cp .env.local.example .env.local
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml up -d --build
```
