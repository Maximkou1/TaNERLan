# ModernBERT NER

Дообучение ModernBERT (после MLM с новым токенизатором) на exact-span NER.
Структура повторяет `tanerlan/modern_bert/mlm`: `config.py` -> `data_module.py` ->
`training_module.py` -> `train.py`, плюс `predict.py`.

```
ner/
  config.py              pydantic-конфиг эксперимента (configs/ner.yaml)
  labels.py              LabelSchema: типы сущностей из данных -> BIO-теги, label2id/id2label
  data/records.py        чтение/валидация jsonl (hash, text, entities)
  data/dataset_preparation.py  токенизация с оффсетами, BIO-выравнивание, кэш в ~/.cache/tanerlan/ner-datasets
  data/augmentation.py   транслитерация куска текста между кириллицей и латиницей с пересчётом spans
  data/collator.py       NerCollator: аугментация на лету, паддинг, текст/spans/оффсеты в батче
  data/sampler.py        WeightedSourceSampler: доли источников в эпохе, шардирование под DDP
  data/span_targets.py   для span-головы: слова из токенов и ленточная матрица меток спанов
  decoding.py            BIO: вероятности тегов -> слова -> constrained Viterbi -> символьные spans
  span_decoding.py       span-голова: вероятности спанов -> непересекающиеся spans (greedy | dp)
  models/span_ner.py     ModernBertForSpanNer: affine / biaffine / multi-head biaffine + CNN по спанам
  optim/losses/          focal loss (BIO), span cross-entropy с boundary smoothing
  metrics.py             SpanMetrics: P/R/F1 по типам, micro, macro, sentence accuracy (как evaluation/core.py)
  optim/param_groups.py  группы параметров: тушка и голова с разными lr/шедулерами
  training_module.py     LightningModule: общий каркас + BioTrainingModule / SpanTrainingModule
  train.py               запуск обучения, сохранение лучшего чекпоинта в HF-формате
  ensemble.py            ансамбль на канонической сетке слов: BIO-эмиссии, лента спанов, перенос выходов
  predict.py             NerPredictor (сервинг: тексты -> сущности, ансамбль, длинные документы, torch | OpenVINO) + CLI
  export_to_onnx.py      экспорт обученной модели (bio | span) в ONNX и OpenVINO IR
```

## Запуск

```bash
export PYTHONPATH=/workspace/uzbeki
# пул упоминаний для аугментации заменой сущностей (один раз; ~1 мин, HF-датасеты кэшируются)
python -m tanerlan.modern_bert.ner.data.build_mention_pool --output data/mention_pool.jsonl
python tanerlan/modern_bert/ner/train.py -c configs/ner-mmbert.yaml -e artifacts/experiments

python tanerlan/modern_bert/ner/predict.py \
    --test-path data/dev.jsonl \
    --model-dir artifacts/experiments/NerModernBertUz/<run>/hf_model \
    --model-dir artifacts/experiments/NeRmmBERTBaseUZ/<run>/hf_model \   # ансамбль: повторить --model-dir
    --output artifacts/predictions/dev.jsonl
python evaluation/evaluate_model.py --gold data/dev.jsonl --predictions artifacts/predictions/dev.jsonl
```

## Инференс: NerPredictor

```python
from tanerlan.modern_bert.ner.predict import NerPredictor

predictor = NerPredictor.from_pretrained(["<run1>/hf_model", "<run2>/hf_model"], device="cuda")
entities = predictor.predict(["Toshkent shahar hokimligi ...", "..."])
# [[{"label": "ORG", "start": 0, "end": 25}, ...], ...] — координаты в исходном тексте
```

Внутри: `prepare_input` (нормализация с сохранением длины) -> токенизация каждой
моделью своим токенизатором -> forward -> декодер -> сущности. CLI выше — обёртка
над этим классом.

- **Одна модель** декодируется ровно так же, как на валидации при обучении
  (проверено: tp/fp/fn совпадают с `SpanMetrics` до единицы).
- **Ансамбль** (`weights` — веса моделей). Слова у разных токенизаторов могут
  расходиться там, где токен склеивает букву с пунктуацией, поэтому выходы
  переносятся на каноническую разбивку по тексту (`ensemble.py`): если все модели
  BIO — усредняются пословные вероятности тегов и идёт один Viterbi; если есть
  span-модель — всё приводится к ленте спанов (BIO через
  `p = P(B-t) · Π P(I-t) · (1 − P(I-t) после)`), усредняется и декодируется
  span-декодером (`span_decoding`: greedy | dp). На первых 200 документах dev:
  ModernBERT-uz 0.842, mmBERT 0.854, их ансамбль 0.888.
- **Длинные документы**. Документ длиннее `max_position_embeddings` идёт
  отдельным батчем целиком: окон нет, модель видит весь документ; предел —
  `max_length` (по умолчанию 4 × контекст), дальше обрезка с предупреждением.
  ModernBERT экстраполирует за контекст сам (RoPE theta 160k у global-слоёв,
  локальные окна у остальных): на склейках dev по 9–14k токенов mmBERT даёт
  F1 0.82–0.87 против 0.83–0.85 у тех же документов по отдельности, а
  dynamic-NTK на них столько же или хуже (14k: 0.80 против 0.82). Поэтому
  масштабирование переключается автоматически и только с порога
  `rope_scaling_threshold` (по умолчанию 2 × контекст): на такой батч
  `rope_type` global-слоёв ставится в `dynamic` (частоты пересчитываются под
  длину батча декоратором `dynamic_rope_update` из transformers, множитель
  `rope_scaling_factor`), после батча возвращаются `default` и исходные
  частоты. Ниже порога модель работает ровно как обучена.

## Что происходит с данными

1. Текст нормализуется `prepare_input` (длина сохраняется, оффсеты остаются
   в координатах исходного текста), токенизируется с `return_offsets_mapping`.
2. У токенов обрезаются ведущие пробелы (ByteLevel клеит пробел к слову),
   после чего границы сущностей совпадают с границами токенов: в train 17
   из 66k сущностей не совпадают (`".,"` слился в один токен), в dev — 0.
   Статистика (`entities_misaligned_with_tokens`, `entities_lost_by_truncation`,
   `truncated_records`, распределение длин и письменностей) логируется и
   пишется в `dataset_stats.json` рядом с чекпоинтами.
3. Токен получает `B-X`, если он первый пересекающийся с сущностью, `I-X` для
   остальных; спец-токены и пробельные токены -> `-100`.
4. В датасете остаются `text`, `entities`, `offsets`: коллатор отдаёт их в
   батч, метрики считаются по символьным spans против исходной разметки.
5. Аугментация (`data.augmentation`, одинаковая во всех трёх NER-конфигах;
   три независимых шага в этом порядке, пример токенизируется заново только
   если текст изменился):
   - Замена упоминаний (`mention_replace_prob`): до `mention_max_per_doc`
     основ документа меняются на другие того же типа, той же письменности,
     что оригинал, и того же языка, что документ (`data/language.py`:
     uz / ru / en / other по буквам чужих алфавитов и служебным словам).
     Пул (`mention_pool_path`, jsonl) собирает
     `data/build_mention_pool.py` из внешних корпусов, которых модель не
     видит в train: Mendeley "Uzbek NER Gold", Kaggle courpusNER2015 и
     WikiANN uz для узбекского (латиница плюс транслит-копии в кириллицу),
     WikiNEuRal ru (только именительный падеж по pymorphy3, плюс копии в
     латинице) и WikiNEuRal en. Без пути пул строится из train-разметки.
     Все формы одной основы в документе (Toshkent, Toshkentda) получают одну
     замену, падежное окончание оригинала переносится на неё. Вес кандидата
     `слов ** mention_long_bias`, для ORG bias выше
     (`mention_long_bias_by_label`): recall на ORG из 3+ слов заметно ниже,
     чем на коротких.
   - Транслитерация: пример с кириллицей с вероятностью `cyr2lat_prob`
     переводится в латиницу, иначе пример с латиницей с вероятностью
     `lat2cyr_prob` в кириллицу. Переводится весь текст (`full_text_prob`)
     или кусок от `min_chunk_words` до `max_chunk_fraction` слов текста;
     spans пересчитываются. lat2cyr пропускает пример, если граница сущности
     попадает внутрь диграфа (sh, ch, oʻ, ...).
   - Регистр всего текста (`lower_prob`, `upper_prob`), посимвольно, длина
     сохраняется. `entity_lower_prob`/`entity_upper_prob` (регистр только
     спанов) оставлены в конфиге, но выключены: на dev они ухудшали и общий
     F1, и срезы строчных/капса.

## Модель и головы

Голова выбирается в yaml через `model.head.type`; всё остальное (данные,
аугментации, оптимизатор, метрики, экспорт в HF, `predict.py`) общее.

`type: bio` — стандартный `ModernBertForTokenClassification`: энкодер ->
`dense -> GELU -> LayerNorm` (веса `head.*` берутся из MLM-чекпоинта) ->
dropout -> `classifier`, BIO-тег на токен. Loss — `head.loss`: `ce` (с
`label_smoothing`) или `focal` (`focal_gamma`). Декодер `decoding.py`:
токены сворачиваются в слова, по словам идёт constrained Viterbi.

`type: span` — `ModernBertForSpanNer` (`models/span_ner.py`): те же энкодер и
`head.*`, затем токены сворачиваются в слова (`word_pooling: first | mean`),
каждое слово получает start/end-проекции размера `proj_size`, и каждый спан
(i, j) с j - i < `max_span_width` слов получает вектор размера `hidden_size`:

- `scorer: affine` — `Linear([start_i; end_j]) + emb(j - i)`;
- `scorer: biaffine` — плюс `start_i^T U end_j`; `num_heads > 1` делает U
  блочно-диагональной (multi-head biaffine, в num_heads раз меньше параметров);
- `cnn_depth > 0` — столько свёрточных блоков `cnn_kernel_size`² по сетке
  спанов (CNN-NER): спан видит соседние спаны, что помогает с границами.

Матрица спанов хранится лентой W × max_span_width, а не W × W: документы в
train достигают 3874 слов, сущности — 35 (99% ≤ 7). Классы спанов — типы + 1
(`O`), loss — cross-entropy по всем допустимым спанам; `boundary_smoothing`
(`epsilon`, `distance`) раздаёт долю вероятности gold-спана соседям по
границам (Zhu & Li, 2022). Декодер `span_decoding.py`: `greedy` (по убыванию
вероятности без пересечений) или `dp` (максимум суммы log-odds по
непересекающимся спанам). `data/span_targets.py` считает, сколько gold-спанов
не выразимы (длиннее ленты или за обрезкой) — в `dataset_stats.json` как
`span_entities_lost`.

`model.name_or_path` — HF-директория или Lightning `.ckpt` (MLM или NER):
веса под `model.` и `head.` подхватываются любой головой, остальное
инициализируется заново. `model.from_pretrained_kwargs` пробрасывается в
`from_pretrained` как есть (`classifier_dropout`, `attention_dropout`,
`mlp_dropout`, `embedding_dropout`, ...), метки уезжают как
`num_labels/id2label/label2id` и сохраняются в `config.json`, откуда
`predict.py` восстанавливает схему и тип головы (по `architectures`).
Неизвестный ключ в yaml — ошибка (`extra="forbid"`).

## Экспорт в ONNX / OpenVINO

```bash
python tanerlan/modern_bert/ner/export_to_onnx.py \
    --model-dir artifacts/experiments/<exp>/<run>/hf_model \
    --output artifacts/onnx/<name>            # + --compress-to-fp16, --no-openvino, --attn-implementation eager
```

Экспортируется вся модель до логитов через `torch.onnx.export(dynamo=True)`,
все размерности динамические: BIO `(input_ids, attention_mask) -> (B, T, tags)`,
span `(input_ids, attention_mask, word_index, word_mask) -> (B, W, K, C)`. Рядом
кладутся `config.json`, токенизатор и `export.json` (opset, версии, разница
логитов с torch на примерных входах), поэтому директория экспорта для
`NerPredictor`, `predict.py` и сервиса неотличима от HF-модели: голова
читается из `architectures` в `config.json`, бэкенд выбирается по наличию
`model.xml` (OpenVINO IR) или `model.onnx` (`--backend auto | torch | openvino`).
На dev (200 документов) OpenVINO даёт те же сущности, что torch fp32, для
обеих голов: 200/200 документов совпали, разница логитов ~1e-5. Opset 23
(по умолчанию) экспортирует attention одним опом `Attention`, и OpenVINO на
CPU (EPYC 7452, 32 ядра) обгоняет torch fp32 в 1.7–1.9 раза на средних и
длинных документах (300–800 токенов: 3.0 с против 5.6 с на 8 документов;
4180 токенов: 3.7 с против 6.2 с), на коротких паритет; в opset 18 attention
разложен на MatMul/Softmax и на длинных документах OpenVINO втрое медленнее.
`--openvino-threads` задаёт число потоков (на этой машине 16 быстрее 32 на
коротких батчах).

Выбор бэкенда (`--backend auto`): директория с экспортом на CPU исполняется
через OpenVINO, на CUDA — через ONNX Runtime (`model.ort.onnx`, opset 18:
CUDA-ядра ORT не знают `RotaryEmbedding`/`Attention` opset 23), директория
с HF-весами — через torch. Замер на полном dev (1500 документов, mmBERT,
RTX 3090, только предикт, батч 32, бюджет 8192 токенов):

| бэкенд | full dev |
|---|---|
| torch fp32 sdpa (dtype auto) | 11.8 с |
| torch bf16 flash-attention | 23.6 с |
| ONNX Runtime CUDA EP fp32 | 24.7 с |
| TensorRT fp16 | не быстрее ORT на средних/длинных документах, сборка движка 3 мин, теряет 1.6% решений на 4k токенов |

Поэтому на CUDA при HF-весах остаётся torch, `dtype auto` = fp32 (совпадает
с CPU до сущности, bf16 меняет 3 документа из 200). TensorRT остаётся
опцией (`--tensorrt`, группа зависимостей `tensorrt`). `--max-batch-tokens`
ограничивает батч произведением документов на длину: у разложенного
attention в ORT/OpenVINO память растёт как B × T², два документа по 8k
токенов в одном батче не влезают в 24 ГБ. Ограничение экспорта: в графе
исходные частоты RoPE, dynamic-NTK для документов длиннее порога доступен
только в torch-бэкенде.

## Метрики (TensorBoard)

- `train/loss`, `train/micro/f1`, `train/sentence_accuracy`, `train/aug_fraction`,
  `train/seq_len` — по шагам; `train_epoch/*` — по эпохе. Плюс метрика головы:
  `token_accuracy` (bio) или `gold_span_accuracy` (span: доля gold-спанов с верным argmax).
- `val/{GEO,NAME,ORG}/{precision,recall,f1}`, `val/micro/*`, `val/macro/*`,
  `val/sentence_accuracy`, `val/counts/{tp,fp,fn}`, `val/loss` и метрика головы.
  `val/micro/f1` считается тем же декодером и той же схемой, что
  `predict.py` + `evaluation/evaluate_model.py` (совпадение проверено по tp/fp/fn).
- `val/error_examples` — текстом примеры документов с ошибками.
- Чекпоинты и early stopping мониторят `val_micro_f1`.
