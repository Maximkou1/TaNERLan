# Локальный inference benchmark

Пакет запускает пять готовых NER checkpoint'ов. Дообучения здесь нет. Модель и
tokenizer загружаются только после явного запуска команды; при импорте ничего
не скачивается и не запускается.

## Зависимости

Для BERT/XLM-R используются зависимости проекта. Для GLiNER установите пакеты
вручную, если хотите запускать эти модели:

```powershell
uv pip install gliner
uv pip install gliner2
```

Эти команды могут скачать пакеты, но не checkpoint'ы. Перед ними активируйте
окружение проекта и проверьте CUDA-версию PyTorch.

## Smoke-test

Работайте из корня репозитория. Команда скачает и загрузит только BERTbek,
обработает 20 записей и сохранит временный результат:

```powershell
python -m evaluation.evaluate_models.run_model `
  --model bertbek-ner-uznews `
  --max-records 20 `
  --output artifacts/models/bertbek-ner-uznews/smoke_predictions.jsonl `
  --device cuda
```

Проверьте, что файл содержит 20 строк, hashes совпадают с входом, а
`text[start:end]` соответствует найденной сущности.

## Полный запуск одной модели

После smoke-test:

```powershell
python -m evaluation.evaluate_models.run_model `
  --model bertbek-ner-uznews `
  --output artifacts/models/bertbek-ner-uznews/dev_predictions.jsonl `
  --device cuda
```

Повторите команду для следующих значений `--model`:

```text
gliner-multi-v2.1
gliner2.5-multi-v1
xlm-roberta-base-ner-hrl
gliner2-multi-v1
```

Запускайте модели по одной, чтобы не держать несколько checkpoint'ов в VRAM.

## Оценка

После каждого полного inference запускайте официальный scorer:

```powershell
python -m evaluation.evaluate_model `
  --gold data/dev.jsonl `
  --predictions artifacts/models/bertbek-ner-uznews/dev_predictions.jsonl `
  --output artifacts/models/bertbek-ner-uznews/dev_metrics.json
```

Главная метрика — exact-span micro-F1. Также сохраняются метрики по ORG, NAME и
GEO, а `metadata.json` содержит время, скорость и peak VRAM.

## Важные особенности

Для GLiNER2.5 код требует character offsets. Если GLiNER2 возвращает только
тексты сущностей без offsets, запуск завершится ошибкой. Восстанавливать
координаты через `text.find()` намеренно запрещено: повторяющиеся упоминания
могут дать неправильный span.

GLiNER legacy ограничивает один вход 384 токенами. Runner поэтому автоматически
разбивает длинные документы на консервативные перекрывающиеся character-окна по
границам слов и переводит offsets обратно в координаты исходного текста. Для
этого runner не требует дополнительной загрузки tokenizer через Transformers.
Предупреждения о truncated sentence после обновления runner'а появляться не
должны.

Первый основной прогон выполняется со стандартным threshold `0.5`. Перебор
threshold — отдельный эксперимент и должен сохраняться в отдельной папке.

## Последовательный launcher

После проверки отдельных запусков все модели можно запустить последовательно:

```powershell
python -m evaluation.evaluate_models.run_all --device cuda
```

Launcher не вызывается автоматически.

## Сводка

После оценки всех завершившихся моделей:

```powershell
python -m evaluation.evaluate_models.collect_summary
```

Результат будет записан в `artifacts/summary.json`.

## Срезы по письменности

Для сохранённых полных предсказаний посчитать exact-span micro-F1 и статистику
tokenizer по срезам `Latin`, `Cyrillic` и `mixed`:

```powershell
uv run --no-sync python -m evaluation.evaluate_models.slice_metrics
```

Результат будет записан в `artifacts/script_slice_summary.json`. Длина включает
special tokens и считается без truncation; `UNK` содержит общее число
неизвестных токенов и число документов, в которых они встретились.
