from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rich_click as click
from rich.console import Console
from rich.table import Table

from evaluation.core import JsonObject, load_gold, write_metrics

DEFAULT_TOKENIZERS: tuple[str, ...] = (
    "FacebookAI/xlm-roberta-large",
    "FacebookAI/xlm-roberta-base",
    "microsoft/mdeberta-v3-base",
    "jhu-clsp/mmBERT-base",
    "jhu-clsp/mmBERT-small",
    "Twitter/twhin-bert-base",
    "facebook/xlm-v-base",
    "google-bert/bert-base-multilingual-cased",
    "tahrirchi/tahrirchi-bert-base",
    "elmurod1202/bertbek-news-big-cased",
    "sinonimayzer/UzRoBERTa-v2",
    "rifkat/uztext-3Gb-BPE-Roberta",
    "DGurgurov/mbert_uzn-latn",
    "Davlan/xlm-roberta-base-ner-hrl",
    "Babelscape/wikineural-multilingual-ner",
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
    "deepvk/RuModernBERT-base",
)

WORD_RE = re.compile(r"\S+")
CYRILLIC_RANGE = (0x0400, 0x04FF)

# (ключ метрики, заголовок колонки, чем лучше: min или max, формат: ratio или percent)
METRIC_COLUMNS = (
    ("tokens_per_word", "tok/w↓", min, "ratio"),
    ("chars_per_token", "ch/t↑", max, "ratio"),
    ("intact_word_rate", "intact↑", max, "percent"),
    ("unk_rate", "UNK↓", min, "percent"),
    ("tokens_per_word_latin", "lat↓", min, "ratio"),
    ("tokens_per_word_cyrillic", "cyr↓", min, "ratio"),
    ("entity_boundary_rate", "ent↑", max, "percent"),
)


def _parse_tokenizers(raw: str | None) -> list[str]:
    """Разбирает список токенизаторов из CLI; DEFAULT разворачивается во встроенный shortlist."""

    if raw is None:
        return list(DEFAULT_TOKENIZERS)
    names: list[str] = []
    for name in (part.strip() for part in raw.split(",")):
        if not name:
            continue
        if name == "DEFAULT":
            names.extend(DEFAULT_TOKENIZERS)
        else:
            names.append(name)
    if not names:
        raise ValueError("tokenizers list must contain at least one name or path")
    return list(dict.fromkeys(names))


def _word_script(word: str) -> str:
    """Относит слово к латинице, кириллице, смеси или прочим символам."""

    has_latin = any("A" <= char <= "Z" or "a" <= char <= "z" for char in word)
    has_cyrillic = any(
        CYRILLIC_RANGE[0] <= ord(char) <= CYRILLIC_RANGE[1] for char in word
    )
    if has_latin and has_cyrillic:
        return "mixed"
    if has_latin:
        return "latin"
    if has_cyrillic:
        return "cyrillic"
    return "other"


def _prepare_documents(
    records: list[JsonObject],
    gold: dict[str, JsonObject],
) -> list[JsonObject]:
    """Заранее считает спаны слов, их письменность и gold-сущности."""

    documents: list[JsonObject] = []
    for record in records:
        text = record["text"]
        word_spans = [(match.start(), match.end()) for match in WORD_RE.finditer(text)]
        scripts = [_word_script(text[start:end]) for start, end in word_spans]
        documents.append(
            {
                "text": text,
                "word_spans": word_spans,
                "scripts": scripts,
                "entities": gold[record["hash"]]["entities"],
            }
        )
    return documents


def _load_tokenizer(name: str) -> Any:
    """Загружает fast tokenizer, необходимый для символьных offset_mapping."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("fast tokenizer with offset_mapping support is required")
    return tokenizer


def _evaluate_tokenizer(tokenizer: Any, documents: list[JsonObject]) -> JsonObject:
    """Считает fertility, UNK, целостность слов и совпадение границ сущностей."""

    token_total = char_total = unk_total = 0
    word_total = intact_words = 0
    script_tokens = {"latin": 0, "cyrillic": 0}
    script_words = {"latin": 0, "cyrillic": 0}
    entity_total = entity_aligned = 0
    unk_id = tokenizer.unk_token_id

    for document in documents:
        text = document["text"]
        encoding = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = encoding["input_ids"]
        token_spans = [
            (start, end) for start, end in encoding["offset_mapping"] if start < end
        ]
        token_total += len(ids)
        char_total += len(text)
        if unk_id is not None:
            unk_total += sum(1 for token_id in ids if token_id == unk_id)

        token_starts = {start for start, _ in token_spans}
        token_ends = {end for _, end in token_spans}
        for _, start, end in document["entities"]:
            entity_total += 1
            if start in token_starts and end in token_ends:
                entity_aligned += 1

        token_index = 0
        for (word_start, word_end), script in zip(
            document["word_spans"],
            document["scripts"],
            strict=True,
        ):
            while (
                token_index < len(token_spans)
                and token_spans[token_index][1] <= word_start
            ):
                token_index += 1
            count = 0
            cursor = token_index
            while cursor < len(token_spans) and token_spans[cursor][0] < word_end:
                count += 1
                cursor += 1
            word_total += 1
            if count == 1:
                intact_words += 1
            if script in script_tokens:
                script_tokens[script] += count
                script_words[script] += 1

    return {
        "vocab_size": len(tokenizer),
        "tokens": token_total,
        "tokens_per_word": token_total / word_total if word_total else None,
        "chars_per_token": char_total / token_total if token_total else None,
        "intact_word_rate": intact_words / word_total if word_total else None,
        "unk_rate": unk_total / token_total if token_total else None,
        "tokens_per_word_latin": (
            script_tokens["latin"] / script_words["latin"]
            if script_words["latin"]
            else None
        ),
        "tokens_per_word_cyrillic": (
            script_tokens["cyrillic"] / script_words["cyrillic"]
            if script_words["cyrillic"]
            else None
        ),
        "entity_boundary_rate": entity_aligned / entity_total if entity_total else None,
    }


def _format_metric(value: float | None, kind: str) -> str:
    """Форматирует ratio или percent, показывая прочерк для отсутствующих."""

    if value is None:
        return "—"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def _best_values(results: list[JsonObject]) -> dict[str, float]:
    """Находит лучшее значение каждой метрики среди успешных токенизаторов."""

    best: dict[str, float] = {}
    for key, _, better, _ in METRIC_COLUMNS:
        values = [
            result["metrics"][key]
            for result in results
            if result["ok"] and result["metrics"][key] is not None
        ]
        if values:
            best[key] = better(values)
    return best


def _build_table(results: list[JsonObject], data_path: Path) -> Table:
    """Собирает rich-таблицу с подсветкой лучших значений в колонках."""

    table = Table(title=f"Tokenization quality — {data_path.name}")
    table.add_column("tokenizer")
    table.add_column("vocab", justify="right")
    for _, header, _, _ in METRIC_COLUMNS:
        table.add_column(header, justify="right")

    best = _best_values(results)
    for result in results:
        if not result["ok"]:
            continue
        metrics = result["metrics"]
        cells = [result["name"], f"{metrics['vocab_size']:,}"]
        for key, _, _, kind in METRIC_COLUMNS:
            value = metrics[key]
            text = _format_metric(value, kind)
            if value is not None and best.get(key) == value:
                text = f"[bold green]{text}[/]"
            cells.append(text)
        table.add_row(*cells)
    return table


def run(args: SimpleNamespace) -> JsonObject:
    """Меряет качество токенизации списка токенизаторов и печатает отчёт."""

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    console = Console()

    tokenizers = _parse_tokenizers(args.tokenizers)
    data_path = args.data.expanduser().resolve()
    records, gold = load_gold(data_path)
    documents = _prepare_documents(records, gold)
    word_total = sum(len(document["word_spans"]) for document in documents)
    entity_total = sum(len(document["entities"]) for document in documents)
    console.print(
        f"Data: {data_path} — {len(documents)} documents, "
        f"{word_total} words, {entity_total} entities"
    )

    results: list[JsonObject] = []
    for name in tokenizers:
        with console.status(f"[bold]{name}[/]"):
            try:
                metrics = _evaluate_tokenizer(_load_tokenizer(name), documents)
            except Exception as error:  # noqa: BLE001 - hub/загрузка падают чем угодно
                results.append({"name": name, "ok": False, "error": str(error)})
                console.print(f"[red]FAIL[/] {name}: {error}")
                continue
        results.append({"name": name, "ok": True, "metrics": metrics})
        console.print(f"[green]OK[/]   {name}")

    if any(result["ok"] for result in results):
        console.print(_build_table(results, data_path))
    failures = [result for result in results if not result["ok"]]
    for failure in failures:
        console.print(f"[red]{failure['name']}: {failure['error']}[/]")

    report = {
        "schema_version": 1,
        "data": str(data_path),
        "documents": len(documents),
        "words": word_total,
        "entities": entity_total,
        "tokenizers": results,
    }
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        write_metrics(output_path, report)
        console.print(f"Report: {output_path}")
    return report


@click.command(
    help="Measure tokenization quality of candidate tokenizers on a JSONL subset.",
    context_settings={"show_default": True},
)
@click.option(
    "--data",
    "-d",
    type=click.Path(path_type=Path),
    required=True,
    help="JSONL в формате train/dev (hash, text, entities).",
)
@click.option(
    "--tokenizers",
    "-t",
    default=None,
    help="Имена или локальные пути токенизаторов через запятую; DEFAULT в списке разворачивается во встроенный shortlist, без опции берётся только он.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Куда записать JSON-отчёт.",
    required=True,
)
def main(**options: Any) -> None:
    """Запускает сравнение токенизаторов с компактным сообщением об ошибке."""

    try:
        run(SimpleNamespace(**options))
    except (OSError, TypeError, ValueError) as error:
        click.echo(f"ERROR: {error}", err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
