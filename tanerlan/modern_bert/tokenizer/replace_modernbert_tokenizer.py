"""Пересадка произвольного byte-level BPE-токенизатора в RuModernBERT.

Из донора берётся только модель BPE — словарь и мерджи. Всё остальное от
базы: претокенизатор, постпроцессор с [CLS]/[SEP], имена спецтокенов,
tokenizer_config. Нормализатор базы дополняется NFC, nbsp → пробел и
удалением невидимых Cf. Спецтокены донора выбрасываются, на их id встают
спецтокены базы.

Эмбеддинги нового словаря: спецтокены — строки базы по имени; токены,
совпавшие с базой по строке, — копия; остальные — среднее эмбеддингов
кусков, на которые базовый токенизатор режет поверхностную форму токена.
Модель грузится с MLM-головой: head.dense/head.norm словаря не касаются и
сохраняются предобученными, decoder.bias переносится по той же схеме
copy/average, что и эмбеддинги.

Запуск:
    python replace_modernbert_tokenizer.py --donor rifkat/uztext-3Gb-BPE-Roberta --output model-uz
    python replace_modernbert_tokenizer.py --donor ./my-tokenizer --output model-uz
"""

from typing import Any, cast

import json
from pathlib import Path

import rich_click as click
import torch
from rich.console import Console
from rich.table import Table
from tokenizers import Tokenizer, models
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    TokenizersBackend,
)

BASE_REPO = "deepvk/RuModernBERT-base"
BASE_REVISION = "patched-tokenizer"

NBSP_PATTERN = "[\u00a0\u202f\u2007\u2009\u200a]"
CF_PATTERN = r"\p{Cf}"
DESTRUCTIVE_NORMALIZERS = (
    "Lowercase",
    "StripAccents",
    "NFD",
    "NFKC",
    "NFKD",
    "BertNormalizer",
)

click.rich_click.USE_MARKDOWN = True
console = Console()


# --- загрузка донора -----------------------------------------------------------


def load_donor(source: str) -> Tokenizer:
    """Репозиторий HF, директория с tokenizer.json, директория с vocab.json + merges.txt или сам файл."""
    path = Path(source)
    if path.is_file():
        return Tokenizer.from_file(str(path))
    if path.is_dir():
        if (path / "tokenizer.json").exists():
            return Tokenizer.from_file(str(path / "tokenizer.json"))
        if (path / "vocab.json").exists() and (path / "merges.txt").exists():
            return Tokenizer(
                models.BPE.from_file(str(path / "vocab.json"), str(path / "merges.txt"))
            )
        raise click.UsageError(
            f"В {path} нет ни tokenizer.json, ни пары vocab.json + merges.txt"
        )
    tokenizer = AutoTokenizer.from_pretrained(source)
    if not isinstance(tokenizer, TokenizersBackend):
        raise click.UsageError(
            f"{source} не поддерживается AutoTokenizer, используйте путь к файлу или HF-репозиторий"
        )
    return tokenizer.backend_tokenizer


def ensure_bpe(payload: dict[str, Any], label: str) -> None:
    model_type = payload["model"]["type"]
    if model_type != "BPE":
        raise NotImplementedError(
            f"{label}: поддерживается только BPE, получено {model_type}"
        )


def is_byte_level(payload: dict[str, Any]) -> bool:
    pre = payload.get("pre_tokenizer")
    return pre is not None and "ByteLevel" in json.dumps(pre)


# --- нормализатор -------------------------------------------------------------


def _normalizer_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    current = payload["normalizer"]
    if current is None:
        return []
    if current["type"] == "Sequence":
        return list(current["normalizers"])
    return [current]


def _has_replace(steps: list[dict[str, Any]], pattern: str) -> bool:
    return any(
        s["type"] == "Replace" and s["pattern"].get("Regex") == pattern for s in steps
    )


def complete_normalizer(payload: dict[str, Any]) -> list[str]:
    """Дописывает недостающие шаги к нормализатору базы, сохраняя её собственные."""
    steps = _normalizer_steps(payload)
    destructive = [s["type"] for s in steps if s["type"] in DESTRUCTIVE_NORMALIZERS]
    if destructive:
        raise ValueError(
            f"Нормализатор базы содержит {destructive}: регистр или диакритика будут потеряны"
        )
    added: list[str] = []
    if not any(s["type"] == "NFC" for s in steps):
        steps.append({"type": "NFC"})
        added.append("NFC")
    if not _has_replace(steps, NBSP_PATTERN):
        steps.append(
            {"type": "Replace", "pattern": {"Regex": NBSP_PATTERN}, "content": " "}
        )
        added.append("nbsp → space")
    if not _has_replace(steps, CF_PATTERN):
        steps.append(
            {"type": "Replace", "pattern": {"Regex": CF_PATTERN}, "content": ""}
        )
        added.append("drop Cf")
    payload["normalizer"] = {"type": "Sequence", "normalizers": steps}
    return added


# --- сборка нового tokenizer.json ---------------------------------------------


def _special_tokens(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in payload.get("added_tokens", []) if t.get("special")]


def transplant(
    base_payload: dict[str, Any], donor_payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Словарь и мерджи донора внутри оболочки базы. Возвращает JSON и новые id спецтокенов базы."""
    vocab: dict[str, int] = dict(donor_payload["model"]["vocab"])

    donor_specials = {t["content"] for t in _special_tokens(donor_payload)}
    donor_specials |= {
        t for t in vocab if t in ("<s>", "</s>", "<pad>", "<unk>", "<mask>")
    }
    freed = sorted(vocab.pop(t) for t in donor_specials if t in vocab)

    base_specials = _special_tokens(base_payload)
    next_id = max(vocab.values()) + 1
    special_ids: dict[str, int] = {}
    for entry in base_specials:
        name = entry["content"]
        if name in vocab:
            special_ids[name] = vocab[name]
            continue
        new_id = freed.pop(0) if freed else next_id
        if new_id == next_id:
            next_id += 1
        vocab[name] = new_id
        special_ids[name] = new_id

    merged = json.loads(json.dumps(base_payload))
    merged["model"] = dict(donor_payload["model"])
    merged["model"]["vocab"] = vocab
    merged["model"]["unk_token"] = base_payload["model"].get("unk_token")
    merged["added_tokens"] = [
        dict(entry, id=special_ids[entry["content"]]) for entry in base_specials
    ]

    post = merged.get("post_processor")
    if post and post.get("type") == "TemplateProcessing":
        for name, spec in post["special_tokens"].items():
            spec["ids"] = [special_ids[tok] for tok in spec["tokens"]]

    return merged, special_ids


# --- эмбеддинги ---------------------------------------------------------------


def init_embeddings(
    model: PreTrainedModel,
    base_tokenizer: TokenizersBackend,
    new_tokenizer: TokenizersBackend,
    special_ids: dict[str, int],
    pad_to_multiple_of: int,
) -> dict[str, int | str]:
    old_embeddings = cast(torch.nn.Embedding, model.get_input_embeddings())
    old = old_embeddings.weight.data.clone()
    output = cast(torch.nn.Linear | None, model.get_output_embeddings())
    old_bias = (
        output.bias.data.clone()
        if output is not None and output.bias is not None
        else None
    )
    base_vocab = base_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()

    model.resize_token_embeddings(
        len(new_tokenizer), pad_to_multiple_of=pad_to_multiple_of
    )
    new_embeddings = cast(torch.nn.Embedding, model.get_input_embeddings())
    weight = new_embeddings.weight.data
    output = cast(torch.nn.Linear | None, model.get_output_embeddings())
    bias = (
        output.bias.data
        if output is not None and output.bias is not None and old_bias is not None
        else None
    )

    copied = averaged = fallback = 0
    with torch.no_grad():
        if bias is not None:
            # fallback-токенам и паддинг-строкам — ноль вместо случайной инициализации
            bias.zero_()
        for token, new_id in new_vocab.items():
            if token in special_ids:
                weight[new_id] = old[base_vocab[token]]
                if bias is not None and old_bias is not None:
                    bias[new_id] = old_bias[base_vocab[token]]
                continue
            base_id = base_vocab.get(token)
            if base_id is not None:
                weight[new_id] = old[base_id]
                if bias is not None and old_bias is not None:
                    bias[new_id] = old_bias[base_id]
                copied += 1
                continue
            surface = new_tokenizer.convert_tokens_to_string([token])
            pieces = base_tokenizer(surface, add_special_tokens=False)["input_ids"]
            if pieces:
                weight[new_id] = old[pieces].mean(dim=0)
                if bias is not None and old_bias is not None:
                    bias[new_id] = old_bias[pieces].mean()
                averaged += 1
            else:
                fallback += 1

    tied = output is not None and output.weight.data_ptr() == weight.data_ptr()
    if output is not None and not tied:
        output.weight.data[: weight.shape[0]] = weight

    return {
        "old_rows": old.shape[0],
        "new_rows": weight.shape[0],
        "tokenizer_size": len(new_tokenizer),
        "specials": len(special_ids),
        "copied": copied,
        "averaged": averaged,
        "fallback": fallback,
        "output_embeddings": "tied"
        if tied
        else ("copied" if output is not None else "absent"),
        "decoder_bias": "transferred" if bias is not None else "absent",
    }


def update_config(
    model: PreTrainedModel, new_tokenizer: TokenizersBackend
) -> dict[str, int]:
    changed: dict[str, int] = {}
    for key, value in [
        ("pad_token_id", new_tokenizer.pad_token_id),
        ("bos_token_id", new_tokenizer.cls_token_id),
        ("eos_token_id", new_tokenizer.sep_token_id),
        ("cls_token_id", new_tokenizer.cls_token_id),
        ("sep_token_id", new_tokenizer.sep_token_id),
        ("mask_token_id", new_tokenizer.mask_token_id),
    ]:
        if hasattr(model.config, key) and value is not None:
            setattr(model.config, key, value)
            changed[key] = value
    return changed


# --- CLI ----------------------------------------------------------------------


@click.command()
@click.option(
    "--donor",
    required=True,
    help="Репозиторий HF, директория (tokenizer.json или vocab.json + merges.txt) или файл tokenizer.json.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Куда сложить модель, конфиг и токенизатор.",
)
@click.option(
    "--pad-to-multiple-of", type=click.IntRange(min=1), default=64, show_default=True
)
def main(donor: str, output: Path, pad_to_multiple_of: int) -> None:
    """Пересаживает byte-level BPE донора в RuModernBERT и инициализирует под него эмбеддинги."""
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION)
    base_tokenizer = cast(TokenizersBackend, base_tokenizer)
 
    base_payload = json.loads(base_tokenizer.backend_tokenizer.to_str())
    ensure_bpe(base_payload, "база")

    donor_backend = load_donor(donor)
    donor_payload = json.loads(donor_backend.to_str())
    ensure_bpe(donor_payload, "донор")
    if (
        is_byte_level(base_payload)
        and donor_payload.get("pre_tokenizer")
        and not is_byte_level(donor_payload)
    ):
        console.print(
            "[yellow]Донор не byte-level, а база byte-level: строки токенов несопоставимы, пересечение будет пустым[/]"
        )

    added_steps = complete_normalizer(base_payload)
    merged_payload, special_ids = transplant(base_payload, donor_payload)

    output.mkdir(parents=True, exist_ok=True)
    base_tokenizer.save_pretrained(output)
    Tokenizer.from_str(json.dumps(merged_payload)).save(str(output / "tokenizer.json"))
    new_tokenizer = AutoTokenizer.from_pretrained(output)
    new_tokenizer = cast(TokenizersBackend, new_tokenizer)

    probe = "Oʻzbekiston Respublikasi Toshkent shahri"
    encoded = new_tokenizer(probe)
    assert (
        encoded["input_ids"][0] == new_tokenizer.cls_token_id
        and encoded["input_ids"][-1] == new_tokenizer.sep_token_id
    )

    with console.status("Инициализирую эмбеддинги"):
        model = AutoModelForMaskedLM.from_pretrained(BASE_REPO, revision=BASE_REVISION)
        report = init_embeddings(
            model, base_tokenizer, new_tokenizer, special_ids, pad_to_multiple_of
        )
        config_changes = update_config(model, new_tokenizer)
        model.save_pretrained(output)

    (output / "replace_report.json").write_text(
        json.dumps(
            {
                "donor": donor,
                "normalizer_added": added_steps,
                "special_ids": special_ids,
                "config": config_changes,
                **report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    table = Table(title=f"{BASE_REPO}@{BASE_REVISION} ← {donor}")
    table.add_column("метрика")
    table.add_column("значение", justify="right")
    table.add_row(
        "словарь базы → новый", f"{len(base_tokenizer)} → {len(new_tokenizer)}"
    )
    table.add_row("строк эмбеддингов", f"{report['old_rows']} → {report['new_rows']}")
    table.add_row("спецтокены по имени", str(report["specials"]))
    table.add_row("скопировано по строке", str(report["copied"]))
    table.add_row("усреднено по разбору", str(report["averaged"]))
    table.add_row("без инициализации", str(report["fallback"]))
    table.add_row("выходные эмбеддинги", str(report["output_embeddings"]))
    table.add_row("decoder.bias", str(report["decoder_bias"]))
    table.add_row("нормализатор дополнен", ", ".join(added_steps) or "уже полный")
    console.print(table)
    console.print(f"проба: {new_tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")


if __name__ == "__main__":
    main()
