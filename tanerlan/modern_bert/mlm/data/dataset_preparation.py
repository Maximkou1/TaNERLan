"""Подготовка данных для MLM: загрузка источников, кэш токенизации, синтетический code-switching.

Кэш лежит в ~/.cache/tanerlan/tokenized-datasets. Ключ учитывает содержимое
источника, сплит, колонку текста, лимит и сам токенизатор: смена любого из
них пересобирает датасет, повторный запуск читает готовый arrow с диска.

Синтетический code-switching собирается из моноязычных источников по их
lang: каждый пример — узбекское предложение (латиница или кириллица, доля по
uz_*_ratio) с русской или английской примесью (доля по ru/en_ratio) — либо
чужое предложение целиком рядом, либо вставка из 1-4 слов внутрь.
"""

from typing import Literal, cast

import hashlib
from collections.abc import Callable
import json
import random
import re
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase
from kostyl.utils import DirLock

from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

from tanerlan.modern_bert.mlm.config import (
    LANG_IDS,
    CodeSwitchingDataConfig,
    DatasetConfig,
)

CACHE_ROOT = Path.home() / ".cache" / "tanerlan" / "tokenized-datasets"

Subset = Literal["train", "val"]
Lang = Literal["ru", "uz-lat", "uz-cyr", "en"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_POOL_TEXTS_CAP = 50_000
_MIN_SENTENCE_WORDS = 6
_MAX_SENTENCE_WORDS = 40
_VAL_FRACTION = 10  # валидации достаётся total_samples // 10 примеров


def _tokenizer_fingerprint(tokenizer: PreTrainedTokenizerBase) -> str:
    """Хэш содержимого токенизатора: другой словарь или нормализатор — другой кэш."""
    if getattr(tokenizer, "is_fast", False):
        payload = tokenizer.backend_tokenizer.to_str()
    else:
        payload = json.dumps(sorted(tokenizer.get_vocab().items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_raw_dataset(config: DatasetConfig, subset: Subset) -> Dataset:
    """Сырой датасет нужного сплита с применённым limit_samples."""
    split = config.train_split_name if subset == "train" else config.val_split_name
    raw = load_dataset(config.name_or_path, split=split)
    if config.limit_samples is not None:
        raw = raw.select(range(min(config.limit_samples, len(raw))))
    return raw


def _real_cache_payload(
    config: DatasetConfig, raw: Dataset, subset: Subset, tokenizer_fp: str
) -> dict[str, object]:
    """Fingerprint arrow-датасета меняется и при изменении данных, и при другом лимите."""
    return {
        "dataset": config.name_or_path,
        "subset": subset,
        "fingerprint": raw._fingerprint,
        "text_colname": config.text_colname,
        "limit_samples": config.limit_samples,
        "tokenizer": tokenizer_fp,
    }


def _tokenize_to_cache(
    raw: Dataset,
    text_colname: str,
    tokenizer: PreTrainedTokenizerBase,
    cache_path: Path,
    desc: str,
    lang_id: int,
) -> None:
    tokenized = raw.map(
        lambda batch, col=text_colname: {
            "input_ids": tokenizer(
                [prepare_input(text) for text in batch[col]], truncation=True
            )["input_ids"],
            "lang_id": [lang_id] * len(batch[col]),
        },
        batched=True,
        remove_columns=raw.column_names,
        desc=desc,
    )
    tokenized.save_to_disk(str(cache_path))


def _tokenize_with_cache(
    cache_payload: dict[str, object],
    build: Callable[[], tuple[Dataset, str]],
    tokenizer: PreTrainedTokenizerBase,
    desc: str,
    lang_id: int,
) -> Dataset:
    """Читает токенизированный датасет из кэша, при промахе строит через build.

    build отдаёт (сырой датасет, имя текстовой колонки) и вызывается только
    при промахе: генерация синтетики и нарезка пулов не выполняются зря.
    """
    cache_payload = {**cache_payload, "lang_id": lang_id, "normalize": "prepare_input"}
    key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    cache_path = CACHE_ROOT / key
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # Проверка и сборка под замком: иначе local ranks одновременно видят промах
    # и строят один и тот же кэш, читая при этом недописанную директорию.
    with DirLock(CACHE_ROOT):
        if not cache_path.exists():
            raw, text_colname = build()
            _tokenize_to_cache(raw, text_colname, tokenizer, cache_path, desc, lang_id)
    return cast(Dataset, load_from_disk(str(cache_path)))


def _prepare_real_dataset(
    config: DatasetConfig,
    raw: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_fp: str,
    subset: Subset,
) -> Dataset:
    return _tokenize_with_cache(
        _real_cache_payload(config, raw, subset, tokenizer_fp),
        lambda: (raw, config.text_colname),
        tokenizer,
        desc=f"Tokenizing {config.name_or_path} [{subset}]",
        lang_id=LANG_IDS[config.lang],
    )


# --- code-switching ------------------------------------------------------------


def _sentence_pool(raw: Dataset, text_colname: str) -> list[str]:
    """Предложения 3-40 слов из первых _POOL_TEXTS_CAP документов."""
    pool: list[str] = []
    capped = raw.select(range(min(len(raw), _POOL_TEXTS_CAP)))
    for text in capped[text_colname]:
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if _MIN_SENTENCE_WORDS <= len(sentence.split()) <= _MAX_SENTENCE_WORDS:
                pool.append(sentence)
    return pool


def _mix_pair(rng: random.Random, uz: str, foreign: str) -> str:
    """Межфразовое переключение (предложения рядом) или вставка 1-4 чужих слов внутрь."""
    if rng.random() < 0.5:
        pair = [uz, foreign]
        rng.shuffle(pair)
        return " ".join(pair)
    uz_words = uz.split()
    foreign_words = foreign.split()
    span = rng.randint(1, min(4, len(foreign_words)))
    start = rng.randrange(len(foreign_words) - span + 1)
    position = rng.randrange(1, len(uz_words))
    return " ".join(
        uz_words[:position] + foreign_words[start : start + span] + uz_words[position:]
    )


def _build_code_switching_texts(
    config: CodeSwitchingDataConfig,
    pools: dict[str, list[str]],
    total_samples: int,
    seed: int,
) -> list[str]:
    """total_samples смешанных примеров: узбекская основа + русская или английская примесь.

    Письменность основы выбирается по uz_lat_ratio/uz_cyr_ratio, язык примеси —
    по en_ratio/ru_ratio; доля материала каждого языка в корпусе следует ratio.
    """
    rng = random.Random(seed)

    def _weighted(weights: dict[str, float]) -> tuple[list[str], list[float]]:
        langs = [
            lang for lang, weight in weights.items() if weight > 0 and pools.get(lang)
        ]
        if not langs:
            missing = [lang for lang, weight in weights.items() if weight > 0]
            raise ValueError(
                f"Нет источников с предложениями для языков {missing}: "
                "добавьте DatasetConfig с соответствующим lang"
            )
        return langs, [weights[lang] for lang in langs]

    uz_langs, uz_weights = _weighted(
        {"uz-lat": config.uz_lat_ratio, "uz-cyr": config.uz_cyr_ratio}
    )
    foreign_langs, foreign_weights = _weighted(
        {"en": config.en_ratio, "ru": config.ru_ratio}
    )

    texts: list[str] = []
    for _ in range(total_samples):
        uz_lang = rng.choices(uz_langs, uz_weights)[0]
        foreign_lang = rng.choices(foreign_langs, foreign_weights)[0]
        texts.append(
            _mix_pair(rng, rng.choice(pools[uz_lang]), rng.choice(pools[foreign_lang]))
        )
    return texts


def _prepare_code_switching_dataset(
    config: CodeSwitchingDataConfig,
    source_configs: list[DatasetConfig],
    source_raws: list[Dataset],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_fp: str,
    subset: Subset,
    seed: int,
) -> Dataset:
    """Синтетический code-switching датасет с тем же дисковым кэшем, что и у реальных.

    Материал берётся из моноязычных источников по их lang. Ключ кэша включает
    fingerprint каждого источника, ratio, total_samples и seed, поэтому смена
    исходных данных или пропорций пересобирает синтетику. Валидации достаётся
    total_samples // 10 примеров.
    """
    total_samples = (
        config.total_samples
        if subset == "train"
        else max(config.total_samples // _VAL_FRACTION, 1)
    )
    cache_payload = {
        "code_switching": config.model_dump(),
        "sources": {
            c.lang: raw._fingerprint for c, raw in zip(source_configs, source_raws)
        },
        "subset": subset,
        "total_samples": total_samples,
        "seed": seed,
        "tokenizer": tokenizer_fp,
    }

    def build() -> tuple[Dataset, str]:
        pools: dict[str, list[str]] = {}
        for source_config, raw in zip(source_configs, source_raws):
            pools.setdefault(source_config.lang, []).extend(
                _sentence_pool(raw, source_config.text_colname)
            )
        texts = _build_code_switching_texts(config, pools, total_samples, seed)
        return Dataset.from_dict({"text": texts}), "text"

    return _tokenize_with_cache(
        cache_payload,
        build,
        tokenizer,
        desc=f"Tokenizing code-switching [{subset}]",
        lang_id=LANG_IDS["code-switching"],
    )


# --- внешний интерфейс ----------------------------------------------------------


def prepare_dataset(
    dataset_configs: list[DatasetConfig | CodeSwitchingDataConfig],
    tokenizer: PreTrainedTokenizerBase,
    subset: Subset,
    seed: int = 1337,
) -> Dataset:
    """Токенизированные датасеты из кэша: реальные источники плюс синтетический code-switching.

    Синтетика собирается из предложений реальных источников по их lang,
    поэтому CodeSwitchingDataConfig требует хотя бы по одному DatasetConfig
    на каждый язык с ненулевым ratio.
    """
    if not dataset_configs:
        raise ValueError("dataset_configs is empty")

    tokenizer_fp = _tokenizer_fingerprint(tokenizer)
    real_configs = [
        config
        for config in dataset_configs
        if not isinstance(config, CodeSwitchingDataConfig)
    ]
    cs_configs = [
        config
        for config in dataset_configs
        if isinstance(config, CodeSwitchingDataConfig)
    ]

    raws = [_load_raw_dataset(config, subset) for config in real_configs]
    parts = [
        _prepare_real_dataset(config, raw, tokenizer, tokenizer_fp, subset)
        for config, raw in zip(real_configs, raws, strict=True)
    ]
    for cs_config in cs_configs:
        parts.append(
            _prepare_code_switching_dataset(
                cs_config, real_configs, raws, tokenizer, tokenizer_fp, subset, seed
            )
        )
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]
