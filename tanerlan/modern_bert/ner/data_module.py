from pathlib import Path
from typing import Literal, override

import lightning as L
import torch
from datasets.arrow_dataset import Dataset
from kostyl.utils import setup_logger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from tanerlan.modern_bert.ner.config import (
    BioHeadConfig,
    DataConfig,
    HeadConfig,
    SpanHeadConfig,
)
from tanerlan.modern_bert.ner.data.collator import NerCollator
from tanerlan.modern_bert.ner.data.dataset_preparation import (
    dataset_stats,
    entity_type_counts,
    prepare_dataset,
)
from tanerlan.modern_bert.ner.data.mention_pool import (
    MentionPool,
    build_mention_pool,
    read_gold_tsv,
    read_pool_jsonl,
)
from tanerlan.modern_bert.ner.data.records import Record, read_records
from tanerlan.modern_bert.ner.data.sampler import WeightedSourceSampler, check_sources
from tanerlan.modern_bert.ner.data.span_targets import build_span_targets
from tanerlan.modern_bert.ner.labels import LabelSchema

logger = setup_logger(fmt="only_message")

_SUBSET_SEED_OFFSETS: dict[str, int] = {"train": 0, "val": 1}


def _subset_seed(seed: int, subset: Literal["train", "val"]) -> int:
    """Разные, но стабильные сиды для train и val (hash(str) рандомизирован per-process)."""
    return seed + _SUBSET_SEED_OFFSETS[subset]


def load_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer is None:
        raise ValueError(f"Tokenizer not found at {name_or_path}")
    if not tokenizer.is_fast:
        raise ValueError("NER requires a fast tokenizer with offset_mapping support")
    return tokenizer


# если model_max_length не задан в tokenizer_config.json (EuroBERT), transformers
# подставляет VERY_LARGE_INTEGER (1e30); такое значение не влезает в int32 внутри
# tokenizers.enable_truncation и роняет токенизацию с OverflowError
_MODEL_MAX_LENGTH_SANITY_LIMIT = 1_000_000


def model_max_positions(name_or_path: str) -> int | None:
    """max_position_embeddings из HF-конфига модели или Lightning-чекпоинта; None — если его там нет."""
    source = Path(name_or_path)
    if source.is_file() and source.suffix == ".ckpt":
        # map_location="meta": веса не материализуются, читаем только сохранённый конфиг
        checkpoint = torch.load(source, map_location="meta", weights_only=False)
        value = (checkpoint.get("config") or {}).get("max_position_embeddings")
    else:
        value = getattr(AutoConfig.from_pretrained(name_or_path), "max_position_embeddings", None)
    return int(value) if value else None


def resolve_max_length(
    tokenizer: PreTrainedTokenizerBase,
    configured: int | None,
    *model_sources: str | None,
) -> int:
    """Явный max_length -> tokenizer.model_max_length -> max_position_embeddings модели.

    model_sources перебираются по порядку (name_or_path модели, затем токенизатора)
    и нужны, только когда в tokenizer_config.json нет model_max_length.
    """
    if configured is not None:
        return configured
    tokenizer_limit = int(tokenizer.model_max_length)
    if tokenizer_limit <= _MODEL_MAX_LENGTH_SANITY_LIMIT:
        return tokenizer_limit
    candidates = list(dict.fromkeys(s for s in model_sources if s))  # порядок сохранён, дубли отброшены
    for candidate in candidates:
        positions = model_max_positions(candidate)
        if positions is not None:
            logger.info(f"tokenizer.model_max_length не задан; max_length = max_position_embeddings из {candidate}")
            return positions
    raise ValueError(
        f"Не удалось определить max_length: model_max_length нет в tokenizer_config.json "
        f"(transformers подставил {tokenizer_limit}), max_position_embeddings нет в конфигах "
        f"{candidates}. Задайте data.max_length в yaml."
    )


class NerDataModule(L.LightningDataModule):
    """Схема меток собирается по train-выборке в setup('fit') и доступна как label_schema.

    head определяет, что коллатор кладёт в батч помимо BIO-меток: для span-головы —
    слова и ленточные метки спанов (см. data/span_targets.py).
    """

    def __init__(
        self,
        config: DataConfig,
        head: HeadConfig = BioHeadConfig(),
        model_name_or_path: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.head = head
        self.tokenizer = load_tokenizer(config.tokenizer_name_or_path)
        self.max_length = resolve_max_length(
            self.tokenizer,
            config.max_length,
            model_name_or_path,
            config.tokenizer_name_or_path,
        )
        logger.info(f"NER max_length: {self.max_length}")
        self.label_schema: LabelSchema | None = None
        self.mention_pool: MentionPool | None = None
        self.training_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None
        self.stats: dict[str, dict[str, object]] = {}
        return

    @override
    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:  # ty: ignore[invalid-method-override]
        if stage not in ("fit", "validate"):
            raise NotImplementedError(f"Stage {stage} is not implemented in NerDataModule")
        if self.training_dataset is not None and self.validation_dataset is not None:
            return  # estimate_total_steps вызывает setup('fit') повторно

        train_records = read_records(self.config.train_path, limit=self.config.limit_train_samples)
        val_records = read_records(self.config.val_path, limit=self.config.limit_val_samples)

        train_types = entity_type_counts(train_records)
        val_types = entity_type_counts(val_records)
        unknown = set(val_types) - set(train_types)
        if unknown:
            raise ValueError(f"Validation has entity types absent in train: {sorted(unknown)}")
        schema = LabelSchema(train_types)
        logger.info(f"Entity types (train): {dict(train_types)}; (val): {dict(val_types)}")
        if self.config.source_weights is not None:
            # падаем здесь, до токенизации, если источники в данных и в конфиге расходятся
            counts = check_sources([r["source"] for r in train_records], self.config.source_weights)
            logger.info(f"Train sources: {dict(counts)}; weights: {self.config.source_weights}")
        logger.info(f"Label schema: {schema}")
        self.label_schema = schema

        aug = self.config.augmentation
        if aug.mention_replace_prob > 0.0:
            # val не подглядываем: источник пула — собранный build_mention_pool.py jsonl,
            # Mendeley Gold tsv или сам train
            path = aug.mention_pool_path
            if path is not None and path.suffix == ".jsonl":
                self.mention_pool = read_pool_jsonl(path)
                self.mention_pool.learn_stems(train_records)  # окончания train-оригиналов отщепляются
                logger.info(f"Mention pool source: {path}")
            elif path is not None:
                gold_records = read_gold_tsv(path)
                logger.info(
                    f"Mention pool source: {path} "
                    f"({len(gold_records)} sentences, {dict(entity_type_counts(gold_records))})"
                )
                self.mention_pool = build_mention_pool(
                    gold_records,
                    min_count=aug.mention_min_count,
                    min_type_purity=aug.mention_min_type_purity,
                    reference_records=train_records,
                    transliterate=aug.mention_pool_transliterate,
                    lang="uz",
                    source="gold",
                )
            else:
                self.mention_pool = build_mention_pool(
                    train_records,
                    min_count=aug.mention_min_count,
                    min_type_purity=aug.mention_min_type_purity,
                    reference_records=train_records,
                    transliterate=aug.mention_pool_transliterate,
                    source="train",
                )
            logger.info(f"Mention pool (stems per type): {self.mention_pool.stats()}")

        self.training_dataset = prepare_dataset(
            self.config.train_path,
            train_records,
            self.tokenizer,
            schema,
            self.max_length,
            self.config.limit_train_samples,
            desc="Tokenizing train",
        )
        self.validation_dataset = prepare_dataset(
            self.config.val_path,
            val_records,
            self.tokenizer,
            schema,
            self.max_length,
            self.config.limit_val_samples,
            desc="Tokenizing val",
        )
        self.stats = {
            "train": dataset_stats(self.training_dataset, train_records),
            "val": dataset_stats(self.validation_dataset, val_records),
        }
        if isinstance(self.head, SpanHeadConfig):
            for subset, dataset, records in (
                ("train", self.training_dataset, train_records),
                ("val", self.validation_dataset, val_records),
            ):
                self.stats[subset].update(self._span_stats(dataset, records, self.head.max_span_width))
        for subset, stats in self.stats.items():
            logger.info(f"Dataset stats [{subset}]: {stats}")
        return

    def _span_stats(self, dataset: Dataset, records: list[Record], max_width: int) -> dict[str, object]:
        """Сколько gold-спанов span-голова не может выразить: длиннее max_span_width или за обрезкой."""
        n_lost = n_misaligned = 0
        for record, offsets in zip(records, dataset["offsets"], strict=True):
            targets = build_span_targets(
                record["text"], [(int(s), int(e)) for s, e in offsets], record["entities"], self.schema, max_width
            )
            n_lost += targets.n_lost
            n_misaligned += targets.n_misaligned
        return {"span_entities_lost": n_lost, "span_entities_misaligned_with_words": n_misaligned}

    @property
    def schema(self) -> LabelSchema:
        if self.label_schema is None:
            raise ValueError("Label schema is not built yet. Call setup('fit') first.")
        return self.label_schema

    def _collator(self, subset: Literal["train", "val"]) -> NerCollator:
        return NerCollator(
            tokenizer=self.tokenizer,
            schema=self.schema,
            max_length=self.max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            seed=_subset_seed(self.config.seed, subset),
            augmentation=self.config.augmentation if subset == "train" else None,
            mention_pool=self.mention_pool if subset == "train" else None,
            head=self.head,
        )

    def _loader_generator(self, subset: Literal["train", "val"]) -> torch.Generator:
        return torch.Generator().manual_seed(_subset_seed(self.config.seed, subset))

    def _train_sampler(self) -> WeightedSourceSampler | None:
        if self.config.source_weights is None or self.training_dataset is None:
            return None
        sampler = WeightedSourceSampler(
            sources=self.training_dataset["source"],
            weights=self.config.source_weights,
            seed=_subset_seed(self.config.seed, "train"),
            epoch_size=self.config.epoch_size,
        )
        logger.log_rank_zero(level="INFO", msg=f"Weighted sampling: {sampler.describe()}")
        return sampler

    @override
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.training_dataset is None:
            raise ValueError("Training dataset is not set up. Call setup('fit') first.")
        sampler = self._train_sampler()
        return DataLoader(
            self.training_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=self._collator("train"),
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.config.num_workers > 0,
            generator=self._loader_generator("train"),
        )

    @override
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.validation_dataset is None:
            raise ValueError("Validation dataset is not set up. Call setup('validate') first.")
        return DataLoader(
            self.validation_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collator("val"),
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.config.num_workers > 0,
            generator=self._loader_generator("val"),
        )
