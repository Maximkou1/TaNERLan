from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from datasets.arrow_dataset import Dataset
import lightning as L
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tanerlan.modern_bert.mlm.config import DataConfig
from tanerlan.modern_bert.mlm.data.dataset_preparation import prepare_dataset
from typing import TypedDict, Literal, override
import torch
from torch.utils.data import DataLoader


class CollatorOutputDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    lang_ids: torch.Tensor


_SUBSET_SEED_OFFSETS: dict[str, int] = {"train": 0, "val": 1}


def _subset_seed(seed: int, subset: Literal["train", "val"]) -> int:
    """Разные, но стабильные сиды для train и val.

    Смещения фиксированы: hash(str) в Python рандомизирован per-process
    (PYTHONHASHSEED) и давал бы разный seed на каждом запуске и ранге.
    """
    return seed + _SUBSET_SEED_OFFSETS[subset]


class _LangTaggingCollator:
    """Вынимает lang_id до MLM-коллатора и возвращает его в батче тензором lang_ids.

    DataCollatorForLanguageModeling не должен видеть лишнюю колонку: она не
    паддится и не участвует в маскировании, это метка примера целиком.
    """

    def __init__(self, base: DataCollatorForLanguageModeling) -> None:
        self.base = base

    def __call__(self, examples: list[dict[str, object]]) -> CollatorOutputDict:
        lang_ids = torch.tensor([example.pop("lang_id") for example in examples])
        batch = self.base(examples)
        batch["lang_ids"] = lang_ids
        return batch


class MLMDataModule(L.LightningDataModule):
    def __init__(self, config: DataConfig) -> None:
        super().__init__()

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
        if tokenizer is None:
            raise ValueError(f"Tokenizer not found at {config.tokenizer_name_or_path}")

        self.training_collator = _LangTaggingCollator(
            DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=config.mlm_probability,
                pad_to_multiple_of=config.pad_to_multiple_of,
                mask_replace_prob=config.mask_replace_prob,
                random_replace_prob=config.random_replace_prob,
                seed=_subset_seed(config.seed, "train"),
            )
        )
        self.validation_collator = _LangTaggingCollator(
            DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=config.mlm_probability,
                pad_to_multiple_of=config.pad_to_multiple_of,
                mask_replace_prob=config.mask_replace_prob,
                random_replace_prob=config.random_replace_prob,
                seed=_subset_seed(config.seed, "val"),
            )
        )

        self.tokenizer = tokenizer
        self.config = config

        self.training_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None
        return

    @override
    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:  # ty: ignore[invalid-method-override]
        match stage:
            case "fit":
                self.training_dataset = prepare_dataset(
                    dataset_configs=self.config.dataset_configs,
                    subset="train",
                    tokenizer=self.tokenizer,
                    seed=self.config.seed,
                )
                self.validation_dataset = prepare_dataset(
                    dataset_configs=self.config.dataset_configs,
                    subset="val",
                    tokenizer=self.tokenizer,
                    seed=self.config.seed,
                )
            case "validate":
                self.validation_dataset = prepare_dataset(
                    dataset_configs=self.config.dataset_configs,
                    subset="val",
                    tokenizer=self.tokenizer,
                    seed=self.config.seed,
                )
            case _:
                raise NotImplementedError(
                    f"Stage {stage} is not implemented in MLMDataModule"
                )
        return

    def _loader_generator(self, subset: Literal["train", "val"]) -> torch.Generator:
        """Генератор base seed воркеров лоадера данного сабсета."""
        return torch.Generator().manual_seed(_subset_seed(self.config.seed, subset))

    @override
    def train_dataloader(
        self,
    ) -> TRAIN_DATALOADERS:
        if self.training_dataset is None:
            raise ValueError("Training dataset is not set up. Call setup('fit') first.")
        return DataLoader(
            self.training_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.training_collator,
            pin_memory=True,
            drop_last=True,
            generator=self._loader_generator("train"),
        )

    @override
    def val_dataloader(
        self,
    ) -> EVAL_DATALOADERS:
        if self.validation_dataset is None:
            raise ValueError(
                "Validation dataset is not set up. Call setup('validate') first."
            )
        return DataLoader(
            self.validation_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.validation_collator,
            pin_memory=True,
            drop_last=True,
            generator=self._loader_generator("val"),
        )
