from typing import Literal
from kostyl.ml.configs import (
    ConfigLoadingMixin,
    LightningTrainerParameters,
    EarlyStoppingConfig,
    CheckpointConfig,
    OptimizerConfig,
    Lr,
    WeightDecay,
)
from pydantic import BaseModel, Field, model_validator

# id письменности в колонке lang_id токенизированного датасета;
# по нему в валидации perplexity считается отдельно на каждую группу
LANG_IDS: dict[str, int] = {
    "ru": 0,
    "uz-lat": 1,
    "uz-cyr": 2,
    "en": 3,
    "code-switching": 4,
}


class DatasetConfig(BaseModel):
    lang: Literal["ru", "uz-lat", "uz-cyr", "en"]
    name_or_path: str
    text_colname: str
    train_split_name: str
    val_split_name: str
    limit_samples: int | None = None


class CodeSwitchingDataConfig(BaseModel):
    en_ratio: float
    ru_ratio: float
    uz_lat_ratio: float
    uz_cyr_ratio: float
    total_samples: int

    @property
    def en_budget(self) -> int:
        return int(self.total_samples * self.en_ratio)

    @property
    def ru_budget(self) -> int:
        return int(self.total_samples * self.ru_ratio)

    @property
    def uz_lat_budget(self) -> int:
        return int(self.total_samples * self.uz_lat_ratio)

    @property
    def uz_cyr_budget(self) -> int:
        return int(self.total_samples * self.uz_cyr_ratio)

    @model_validator(mode="after")
    def check_ratios(self) -> "CodeSwitchingDataConfig":
        total = self.en_ratio + self.ru_ratio + self.uz_lat_ratio + self.uz_cyr_ratio
        if total != 1.0:
            raise ValueError("Ratios must sum to 1.0")
        return self


class DataConfig(BaseModel):
    dataset_configs: list[DatasetConfig | CodeSwitchingDataConfig]
    num_workers: int = 4
    batch_size: int
    seed: int
    mlm_probability: float | None = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: int | None = None
    tokenizer_name_or_path: str | None = None

    @model_validator(mode="after")
    def check_langs(self) -> "DataConfig":
        for config in self.dataset_configs:
            if isinstance(config, CodeSwitchingDataConfig):
                continue
            if config.lang not in LANG_IDS:
                raise ValueError(f"Unsupported language: {config.lang}")
        return self


class HyperparamsConfig(BaseModel):
    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    optimizer: OptimizerConfig
    embeddings_lr: Lr
    backbone_lr: Lr
    weight_decay: WeightDecay

    @property
    def lrs(self) -> dict[str, Lr]:
        return {
            "embeddings_lr": self.embeddings_lr,
            "backbone_lr": self.backbone_lr,
        }


class TrainingConfig(BaseModel, ConfigLoadingMixin):
    experiment_name: str
    model_name_or_path: str
    data: DataConfig
    hyperparams: HyperparamsConfig
    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpointing: CheckpointConfig = CheckpointConfig()
