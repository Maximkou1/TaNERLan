from pathlib import Path
from typing import Annotated, Any, Literal

from kostyl.ml.configs import (
    CheckpointConfig,
    ConfigLoadingMixin,
    EarlyStoppingConfig,
    LightningTrainerParameters,
    Lr,
    OptimizerConfig,
    WeightDecay,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Неизвестный ключ в yaml — ошибка, а не молча проигнорированная опечатка."""

    model_config = ConfigDict(extra="forbid")


class AugmentationConfig(StrictModel):
    """Текстовая аугментация в коллаторе: транслитерация куска текста между письменностями
    и смена регистра всего текста.

    Если в примере есть кириллица, с вероятностью cyr2lat_prob кусок
    переводится в латиницу; иначе, если есть латиница, с вероятностью
    lat2cyr_prob — в кириллицу. С вероятностью full_text_prob переводится
    весь текст, иначе случайный непрерывный кусок длиной от min_chunk_words
    до max_chunk_fraction от числа слов в тексте (слова и сущности внутри
    куска переводятся вместе, координаты сущностей пересчитываются).
    """

    cyr2lat_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    lat2cyr_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    full_text_prob: float = Field(default=0.3, ge=0.0, le=1.0)
    min_chunk_words: int = Field(default=3, ge=1)
    max_chunk_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    # регистр, посимвольно с сохранением длины (оффсеты не меняются), одна выборка на документ:
    # lower_prob / upper_prob -> весь текст; entity_lower_prob / entity_upper_prob -> только спаны
    # сущностей (ХАМИД; НАРГИЗА в обычном тексте, jkga/instagram со строчной)
    lower_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    upper_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_lower_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_upper_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    # замена упоминаний: с вероятностью mention_replace_prob в документе до mention_max_per_doc
    # разных сущностей меняются на другие того же типа и письменности из пула
    # (см. data/mention_pool.py); все формы одной основы в документе (Toshkent, Toshkentda)
    # меняются на одну и ту же замену, падежное окончание оригинала переносится на неё
    mention_replace_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    mention_max_per_doc: int = Field(default=2, ge=1)
    mention_long_bias: float = Field(default=1.5, ge=0.0)  # вес кандидата = words ** bias
    mention_long_bias_by_label: dict[str, float] = {}  # переопределение bias по типу, напр. {ORG: 2.5}
    mention_max_extra_words: int = Field(default=3, ge=0)  # замена не длиннее оригинала + столько слов
    mention_min_count: int = Field(default=2, ge=1)  # основа встречается в источнике пула не реже
    mention_min_type_purity: float = Field(default=0.8, gt=0.0, le=1.0)
    # источник пула: None -> train-разметка; путь к Mendeley "Uzbek NER Gold" tsv -> только он
    # (train тогда служит справочником основ и нарицательных слов, см. mention_pool.py)
    mention_pool_path: Path | None = None
    # дополнить пул копиями основ в другой письменности (Gold размечен только латиницей)
    mention_pool_transliterate: bool = True

    @model_validator(mode="after")
    def check_long_bias(self) -> "AugmentationConfig":
        negative = {k: v for k, v in self.mention_long_bias_by_label.items() if v < 0.0}
        if negative:
            raise ValueError(f"mention_long_bias_by_label должен быть >= 0: {negative}")
        return self

    def long_bias(self, label: str) -> float:
        return self.mention_long_bias_by_label.get(label, self.mention_long_bias)

    @model_validator(mode="after")
    def check_case_probs(self) -> "AugmentationConfig":
        total = self.lower_prob + self.upper_prob + self.entity_lower_prob + self.entity_upper_prob
        if total > 1.0:
            raise ValueError("lower_prob + upper_prob + entity_lower_prob + entity_upper_prob должно быть <= 1")
        return self

    @property
    def case_enabled(self) -> bool:
        return (
            self.lower_prob > 0.0
            or self.upper_prob > 0.0
            or self.entity_lower_prob > 0.0
            or self.entity_upper_prob > 0.0
        )

    @property
    def enabled(self) -> bool:
        return (
            self.cyr2lat_prob > 0.0
            or self.lat2cyr_prob > 0.0
            or self.case_enabled
            or self.mention_replace_prob > 0.0
        )


class DataConfig(StrictModel):
    train_path: Path
    val_path: Path
    tokenizer_name_or_path: str
    max_length: int | None = Field(
        default=None,
        ge=8,
        description="None -> tokenizer.model_max_length, а если он не задан — max_position_embeddings модели",
    )
    batch_size: int = Field(ge=1)
    num_workers: int = Field(default=4, ge=0)
    seed: int
    pad_to_multiple_of: int | None = 8
    limit_train_samples: int | None = None
    limit_val_samples: int | None = None
    augmentation: AugmentationConfig = AugmentationConfig()
    # Доля примеров каждого источника (поле source в jsonl) в эпохе. Ключи
    # должны в точности совпадать с источниками в train-файле (0.0 явно
    # исключает источник), сумма равна 1. None — все записи равновероятны.
    source_weights: dict[str, float] | None = None
    # Число примеров в эпохе при взвешенном сэмплировании; None — размер train-файла.
    epoch_size: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check_source_weights(self) -> "DataConfig":
        if self.source_weights is None:
            if self.epoch_size is not None:
                raise ValueError("epoch_size has effect only together with source_weights")
            return self
        if not self.source_weights:
            raise ValueError("source_weights is empty")
        negative = {k: v for k, v in self.source_weights.items() if v < 0}
        if negative:
            raise ValueError(f"source_weights must be non-negative: {negative}")
        total = sum(self.source_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"source_weights must sum to 1, got {total:.6f}: {self.source_weights}")
        return self


# --- головы ------------------------------------------------------------------------


class LossConfig(StrictModel):
    """ce — cross-entropy (опционально с label smoothing); focal — Lin et al. (2017),
    (1 - p_t) ** gamma * CE, без label smoothing."""

    type: Literal["ce", "focal"] = "ce"
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    focal_gamma: float = Field(default=2.0, ge=0.0)

    @model_validator(mode="after")
    def check_smoothing(self) -> "LossConfig":
        if self.type == "focal" and self.label_smoothing > 0.0:
            raise ValueError("label_smoothing поддерживается только для loss.type='ce'")
        return self


class BioHeadConfig(StrictModel):
    """Классика: ModernBertForTokenClassification, BIO-тег на каждый токен, декодер —
    пословная агрегация + constrained Viterbi (decoding.py)."""

    type: Literal["bio"] = "bio"
    loss: LossConfig = LossConfig()


class BoundarySmoothingConfig(StrictModel):
    """Boundary smoothing (Zhu & Li, ACL 2022): доля epsilon вероятности gold-спана
    (i, j) равномерно раздаётся его соседям по границам — спанам (i', j') с
    |i - i'| + |j - j'| <= distance, того же типа. Оставшаяся 1 - epsilon остаётся
    на самом спане; остальные спаны — целиком класс "нет сущности".
    """

    epsilon: float = Field(default=0.1, gt=0.0, lt=1.0)
    distance: int = Field(default=1, ge=1, le=4)


class SpanHeadConfig(StrictModel):
    """Span-классификация по словам (models/span_ner.py).

    Токены сворачиваются в слова (те же буквенно-цифровые цепочки, что в decoding.py),
    каждое слово получает представление start/end через две FFN-проекции, а каждый
    спан (i, j), 0 <= j - i < max_span_width, — вектор размера hidden_size:

      affine    — Linear([start_i; end_j; emb(j - i)]) -> GELU
      biaffine  — start_i^T U end_j + Linear([start_i; end_j]) + emb(j - i) -> GELU,
                  U блочно-диагональная по num_heads головам (multi-head biaffine)

    Поверх вектора спана cnn_depth свёрточных блоков 3x3 по сетке (i, j) (CNN-NER,
    Yan et al. 2022: спан видит соседние спаны), затем LayerNorm -> dropout -> Linear
    в num_types + 1 классов (0 — нет сущности). Loss — cross-entropy по всем допустимым
    спанам, при boundary_smoothing цели мягкие. Декодер — greedy (по убыванию
    вероятности, без пересечений) или dp (максимум суммы log p(type) - log p(none)
    по непересекающимся спанам).
    """

    type: Literal["span"] = "span"
    scorer: Literal["affine", "biaffine"] = "biaffine"
    proj_size: int = Field(default=256, ge=8)  # размер start/end проекций слов
    hidden_size: int = Field(default=150, ge=4)  # размер вектора спана
    num_heads: int = Field(default=1, ge=1)  # только для biaffine
    cnn_depth: int = Field(default=0, ge=0)
    cnn_kernel_size: int = Field(default=3, ge=3, le=7)
    max_span_width: int = Field(default=24, ge=1)  # в словах; более длинные gold-спаны теряются
    word_pooling: Literal["first", "mean"] = "first"
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    boundary_smoothing: BoundarySmoothingConfig | None = None
    decoding: Literal["greedy", "dp"] = "greedy"

    @model_validator(mode="after")
    def check_heads(self) -> "SpanHeadConfig":
        if self.scorer == "affine" and self.num_heads != 1:
            raise ValueError("num_heads > 1 имеет смысл только для scorer='biaffine'")
        if self.proj_size % self.num_heads != 0:
            raise ValueError("proj_size должен делиться на num_heads")
        if self.cnn_kernel_size % 2 == 0:
            raise ValueError("cnn_kernel_size должен быть нечётным")
        return self


HeadConfig = Annotated[BioHeadConfig | SpanHeadConfig, Field(discriminator="type")]


class ModelConfig(StrictModel):
    """Тушка ModernBERT + голова из head.

    name_or_path — либо HF-директория, либо Lightning-чекпоинт (*.ckpt) из
    MLM- или NER-прогона: из него берутся веса под префиксом "model." и
    сохранённый HF-конфиг; отсутствующие в чекпоинте веса головы
    инициализируются заново.
    """

    name_or_path: str
    head: HeadConfig = BioHeadConfig()
    from_pretrained_kwargs: dict[str, Any] = {}

    @model_validator(mode="after")
    def check_reserved_kwargs(self) -> "ModelConfig":
        clash = {"num_labels", "id2label", "label2id"} & set(
            self.from_pretrained_kwargs
        )
        if clash:
            raise ValueError(
                f"from_pretrained_kwargs содержит зарезервированные ключи {sorted(clash)}: "
                "метки собираются по данным"
            )
        return self


class HyperparamsConfig(StrictModel):
    grad_clip_val: float | None = Field(default=None, gt=0, validate_default=False)
    # layer-wise lr decay тушки: верхний слой и final_norm x1, слой i (0-based) из L x decay ** (L-1-i),
    # эмбеддинги x decay ** L; см. optim/param_groups.py
    layer_lr_decay: float | None = Field(default=None, gt=0.0, le=1.0)
    optimizer: OptimizerConfig
    backbone_lr: Lr
    head_lr: Lr
    weight_decay: WeightDecay

    @property
    def lrs(self) -> dict[str, Lr]:
        return {
            "backbone_lr": self.backbone_lr,
            "head_lr": self.head_lr,
        }


class TrainingConfig(BaseModel, ConfigLoadingMixin):
    experiment_name: str
    model: ModelConfig
    data: DataConfig
    hyperparams: HyperparamsConfig
    trainer: LightningTrainerParameters
    early_stopping: EarlyStoppingConfig | None = None
    checkpointing: CheckpointConfig = CheckpointConfig(
        monitor="val_micro_f1",
        mode="max",
        filename="|{epoch}|-|{step}|-|{val_micro_f1:.4f}|",
    )
