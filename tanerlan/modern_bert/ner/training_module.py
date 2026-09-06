"""LightningModule для NER: общая часть (оптимизатор, span-метрики, логирование ошибок)
и две головы — BIO по токенам (BioTrainingModule) и span-классификация по словам
(SpanTrainingModule). Выбор — model.head.type в yaml, см. create_training_module.

Метрики у обеих голов одни: предсказания декодируются в символьные spans и
сравниваются с разметкой точным совпадением (label, start, end) — та же схема,
что в evaluation/evaluate_model.py.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast, override

import torch
import torch.distributed as dist
from kostyl.ml.configs import FSDP1StrategyConfig
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.dist_utils.fsdp import get_fsdp1_policies, select_wrap_policy
from kostyl.ml.integrations.lightning import KostylLightningModule
from kostyl.ml.integrations.lightning.ckpt_utils import (
    LightningCheckpointLoader,
    LightningConfigLoader,
)
from kostyl.ml.integrations.lightning.utils import estimate_total_steps
from kostyl.ml.optim import create_optimizer, create_scheduler
from kostyl.ml.optim.schedulers import BaseScheduler, CompositeScheduler
from kostyl.utils import setup_logger
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy, SingleDeviceStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel
from torchmetrics import MeanMetric, Metric
from torchmetrics.classification import MulticlassAccuracy
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.modernbert import (
    ModernBertConfig,
    ModernBertForTokenClassification,
)

from tanerlan.modern_bert.ner.config import (
    BioHeadConfig,
    SpanHeadConfig,
    TrainingConfig,
)
from tanerlan.modern_bert.ner.data.collator import NerBatch
from tanerlan.modern_bert.ner.data.dataset_preparation import IGNORE_INDEX
from tanerlan.modern_bert.ner.decoding import (
    BioTransitions,
    EntityKey,
    decode_probs,
    entity_keys,
)
from tanerlan.modern_bert.ner.labels import LabelSchema
from tanerlan.modern_bert.ner.metrics import SpanMetrics
from tanerlan.modern_bert.ner.models import (
    ModernBertForSpanNer,
    ModernBertSpanNerConfig,
)
from tanerlan.modern_bert.ner.optim import create_param_groups
from tanerlan.modern_bert.ner.optim.losses import FocalLoss
from tanerlan.modern_bert.ner.optim.losses.span_loss import span_cross_entropy
from tanerlan.modern_bert.ner.span_decoding import decode_span_probs

logger = setup_logger(fmt="only_message")

_MAX_LOGGED_ERROR_EXAMPLES = 20

Predictions = tuple[list[set[EntityKey]], list[set[EntityKey]]]


def load_pretrained(
    model_cls: type[PreTrainedModel],
    config_cls: type[PreTrainedConfig],
    name_or_path: str,
    config_overrides: dict[str, Any],
    device: torch.device,
    attn_implementation: str,
) -> PreTrainedModel:
    """HF-директория или Lightning-чекпоинт (*.ckpt); config_overrides перекрывают сохранённый конфиг."""
    source = Path(name_or_path)
    if source.is_file() and source.suffix == ".ckpt":
        hf_config = LightningConfigLoader.load_lightning_checkpoint(
            config_cls, checkpoint_path=source, **config_overrides
        )
        model = LightningCheckpointLoader.load_lightning_checkpoint(
            model_cls,
            checkpoint_path=source,
            config=hf_config,
            strict_prefix=True,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        model.to(device)  # ty: ignore[invalid-argument-type]
        logger.log_rank_zero(
            level="INFO", msg=f"Loaded weights from Lightning checkpoint {source}"
        )
        return model
    model = model_cls.from_pretrained(
        name_or_path,
        device_map=device,
        attn_implementation=attn_implementation,
        **config_overrides,
    )
    logger.log_rank_zero(
        level="INFO", msg=f"Loaded weights from HuggingFace checkpoint {name_or_path}"
    )
    return model


class NerTrainingModule(KostylLightningModule, ABC):
    """Общий каркас: подклассы задают модель, loss, декодер и дополнительные метрики шага."""

    def __init__(self, config: TrainingConfig, schema: LabelSchema) -> None:
        super().__init__()
        self.config = config
        self.schema = schema
        self.model: PreTrainedModel | None = None
        self.train_span_metrics: SpanMetrics | None = None
        self.val_span_metrics: SpanMetrics | None = None
        # ModuleDict: Lightning находит метрику по атрибуту модуля при self.log(metric) и сам переносит на устройство
        self.train_extra = nn.ModuleDict()
        self.val_extra = nn.ModuleDict()
        self._val_error_examples: list[dict[str, Any]] = []
        return

    # --- что задаёт голова -----------------------------------------------------------

    @abstractmethod
    def _load_model(
        self, device: torch.device, attn_implementation: str
    ) -> PreTrainedModel: ...

    @abstractmethod
    def _forward(self, batch: NerBatch) -> tuple[torch.Tensor, torch.Tensor, int]:
        """(loss, logits.detach(), число размеченных единиц для взвешивания loss)."""

    @abstractmethod
    def _decode_batch(self, logits: torch.Tensor, batch: NerBatch) -> Predictions:
        """Тот же декодер, что в predict.py -> (предсказанные, gold) множества (label, start, end)."""

    @abstractmethod
    def _build_extra_metrics(self) -> dict[str, Metric]:
        """Дополнительные метрики шага (например, token accuracy); по экземпляру на train и val."""

    @abstractmethod
    def _update_extra_metrics(
        self, metrics: nn.ModuleDict, logits: torch.Tensor, batch: NerBatch
    ) -> None: ...

    def _describe_model(self) -> str:
        return ""

    # --- модель ------------------------------------------------------------------------

    @override
    def configure_model(self) -> None:
        if self.model is not None:
            return
        if not isinstance(
            self.trainer.strategy, (DDPStrategy, SingleDeviceStrategy, FSDPStrategy)
        ):
            raise TypeError("Only DDPStrategy and SingleDeviceStrategy are supported.")
        root_device = self.trainer.strategy.root_device

        self.model = self._load_model(root_device, "sdpa")
        backbone_prefix = f"{self.model.base_model_prefix}."
        n_backbone = sum(
            p.numel()
            for n, p in self.model.named_parameters()
            if n.startswith(backbone_prefix)
        )
        n_head = sum(
            p.numel()
            for n, p in self.model.named_parameters()
            if not n.startswith(backbone_prefix)
        )
        cfg = self.model.config
        logger.log_rank_zero(
            level="INFO",
            msg=(
                f"Model: {type(self.model).__name__}, "
                f"classifier_dropout={cfg.classifier_dropout}, "
                f"attention_dropout={cfg.attention_dropout}, "
                f"mlp_dropout={cfg.mlp_dropout}, "
                f"embedding_dropout={cfg.embedding_dropout}, "
                f"attn_implementation={cfg._attn_implementation}; "
                f"params: backbone={n_backbone / 1e6:.1f}M, head={n_head / 1e6:.2f}M; "
                f"labels={cfg.id2label}{self._describe_model()}"
            ),
        )
        if isinstance(self.trainer.strategy, FSDPStrategy):
            if not isinstance(self.config.trainer.strategy, FSDP1StrategyConfig):
                raise TypeError(
                    "FSDPStrategy requires config.trainer.strategy.type == 'fsdp1'"
                )
            policies = get_fsdp1_policies(self.config.trainer.strategy)
            wrap_policy = select_wrap_policy(self.model)
            self.trainer.strategy.model = FullyShardedDataParallel(
                self,
                auto_wrap_policy=wrap_policy,
                **policies,
                use_orig_params=True,
                device_id=root_device,
            )

        self.train_span_metrics = SpanMetrics(self.schema.entity_types)
        self.val_span_metrics = SpanMetrics(self.schema.entity_types)
        self.train_extra = nn.ModuleDict(self._build_extra_metrics()).to(root_device)
        self.val_extra = nn.ModuleDict(self._build_extra_metrics()).to(root_device)
        return

    @override
    @property
    def model_instance(self) -> PreTrainedModel:
        if self.model is None:
            raise RuntimeError(
                "Model is not configured yet. Call `configure_model()` first."
            )
        return self.model

    @override
    @property
    def model_config(self) -> PreTrainedConfig:
        return self.model_instance.config

    @property
    def data_parallel_group(self) -> dist.ProcessGroup | None:
        if not dist.is_initialized():
            return None
        strategy = cast(ParallelStrategy, self.trainer.strategy)
        device_mesh: DeviceMesh | None = getattr(strategy, "device_mesh", None)
        if device_mesh is not None:
            return device_mesh.get_group("data_parallel")
        return dist.group.WORLD

    # --- оптимизатор -------------------------------------------------------------------

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        model = self.model_instance
        model.train()
        hp = self.config.hyperparams

        if dist.is_initialized():
            for attr, lr in hp.lrs.items():
                lrs = {"base_value": lr.base_value}
                if lr.final_value is not None:
                    lrs["final_value"] = lr.final_value
                if lr.warmup_value is not None:
                    lrs["warmup_value"] = lr.warmup_value
                scaled_lrs = scale_lrs_by_world_size(
                    lrs=lrs,
                    verbosity_level="rank-zero-only",
                    group=self.data_parallel_group,
                )
                for key, value in scaled_lrs.items():
                    setattr(lr, key, value)
                setattr(hp, attr, lr)

        def initial_lr(lr_cfg: Any) -> float:
            if lr_cfg.freeze_ratio is not None and lr_cfg.freeze_ratio > 0.0:
                return 0.0
            return (
                lr_cfg.warmup_value
                if lr_cfg.warmup_value is not None
                else lr_cfg.base_value
            )

        param_groups = create_param_groups(
            model=model,
            weight_decay=hp.weight_decay.base_value,
            backbone_lr=initial_lr(hp.backbone_lr),
            head_lr=initial_lr(hp.head_lr),
            layer_lr_decay=hp.layer_lr_decay,
        )
        optim = create_optimizer(
            parameters_groups=param_groups,
            optimizer_config=hp.optimizer,
            lr=hp.backbone_lr.base_value,
            weight_decay=hp.weight_decay.base_value,
        )
        total_steps = estimate_total_steps(
            trainer=self.trainer, dp_process_group=self.data_parallel_group
        )

        schedulers: dict[str, BaseScheduler] = {
            "backbone_lr": create_scheduler(
                config=hp.backbone_lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
                ignore_if_field="is_head",
                multiplier_field="lr_scale",  # layer-wise lr decay, см. optim/param_groups.py
            ),
            "head_lr": create_scheduler(
                config=hp.head_lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
                apply_if_field="is_head",
            ),
        }
        if hp.weight_decay.final_value is not None:
            schedulers["weight_decay"] = create_scheduler(
                config=hp.weight_decay,
                optim=optim,
                num_iters=total_steps,
                param_group_field="weight_decay",
            )
        scheduler = CompositeScheduler(optimizer=optim, **schedulers)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }  # ty: ignore[invalid-return-type]

    @override
    def lr_scheduler_step(self, scheduler: BaseScheduler, metric: Any | None) -> None:  # ty:ignore[invalid-method-override]
        scheduler.step(self.global_step)
        return

    # --- шаги --------------------------------------------------------------------------

    @override
    def training_step(self, batch: NerBatch, batch_idx: int) -> torch.Tensor:
        if self.train_span_metrics is None:
            raise RuntimeError(
                "Metrics are not configured yet. Call `configure_model()` first."
            )

        loss, logits, n_labeled = self._forward(batch)
        self._update_extra_metrics(self.train_extra, logits, batch)
        preds, golds = self._decode_batch(logits, batch)
        step_metrics = self.train_span_metrics(preds, golds)

        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        for name, metric in self.train_extra.items():
            self.log(
                f"train/{name}",
                cast(Metric, metric),
                on_step=True,
                on_epoch=False,
                logger=True,
            )
        self.log_dict(
            {
                "train/micro/f1": step_metrics["micro/f1"],
                "train/sentence_accuracy": step_metrics["sentence_accuracy"],
                "train/aug_fraction": torch.tensor(
                    sum(batch["augmented"]) / len(batch["augmented"])
                ),
                "train/seq_len": torch.tensor(float(batch["input_ids"].shape[1])),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        # Только on_step: прогрессбар живёт лишь на ранге 0 и в on_train_epoch_end
        # вычисляет epoch-метрики до того, как это сделают остальные ранги. Метрика
        # с on_epoch=True + sync_dist=True при этом делает all_reduce только на
        # ранге 0 -> рассинхрон NCCL-коллективов и дедлок в конце эпохи.
        self.log(
            "train_loss",
            loss.detach(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
            batch_size=n_labeled,
        )
        return loss

    @override
    def on_train_epoch_end(self) -> None:
        if self.train_span_metrics is None:
            return
        metrics = self.train_span_metrics.compute()  # ty: ignore[missing-argument]
        self.log_dict(
            {f"train_epoch/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        self.train_span_metrics.reset()
        return

    @override
    def validation_step(self, batch: NerBatch, batch_idx: int) -> torch.Tensor:
        if self.val_span_metrics is None:
            raise RuntimeError(
                "Metrics are not configured yet. Call `configure_model()` first."
            )

        loss, logits, n_labeled = self._forward(batch)
        self._update_extra_metrics(self.val_extra, logits, batch)
        preds, golds = self._decode_batch(logits, batch)
        self.val_span_metrics.update(preds, golds)  # ty: ignore[invalid-argument-type]
        self._collect_error_examples(batch, preds, golds)

        self.log(
            "val/loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
            batch_size=n_labeled,
        )
        for name, metric in self.val_extra.items():
            self.log(
                f"val/{name}",
                cast(Metric, metric),
                on_step=False,
                on_epoch=True,
                logger=True,
            )
        self.log(
            "val_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
            batch_size=n_labeled,
        )
        return loss

    def _collect_error_examples(
        self, batch: NerBatch, preds: list[set[EntityKey]], golds: list[set[EntityKey]]
    ) -> None:
        if (
            not self.trainer.is_global_zero
            or len(self._val_error_examples) >= _MAX_LOGGED_ERROR_EXAMPLES
        ):
            return
        for row, (pred, gold) in enumerate(zip(preds, golds, strict=True)):
            if pred == gold:
                continue
            text = batch["texts"][row]
            self._collect_error_example(batch["hashes"][row], text, pred, gold)
            if len(self._val_error_examples) >= _MAX_LOGGED_ERROR_EXAMPLES:
                return

    def _collect_error_example(
        self, doc_hash: str, text: str, pred: set[EntityKey], gold: set[EntityKey]
    ) -> None:
        def render(keys: set[EntityKey]) -> str:
            return ", ".join(
                f"{label}:{text[start:end]!r}"
                for label, start, end in sorted(keys, key=lambda k: k[1])
            )

        self._val_error_examples.append(
            {
                "hash": doc_hash,
                "text": text[:300] + ("…" if len(text) > 300 else ""),
                "missed": render(gold - pred),
                "spurious": render(pred - gold),
            }
        )

    @override
    def on_validation_epoch_end(self) -> None:
        if self.val_span_metrics is None:
            return
        metrics = self.val_span_metrics.compute()  # ty: ignore[missing-argument]
        self.log_dict(
            {f"val/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        # ключ без "/" для мониторинга чекпоинтов и имени файла
        self.log(
            "val_micro_f1",
            metrics["micro/f1"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.log(
            "val_sentence_accuracy",
            metrics["sentence_accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.val_span_metrics.reset()
        self._log_error_examples()
        return

    def _log_error_examples(self) -> None:
        examples, self._val_error_examples = self._val_error_examples, []
        if not self.trainer.is_global_zero or not examples:
            return
        tb_logger = next(
            (lg for lg in self.trainer.loggers if isinstance(lg, TensorBoardLogger)),
            None,
        )
        if tb_logger is None:
            return
        lines = []
        for example in examples:
            lines.append(
                f"- **{example['hash']}**: {example['text']}\n"
                f"  - missed: {example['missed'] or '-'}\n"
                f"  - spurious: {example['spurious'] or '-'}"
            )
        tb_logger.experiment.add_text(
            "val/error_examples", "\n".join(lines), self.global_step
        )
        return


class BioTrainingModule(NerTrainingModule):
    """ModernBertForTokenClassification: BIO-тег на токен, CE/focal, декодер decoding.py."""

    def __init__(self, config: TrainingConfig, schema: LabelSchema) -> None:
        super().__init__(config, schema)
        if not isinstance(config.model.head, BioHeadConfig):
            raise TypeError("BioTrainingModule requires model.head.type == 'bio'")
        self.head_config: BioHeadConfig = config.model.head
        self.transitions = BioTransitions(schema)  # маска BIO-переходов для декодера
        loss_cfg = self.head_config.loss
        # focal: собственный модуль без параметров; ce: functional, label smoothing из конфига
        self.focal_loss: FocalLoss | None = (
            FocalLoss(gamma=loss_cfg.focal_gamma, ignore_index=IGNORE_INDEX)
            if loss_cfg.type == "focal"
            else None
        )
        return

    @override
    def _load_model(
        self, device: torch.device, attn_implementation: str
    ) -> PreTrainedModel:
        config_overrides: dict[str, Any] = {
            "num_labels": self.schema.num_tags,
            "id2label": self.schema.id2tag,
            "label2id": self.schema.tag2id,
            **self.config.model.from_pretrained_kwargs,
        }
        return load_pretrained(
            ModernBertForTokenClassification,
            ModernBertConfig,
            self.config.model.name_or_path,
            config_overrides,
            device,
            attn_implementation,
        )

    @override
    def _build_extra_metrics(self) -> dict[str, Metric]:
        return {
            "token_accuracy": MulticlassAccuracy(
                num_classes=self.schema.num_tags,
                ignore_index=IGNORE_INDEX,
                average="micro",
            )
        }

    @override
    def _update_extra_metrics(
        self, metrics: nn.ModuleDict, logits: torch.Tensor, batch: NerBatch
    ) -> None:
        cast(Metric, metrics["token_accuracy"]).update(
            logits.flatten(0, 1), batch["labels"].flatten()  # ty: ignore[invalid-argument-type]
        )

    @override
    def _forward(self, batch: NerBatch) -> tuple[torch.Tensor, torch.Tensor, int]:
        labels = batch["labels"]
        outputs = self.model_instance(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits
        loss = self._compute_loss(logits.flatten(0, 1).float(), labels.flatten())
        n_labeled = int((labels != IGNORE_INDEX).sum())
        return loss, logits.detach(), n_labeled

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: (N, num_tags) в float32, labels: (N,), IGNORE_INDEX не учитывается."""
        if self.focal_loss is not None:
            return self.focal_loss(logits, labels)
        return torch.nn.functional.cross_entropy(
            logits,
            labels,
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.head_config.loss.label_smoothing,
        )

    @override
    def _decode_batch(self, logits: torch.Tensor, batch: NerBatch) -> Predictions:
        """Пословная агрегация + constrained Viterbi (decoding.py)."""
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        preds: list[set[EntityKey]] = []
        golds: list[set[EntityKey]] = []
        for row, (offsets, text, entities) in enumerate(
            zip(batch["offsets"], batch["texts"], batch["entities"], strict=True)
        ):
            decoded = decode_probs(
                probs[row, : len(offsets)], offsets, text, self.schema, self.transitions
            )
            preds.append(entity_keys(decoded))
            golds.append(entity_keys(entities))
        return preds, golds


class SpanTrainingModule(NerTrainingModule):
    """ModernBertForSpanNer: класс на каждый спан слов, CE с boundary smoothing, декодер span_decoding.py."""

    def __init__(self, config: TrainingConfig, schema: LabelSchema) -> None:
        super().__init__(config, schema)
        if not isinstance(config.model.head, SpanHeadConfig):
            raise TypeError("SpanTrainingModule requires model.head.type == 'span'")
        self.head_config: SpanHeadConfig = config.model.head
        return

    @override
    def _load_model(
        self, device: torch.device, attn_implementation: str
    ) -> PreTrainedModel:
        head = self.head_config
        config_overrides: dict[str, Any] = {
            "num_labels": self.schema.num_span_classes,
            "id2label": self.schema.span_id2label,
            "label2id": self.schema.span_label2id,
            "span_scorer": head.scorer,
            "span_proj_size": head.proj_size,
            "span_hidden_size": head.hidden_size,
            "span_num_heads": head.num_heads,
            "span_cnn_depth": head.cnn_depth,
            "span_cnn_kernel_size": head.cnn_kernel_size,
            "span_max_width": head.max_span_width,
            "span_word_pooling": head.word_pooling,
            "span_dropout": head.dropout,
            **self.config.model.from_pretrained_kwargs,
        }
        return load_pretrained(
            ModernBertForSpanNer,
            ModernBertSpanNerConfig,
            self.config.model.name_or_path,
            config_overrides,
            device,
            attn_implementation,
        )

    @override
    def _describe_model(self) -> str:
        head = self.head_config
        smoothing = (
            f"eps={head.boundary_smoothing.epsilon}, D={head.boundary_smoothing.distance}"
            if head.boundary_smoothing is not None
            else "off"
        )
        return (
            f"; span head: scorer={head.scorer}, heads={head.num_heads}, cnn_depth={head.cnn_depth}, "
            f"max_width={head.max_span_width}, pooling={head.word_pooling}, boundary_smoothing={smoothing}, "
            f"decoding={head.decoding}"
        )

    @override
    def _build_extra_metrics(self) -> dict[str, Metric]:
        # доля gold-спанов, у которых argmax — их тип (без учёта декодера и ложных спанов)
        return {"gold_span_accuracy": MeanMetric()}

    @override
    def _update_extra_metrics(
        self, metrics: nn.ModuleDict, logits: torch.Tensor, batch: NerBatch
    ) -> None:
        labels = batch["span_labels"]
        gold = labels > 0
        if bool(gold.any()):
            correct = (logits.argmax(dim=-1)[gold] == labels[gold]).float()
            cast(Metric, metrics["gold_span_accuracy"]).update(correct)  # ty: ignore[invalid-argument-type]

    @override
    def _forward(self, batch: NerBatch) -> tuple[torch.Tensor, torch.Tensor, int]:
        outputs = self.model_instance(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_index=batch["word_index"],
            num_words=batch["num_words"],
        )
        logits = outputs.logits.float()
        labels = batch["span_labels"]
        smoothing = self.head_config.boundary_smoothing
        loss = span_cross_entropy(
            logits,
            labels,
            epsilon=smoothing.epsilon if smoothing is not None else None,
            distance=smoothing.distance if smoothing is not None else 1,
        )
        n_spans = int((labels != IGNORE_INDEX).sum())
        return loss, logits.detach(), n_spans

    @override
    def _decode_batch(self, logits: torch.Tensor, batch: NerBatch) -> Predictions:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds: list[set[EntityKey]] = []
        golds: list[set[EntityKey]] = []
        for row, (words, text, entities) in enumerate(
            zip(batch["words"], batch["texts"], batch["entities"], strict=True)
        ):
            decoded = decode_span_probs(
                probs[row], words, text, self.schema, self.head_config.decoding
            )
            preds.append(entity_keys(decoded))
            golds.append(entity_keys(entities))
        return preds, golds


def create_training_module(
    config: TrainingConfig, schema: LabelSchema
) -> NerTrainingModule:
    match config.model.head:
        case BioHeadConfig():
            return BioTrainingModule(config, schema)
        case SpanHeadConfig():
            return SpanTrainingModule(config, schema)
        case _:
            raise ValueError(f"Unsupported head: {config.model.head}")
