from pathlib import Path
from typing import Any, cast, override

import torch
import torch.distributed as dist
from kostyl.ml.dist_utils import scale_lrs_by_world_size
from kostyl.ml.integrations.lightning import (
    KostylLightningModule,
    LightningCheckpointLoader,
)
from kostyl.ml.integrations.lightning.utils import estimate_total_steps
from kostyl.ml.optim import create_optimizer, create_scheduler
from kostyl.ml.optim.schedulers import BaseScheduler, CompositeScheduler
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.distributed.device_mesh import DeviceMesh
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.modernbert import ModernBertForMaskedLM
from transformers.utils import is_flash_attn_2_available

from tanerlan.modern_bert.mlm.config import LANG_IDS, TrainingConfig
from tanerlan.modern_bert.mlm.data_module import CollatorOutputDict
from tanerlan.modern_bert.mlm.optim import create_param_groups


class MLMTrainingModule(KostylLightningModule):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.model: ModernBertForMaskedLM | None = None
        self.train_perplexity: Perplexity | None = None
        self.val_perplexity: Perplexity | None = None
        self.train_accuracy: MulticlassAccuracy | None = None
        self.val_accuracy: MulticlassAccuracy | None = None
        self.val_perplexity_per_lang: torch.nn.ModuleDict | None = None
        return

    @override
    def configure_model(self) -> None:
        if self.model is None:
            if not isinstance(
                self.trainer.strategy, (DDPStrategy, SingleDeviceStrategy)
            ):
                raise TypeError(
                    "Only DDPStrategy and SingleDeviceStrategy are supported."
                )

            if (
                temp_path := Path(self.config.model_name_or_path)
            ).is_file() and temp_path.suffix == ".ckpt":
                self.model = LightningCheckpointLoader.load_lightning_checkpoint(
                    ModernBertForMaskedLM,
                    checkpoint_path=temp_path,
                    strict_prefix=True,
                    attn_implementation="flash_attention_2"
                    if is_flash_attn_2_available()
                    else "sdpa",
                )
                self.model.to(self.trainer.strategy.root_device)  # ty: ignore[invalid-argument-type]
            else:
                self.model = ModernBertForMaskedLM.from_pretrained(
                    self.config.model_name_or_path,
                    device_map=self.trainer.strategy.root_device,
                    attn_implementation="flash_attention_2"
                    if is_flash_attn_2_available()
                    else "sdpa",
                )

            vocab_size = self.model.config.vocab_size
            
            self.train_perplexity = Perplexity(ignore_index=-100)
            self.val_perplexity = Perplexity(ignore_index=-100)
            self.train_accuracy = MulticlassAccuracy(
                num_classes=vocab_size, ignore_index=-100, average="micro"
            )
            self.val_accuracy = MulticlassAccuracy(
                num_classes=vocab_size, ignore_index=-100, average="micro"
            )
            self.val_perplexity_per_lang = torch.nn.ModuleDict(
                {lang: Perplexity(ignore_index=-100) for lang in LANG_IDS}
            )
        return

    @override
    @property
    def model_instance(self) -> ModernBertForMaskedLM:
        if self.model is None:
            raise RuntimeError(
                "Model is not configured yet. Call `configure_model()` first."
            )
        return self.model

    @override
    @property
    def model_config(self) -> PreTrainedConfig:
        if self.model is None:
            raise RuntimeError(
                "Model is not configured yet. Call `configure_model()` first."
            )
        return self.model.config

    @property
    def data_parallel_group(self) -> dist.ProcessGroup | None:
        """Returns the data parallel process group."""
        if not dist.is_initialized():
            return None
        strategy = cast(ParallelStrategy, self.trainer.strategy)
        device_mesh: DeviceMesh | None = getattr(strategy, "device_mesh", None)
        if device_mesh is not None:
            return device_mesh.get_group("data_parallel")
        return dist.group.WORLD

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.model is None:
            raise RuntimeError(
                "Model is not configured yet. Call `configure_model()` first."
            )

        self.model.train()

        if dist.is_initialized():
            for attr, lr in self.config.hyperparams.lrs.items():
                lrs = {
                    "base_value": lr.base_value,
                }
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

                setattr(self.config.hyperparams, attr, lr)

        backbone_lr = (
            0.0
            if self.config.hyperparams.backbone_lr.freeze_ratio is not None
            and self.config.hyperparams.backbone_lr.freeze_ratio > 0.0
            else self.config.hyperparams.backbone_lr.warmup_value
            if self.config.hyperparams.backbone_lr.warmup_value is not None
            else self.config.hyperparams.backbone_lr.base_value
        )
        embeddings_lr = (
            self.config.hyperparams.embeddings_lr.warmup_value
            if self.config.hyperparams.embeddings_lr.warmup_value is not None
            else self.config.hyperparams.embeddings_lr.base_value
        )

        param_groups = create_param_groups(
            model=self.model,
            backbone_lr=backbone_lr,
            embeddings_lr=embeddings_lr,
            weight_decay=self.config.hyperparams.weight_decay.base_value,
            # tied decoder.weight схлопывается в имя tok_embeddings и ловится
            # по "embeddings"; head.* и decoder.bias добавлены явно, чтобы
            # голова MLM училась вместе с эмбеддингами, а не стояла в заморозке
            embedding_keywords={"embeddings", "head.", "decoder."},
        )

        optim = create_optimizer(
            parameters_groups=param_groups,
            optimizer_config=self.config.hyperparams.optimizer,
            lr=self.config.hyperparams.backbone_lr.base_value,
            weight_decay=self.config.hyperparams.weight_decay.base_value,
        )

        total_steps = estimate_total_steps(
            trainer=self.trainer,
            dp_process_group=self.data_parallel_group,
        )

        schedulers: dict[str, BaseScheduler] = {
            "backbone_lr": create_scheduler(
                config=self.config.hyperparams.backbone_lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
                ignore_if_field="is_embedding",
            ),
            "embeddings_lr": create_scheduler(
                config=self.config.hyperparams.embeddings_lr,
                optim=optim,
                num_iters=total_steps,
                param_group_field="lr",
                apply_if_field="is_embedding",
            ),
        }
        if self.config.hyperparams.weight_decay.final_value is not None:
            schedulers["weight_decay"] = create_scheduler(
                config=self.config.hyperparams.weight_decay,
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
    def lr_scheduler_step(
        self,
        scheduler: BaseScheduler,
        metric: Any | None,
    ) -> None:  # ty:ignore[invalid-method-override]
        scheduler.step(self.global_step)
        return

    def _base_step(
        self, batch: CollatorOutputDict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        if self.model is None:
            raise RuntimeError(
                "Model is not configured yet. Call `configure_model()` first."
            )
        labels = batch["labels"]
        n_masked = (labels != -100).sum()
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
            num_items_in_batch=n_masked.clamp(min=1),
        )
        return {"loss": outputs.loss}, outputs.logits.detach(), int(n_masked)

    @override
    def training_step(self, batch: CollatorOutputDict, batch_idx: int) -> torch.Tensor:
        if self.train_perplexity is None or self.train_accuracy is None:
            raise RuntimeError(
                "Metrics are not configured yet. Call `configure_model()` first."
            )

        metrics, logits, n_masked = self._base_step(batch)
        labels = batch["labels"]
        self.train_perplexity(logits.float(), labels)
        self.train_accuracy(logits.flatten(0, 1), labels.flatten())
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
            stage="train",
        )
        self.log(
            "train/perplexity",
            self.train_perplexity,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/masked_accuracy",
            self.train_accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train_loss",
            metrics["loss"].detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
            batch_size=n_masked,
        )
        return metrics["loss"]

    @override
    def validation_step(
        self, batch: CollatorOutputDict, batch_idx: int
    ) -> torch.Tensor:
        if (
            self.val_perplexity is None
            or self.val_accuracy is None
            or self.val_perplexity_per_lang is None
        ):
            raise RuntimeError(
                "Metrics are not configured yet. Call `configure_model()` first."
            )

        metrics, logits, n_masked = self._base_step(batch)
        labels = batch["labels"]
        self.val_perplexity.update(logits.float(), labels)  # ty: ignore[invalid-argument-type]
        self.val_accuracy.update(logits.flatten(0, 1), labels.flatten())  # ty: ignore[invalid-argument-type]
        for lang, lang_id in LANG_IDS.items():
            lang_mask = batch["lang_ids"] == lang_id
            if not bool(lang_mask.any()):
                continue
            lang_perplexity = cast(Perplexity, self.val_perplexity_per_lang[lang])
            lang_perplexity.update(logits[lang_mask].float(), labels[lang_mask])  # ty: ignore[invalid-argument-type]
            self.log(
                f"val_per_lang_perplexity/{lang}",
                lang_perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                sync_dist_group=self.data_parallel_group,
            )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
            batch_size=n_masked,
            stage="val",
        )
        self.log(
            "val/macro_perplexity",
            self.val_perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
        )
        self.log(
            "val/masked_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
        )
        self.log(
            "val_loss",
            metrics["loss"].detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            sync_dist_group=self.data_parallel_group,
            batch_size=n_masked,
        )
        return metrics["loss"]
