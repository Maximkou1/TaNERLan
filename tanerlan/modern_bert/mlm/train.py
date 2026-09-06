from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import caseconverter
import lightning as L
import rich_click as click
import torch
from kostyl.ml.configs import (
    DDPStrategyConfig,
    SingleDeviceStrategyConfig,
    SupportedStrategies,
)
from kostyl.ml.integrations.lightning.callbacks import (
    setup_checkpoint_callback,
    setup_early_stopping_callback,
)
from kostyl.utils import setup_logger
from lightning import Callback, Trainer
from lightning.pytorch.accelerators import (
    Accelerator,
    CPUAccelerator,
    CUDAAccelerator,
    MPSAccelerator,
)
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy, Strategy

from tanerlan.modern_bert.mlm.config import TrainingConfig
from tanerlan.modern_bert.mlm.data_module import MLMDataModule
from tanerlan.modern_bert.mlm.training_module import MLMTrainingModule

module_logger = setup_logger(fmt="detailed")

torch.set_float32_matmul_precision("high")


def _choose_accelerator(accelerator: str) -> type[Accelerator]:
    """Choose the appropriate PyTorch Lightning Accelerator based on the input string."""
    accelerator = accelerator.lower()
    match accelerator:
        case "cuda":
            accelerator_class = CUDAAccelerator
        case "mps":
            accelerator_class = MPSAccelerator
        case "cpu":
            accelerator_class = CPUAccelerator
        case _:
            raise ValueError(f"Unsupported accelerator: {accelerator}")
    return accelerator_class


def setup_strategy(
    accelerator: str,
    strategy_settings: SupportedStrategies,
    devices: list[int] | int,
) -> Strategy:
    """Configure and return a PyTorch Lightning training strategy."""
    if isinstance(devices, list):
        if len(devices) == 0:
            raise ValueError("Device list cannot be empty.")
        num_devices = len(devices)
        device_ids = devices
    else:
        num_devices = devices
        device_ids = list(range(num_devices))

    accelerator_class = _choose_accelerator(accelerator)
    if not accelerator_class.is_available():
        raise ValueError(f"{accelerator_class.name()} accelerator is not available.")

    parallel_devices: list[torch.device] = accelerator_class.get_parallel_devices(
        device_ids
    )

    match strategy_settings:
        case DDPStrategyConfig():
            if num_devices < 2:
                raise ValueError("DDP strategy requires at least two devices.")
            strategy = DDPStrategy(
                accelerator=accelerator,  # Accelerator is already handled by the Trainer
                parallel_devices=parallel_devices,
                find_unused_parameters=strategy_settings.find_unused_parameters,
            )
        case SingleDeviceStrategyConfig():
            if num_devices != 1:
                raise ValueError("SingleDevice strategy requires exactly one device.")

            strategy = SingleDeviceStrategy(
                device=parallel_devices[0], accelerator=accelerator
            )
        case _:
            raise ValueError(
                f"Unsupported strategy: {strategy_settings.__class__.__name__}"
            )
    return strategy


@click.command()
@click.option(
    "--config-path",
    "-c",
    type=click.Path(
        file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Path to the training configuration YAML file.",
)
@click.option(
    "--experiment-registry",
    "-e",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, resolve_path=True, path_type=Path
    ),
    required=True,
    help="Directory where experiment outputs will be stored.",
)
def run(config_path: Path, experiment_registry: Path) -> None:
    """Main function to set up and run the training process."""
    config = TrainingConfig.from_file(config_path)
    L.seed_everything(config.data.seed, workers=True)

    started_at = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    WORK_DIR: Path = (
        experiment_registry
        / caseconverter.pascalcase(config.experiment_name)
        / started_at
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    ### DataModule setup ###
    datamodule = MLMDataModule(config.data)

    ### TrainingModule setup ###
    training_module = MLMTrainingModule(config=config)

    ### Strategy, Callbacks and logger setup ###
    logger = TensorBoardLogger(
        save_dir=WORK_DIR,
        name="tb_logs",
        default_hp_metric=False,
    )
    logger.log_hyperparams(config.model_dump())

    strategy = setup_strategy(
        accelerator=config.trainer.accelerator,
        strategy_settings=config.trainer.strategy,
        devices=config.trainer.devices,
    )

    callbacks: list[Callback] = []
    if config.early_stopping is not None:
        callbacks.append(setup_early_stopping_callback(config.early_stopping))
    callbacks.append(
        setup_checkpoint_callback(
            dirpath=WORK_DIR / "checkpoints", ckpt_cfg=config.checkpointing
        )
    )
    callbacks.append(
        LearningRateMonitor(
            logging_interval="step", log_weight_decay=True, log_momentum=False
        )
    )

    ### Trainer setup and fit ###
    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator
        if strategy.accelerator is None
        else "auto",
        devices=config.trainer.devices,
        strategy=strategy,
        precision=config.trainer.precision,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.hyperparams.grad_clip_val,
        val_check_interval=config.trainer.val_check_interval,
        callbacks=callbacks,
        log_every_n_steps=config.trainer.log_every_n_steps,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        limit_predict_batches=config.trainer.limit_predict_batches,
        logger=[logger],
    )
    trainer.fit(training_module, datamodule=datamodule)

    ### Сохранение лучшего чекпоинта в формате HF ###
    if not trainer.is_global_zero:
        return

    lightning_module = cast(MLMTrainingModule, trainer.strategy.lightning_module)

    ckpt_callback = cast(ModelCheckpoint | None, trainer.checkpoint_callback)
    ckpt_path: Path | None = None
    ckpt_type: Literal["best", "last"] | None = None
    if ckpt_callback is not None:
        if ckpt_callback.best_model_path:
            ckpt_path = Path(ckpt_callback.best_model_path)
            ckpt_type = "best"
        elif ckpt_callback.last_model_path:
            ckpt_path = Path(ckpt_callback.last_model_path)
            ckpt_type = "last"
        else:
            module_logger.warning(
                "No checkpoint path found, but ModelCheckpoint is provided! "
                "Saving the in-memory model as HF model."
            )

    if ckpt_path is not None:
        ckpt = torch.load(
            ckpt_path, map_location=lightning_module.device, weights_only=False
        )
        lightning_module.load_state_dict(ckpt["state_dict"])
        module_logger.info(f"Loaded {ckpt_type} checkpoint from {ckpt_path}.")

    model = lightning_module.model
    if model is None:
        raise RuntimeError(
            "Something went wrong, model is not initialized. Cannot save HF model."
        )
    hf_model_dir = WORK_DIR / "hf_model"
    model.save_pretrained(hf_model_dir)
    datamodule.tokenizer.save_pretrained(hf_model_dir)
    module_logger.info(f"Saved HF model to {hf_model_dir}.")
    return


if __name__ == "__main__":
    run()
