"""Экспорт обученной NER-модели (BIO или span) в ONNX и, опционально, в OpenVINO IR.

    python tanerlan/modern_bert/ner/export_to_onnx.py \\
        --model-dir artifacts/experiments/<exp>/<run>/hf_model \\
        --output artifacts/onnx/<name> [--openvino] [--compress-to-fp16] [--check]

Граф — вся модель до логитов: BIO (input_ids, attention_mask) -> (B, T, tags);
span (input_ids, attention_mask, word_index, word_mask) -> (B, W, K, C). Все
размерности динамические. В выходную директорию копируются config.json и
токенизатор, так что её можно передать в NerPredictor / predict.py / сервис как
обычную модель: бэкенд выбирается по наличию model.xml (OpenVINO IR) или
model.onnx. RoPE в графе с исходными частотами: dynamic-NTK для документов
длиннее контекста в ONNX недоступен, они идут как есть.

Opset 23 (по умолчанию) экспортирует SDPA одним опом Attention: OpenVINO не
материализует T x T матрицу и на документах в тысячи токенов работает в ~2 раза
быстрее torch на CPU; в opset 18 attention разложен и на длинных документах
OpenVINO втрое медленнее torch. Для ONNX Runtime на CUDA (у его ядер нет
RotaryEmbedding-23) дополнительно пишется model.ort.onnx в --ort-opset (18).
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import rich_click as click
import torch
from kostyl.utils import setup_logger
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from tanerlan.modern_bert.ner.config import SpanHeadConfig
from tanerlan.modern_bert.ner.data.collator import NerCollator
from tanerlan.modern_bert.ner.data.dataset_preparation import encode_texts
from tanerlan.modern_bert.ner.data_module import load_tokenizer
from tanerlan.modern_bert.ner.labels import LabelSchema
from tanerlan.modern_bert.ner.predict import head_from_config, load_model

logger = setup_logger(fmt="only_message")

ONNX_FILE = "model.onnx"
ORT_ONNX_FILE = "model.ort.onnx"
OPENVINO_FILE = "model.xml"
EXPORT_INFO_FILE = "export.json"
_SAMPLE_TEXTS = [
    "Ali Toshkent shahrida ishlaydi. O'zbekiston Respublikasi Prezidenti Shavkat Mirziyoyev.",
    "Алишер Навоий Тошкентда туғилган.",
]


class BioExportModule(nn.Module):
    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # ty: ignore[missing-override-decorator]
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class SpanExportModule(nn.Module):
    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(  # ty: ignore[missing-override-decorator]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_index: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, word_index=word_index, word_mask=word_mask
        ).logits


def example_inputs(model_dir: Path, texts: list[str], is_span: bool, max_span_width: int | None) -> dict[str, torch.Tensor]:
    """Батч из нескольких текстов разной длины: паддинг и >1 по обеим осям, чтобы экспорт не
    зафиксировал размерности."""
    tokenizer = load_tokenizer(str(model_dir))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    schema = LabelSchema.from_label2id(config["label2id"])
    head = SpanHeadConfig(max_span_width=max_span_width or 24) if is_span else None
    encoded = encode_texts(texts, [[] for _ in texts], tokenizer, schema, 8192)
    collator = NerCollator(tokenizer, schema, 8192, 8, 0, head=head) if head else NerCollator(tokenizer, schema, 8192, 8, 0)
    batch = collator([{"hash": str(i), "text": t, "entities": [], **e} for i, (t, e) in enumerate(zip(texts, encoded, strict=True))])
    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    if is_span:
        num_words = batch["num_words"]
        max_words = int(batch["span_labels"].shape[1])
        inputs["word_index"] = batch["word_index"]
        inputs["word_mask"] = torch.arange(max_words)[None, :] < num_words[:, None]
    return inputs


def dynamic_shapes(inputs: dict[str, torch.Tensor]) -> dict[str, dict[int, str]]:
    shapes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
    }
    if "word_index" in inputs:
        shapes["word_index"] = {0: "batch", 1: "sequence"}
        shapes["word_mask"] = {0: "batch", 1: "words"}
    return shapes


def export_onnx(module: nn.Module, inputs: dict[str, torch.Tensor], onnx_path: Path, opset: int) -> None:
    names = list(inputs)
    program = torch.onnx.export(
        module,
        args=(),
        kwargs=inputs,
        f=None,
        dynamo=True,
        opset_version=opset,
        input_names=names,
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes(inputs),
        optimize=True,
    )
    assert program is not None
    program.save(str(onnx_path))


def convert_openvino(onnx_path: Path, xml_path: Path, compress_to_fp16: bool) -> None:
    import openvino as ov

    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(xml_path), compress_to_fp16=compress_to_fp16)


def check_outputs(module: nn.Module, inputs: dict[str, torch.Tensor], output: Path) -> dict[str, float]:
    """Максимальная разница логитов torch (fp32, CPU) и экспортированного графа на примерных входах."""
    import openvino as ov

    with torch.inference_mode():
        reference = module(**inputs).float().numpy()
    numpy_inputs = {name: tensor.numpy() for name, tensor in inputs.items()}
    core = ov.Core()
    result: dict[str, float] = {}
    for name in (OPENVINO_FILE, ONNX_FILE):
        path = output / name
        if not path.is_file():
            continue
        compiled = core.compile_model(core.read_model(str(path)), "CPU")
        got = compiled(numpy_inputs)[compiled.output(0)]
        result[name] = float(np.abs(got - reference).max())
    return result


@click.command(context_settings={"show_default": True})
@click.option("--model-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="HF-директория обученной модели.")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True, help="Куда сложить model.onnx, config.json и токенизатор.")
@click.option("--opset", type=int, default=23, help="23: attention одним опом Attention (в OpenVINO в ~2 раза быстрее на длинных документах); 18 — разложенный.")
@click.option("--attn-implementation", type=click.Choice(["sdpa", "eager"]), default="sdpa")
@click.option("--openvino/--no-openvino", default=True, help="Дополнительно сконвертировать в OpenVINO IR (model.xml/.bin).")
@click.option("--compress-to-fp16/--no-compress-to-fp16", default=False, help="Веса IR в fp16 (меньше файл; логиты чуть отличаются).")
@click.option("--keep-onnx/--no-keep-onnx", default=True, help="Оставлять model.onnx рядом с IR.")
@click.option("--ort-opset", type=int, default=18, help="Дополнительный граф model.ort.onnx для ONNX Runtime на CUDA (его ядра не знают RotaryEmbedding/Attention-23); 0 = не делать.")
@click.option("--check/--no-check", default=True, help="Сверить логиты экспорта с torch на примерных входах.")
@click.option("--sample-text", "sample_texts", multiple=True, help="Тексты для трассировки (по умолчанию встроенные).")
def main(
    model_dir: Path,
    output: Path,
    opset: int,
    attn_implementation: str,
    openvino: bool,
    compress_to_fp16: bool,
    keep_onnx: bool,
    ort_opset: int,
    check: bool,
    sample_texts: tuple[str, ...],
) -> None:
    started = time.monotonic()
    head = head_from_config(model_dir)
    is_span = isinstance(head, SpanHeadConfig)
    model = load_model(model_dir, head, torch.device("cpu"), torch.float32)
    model.config._attn_implementation = attn_implementation
    module: nn.Module = SpanExportModule(model) if is_span else BioExportModule(model)
    module.eval()

    texts = list(sample_texts) or _SAMPLE_TEXTS
    inputs = example_inputs(model_dir, texts, is_span, head.max_span_width if is_span else None)
    logger.info(
        f"Exporting {type(model).__name__} ({head.type}, attn={attn_implementation}) "
        f"with inputs {{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in inputs.items())}}}"
    )

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / ONNX_FILE
    export_onnx(module, inputs, onnx_path, opset)
    logger.info(f"ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB)")

    if openvino:
        convert_openvino(onnx_path, output / OPENVINO_FILE, compress_to_fp16)
        logger.info(f"OpenVINO IR: {output / OPENVINO_FILE} (fp16={compress_to_fp16})")
    if ort_opset > 0 and ort_opset != opset:
        export_onnx(module, inputs, output / ORT_ONNX_FILE, ort_opset)
        logger.info(f"ONNX for ONNX Runtime (opset {ort_opset}): {output / ORT_ONNX_FILE}")

    # config.json и токенизатор: директория становится обычной моделью для NerPredictor
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "tokenizer.model"):
        source = model_dir / name
        if source.is_file():
            shutil.copy2(source, output / name)
    info: dict[str, Any] = {
        "source": str(model_dir),
        "architecture": type(model).__name__,
        "head": head.type,
        "opset": opset,
        "ort_opset": ort_opset if ort_opset > 0 and ort_opset != opset else None,
        "attn_implementation": attn_implementation,
        "torch": torch.__version__,
        "openvino_ir": openvino,
        "compress_to_fp16": compress_to_fp16,
        "inputs": {k: list(v.shape) for k, v in inputs.items()},
    }
    if check:
        diffs = check_outputs(module, inputs, output)
        info["max_abs_logit_diff"] = diffs
        logger.info(f"Max |logits diff| vs torch fp32: {diffs}")
    if openvino and not keep_onnx:
        onnx_path.unlink()
        for extra in output.glob("model.onnx.data"):
            extra.unlink()
    (output / EXPORT_INFO_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Done in {time.monotonic() - started:.1f}s -> {output}")


if __name__ == "__main__":
    main()
