"""Предикт NER: класс NerPredictor для сервинга и CLI-обёртка над ним.

    from tanerlan.modern_bert.ner.predict import NerPredictor
    predictor = NerPredictor.from_pretrained(["<run1>/hf_model", "<run2>/hf_model"])
    entities = predictor.predict(["Toshkent shahar hokimligi ...", ...])
    # -> [[{"label": "ORG", "start": 0, "end": 25}, ...], ...] в координатах исходного текста

Одна модель: текст нормализуется (prepare_input, длина сохраняется), токенизируется,
forward, декодер той же головы, что на валидации (BIO -> decoding.py, span ->
span_decoding.py). Несколько моделей: выходы усредняются на канонической сетке
слов текста (ensemble.py), так что можно смешивать токенизаторы и головы.

Бэкенд forward выбирается по содержимому директории модели (backend="auto"):
model.xml (OpenVINO IR) или model.onnx -> OpenVINO на CPU, иначе HF-веса -> torch.
Экспорт делает export_to_onnx.py; директория экспорта содержит config.json и
токенизатор, так что для остального кода она неотличима от HF-модели.

Длинные документы: если документ длиннее max_position_embeddings модели, он идёт
отдельным батчем целиком — окон нет, модель видит весь документ; предел —
max_length токенов. ModernBERT (RoPE theta 160k у global-слоёв + локальные окна)
экстраполирует за контекст сам: на склейках dev по 9–14k токенов F1 без
масштабирования 0.82–0.87 при 0.83–0.85 у тех же документов по отдельности, а
dynamic-NTK на них давал столько же или хуже (14k: 0.80 против 0.82). Поэтому
масштабирование включается автоматически и только с порога: на батч с документом
длиннее rope_scaling_threshold x контекст rope_type global-слоёв переключается в
"dynamic" (transformers пересчитывает частоты под длину батча), после батча
возвращаются "default" и исходные частоты. Ниже порога модель работает ровно как
обучена.
"""

import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import rich_click as click
import torch
from kostyl.utils import setup_logger
from transformers import PreTrainedConfig, PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.models.modernbert import (
    ModernBertConfig,
    ModernBertForTokenClassification,
)
from transformers.utils import is_flash_attn_2_available

from tanerlan.modern_bert.ner.config import BioHeadConfig, HeadConfig, SpanHeadConfig
from tanerlan.modern_bert.ner.data.collator import NerBatch, NerCollator
from tanerlan.modern_bert.ner.data.dataset_preparation import Offsets, trim_offsets
from tanerlan.modern_bert.ner.data.records import Entity, read_records, write_jsonl
from tanerlan.modern_bert.ner.data_module import load_tokenizer
from tanerlan.modern_bert.ner.decoding import (
    BioTransitions,
    Word,
    decode_probs,
    decode_word_emissions,
    group_words,
)
from tanerlan.modern_bert.ner.ensemble import (
    bio_word_emissions,
    canonical_words,
    emissions_to_band,
    span_band_to_canonical,
)
from tanerlan.modern_bert.ner.labels import LabelSchema
from tanerlan.modern_bert.ner.models import (
    ModernBertForSpanNer,
    ModernBertSpanNerConfig,
)
from tanerlan.modern_bert.ner.span_decoding import SpanDecoding, decode_span_probs
from tanerlan.modern_bert.tokenizer.tokenization_utils import prepare_input

logger = setup_logger(fmt="only_message")

_DEFAULT_MAX_LENGTH_FACTOR = 4  # предел длины документа в единицах max_position_embeddings


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """auto = float32 и на CUDA: на RTX 3090 fp32 + sdpa на полном dev в 2 раза быстрее bf16 + flash-attn
    (11.8 с против 23.6 с: документы короткие, unpadding flash-пути дороже самого attention) и
    совпадает с CPU fp32 до сущности; bf16 остаётся опцией ради памяти."""
    if requested == "auto":
        return torch.float32
    return getattr(torch, requested)


def head_from_config(name_or_path: str | Path, span_decoding: SpanDecoding = "greedy") -> HeadConfig:
    """Тип головы по architectures в config.json (локальная директория или имя на HF Hub);
    для span-модели — ещё и ширина ленты."""
    config, _ = PreTrainedConfig.get_config_dict(str(name_or_path))
    architectures = config.get("architectures") or []
    if "ModernBertForSpanNer" in architectures:
        return SpanHeadConfig(max_span_width=int(config["span_max_width"]), decoding=span_decoding)
    if "ModernBertForTokenClassification" in architectures:
        return BioHeadConfig()
    raise ValueError(f"Unsupported architectures in config of {name_or_path}: {architectures}")


def prepare_rope(config: ModernBertConfig, factor: float) -> None:
    """Кладёт factor dynamic-NTK в параметры RoPE global-слоёв; rope_type остаётся default.
    Sliding-слои не трогаем: их окно (local_attention) много короче, растяжение им не нужно."""
    rope_parameters = cast(dict[str, dict[str, Any]], config.rope_parameters)
    params = dict(rope_parameters["full_attention"])
    params["factor"] = float(factor)
    rope_parameters["full_attention"] = params


def set_rope_mode(model: PreTrainedModel, dynamic: bool) -> None:
    """dynamic=True: rope_type global-слоёв -> "dynamic", и декоратор dynamic_rope_update пересчитает
    частоты под длину следующего батча. dynamic=False: обратно "default", исходные частоты и кэш длины."""
    rotary = getattr(model, model.base_model_prefix).rotary_emb
    layer_type = "full_attention"
    if layer_type not in rotary.rope_type:
        return
    rotary.rope_type[layer_type] = "dynamic" if dynamic else "default"
    if not dynamic:
        original = getattr(rotary, f"{layer_type}_original_inv_freq")
        current = getattr(rotary, f"{layer_type}_inv_freq")
        current.copy_(original.to(current.device))
        setattr(rotary, f"{layer_type}_max_seq_len_cached", rotary.original_max_seq_len)


def load_model(
    model_dir: str | Path,
    head: HeadConfig,
    device: torch.device,
    dtype: torch.dtype,
    rope_scaling_factor: float = 1.0,
) -> PreTrainedModel:
    use_flash = (
        device.type == "cuda"
        and dtype in (torch.bfloat16, torch.float16)
        and is_flash_attn_2_available()
    )
    attn_implementation = "flash_attention_2" if use_flash else "sdpa"
    if isinstance(head, SpanHeadConfig):
        model_cls: type[PreTrainedModel] = ModernBertForSpanNer
        config = ModernBertSpanNerConfig.from_pretrained(model_dir)
    else:
        model_cls = ModernBertForTokenClassification
        config = ModernBertConfig.from_pretrained(model_dir)
    prepare_rope(config, rope_scaling_factor)
    model = model_cls.from_pretrained(model_dir, config=config, dtype=dtype, attn_implementation=attn_implementation)
    model.to(device)  # ty: ignore[invalid-argument-type]
    model.eval()
    return model


Backend = Literal["auto", "torch", "openvino", "onnxruntime"]
ONNX_FILE = "model.onnx"
ORT_ONNX_FILE = "model.ort.onnx"  # граф для ONNX Runtime (opset 18: его CUDA-ядра не знают RotaryEmbedding-23)
OPENVINO_FILE = "model.xml"


class ForwardBackend(Protocol):
    name: str

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Именованные входы (numpy) -> логиты (numpy float32)."""


class TorchBackend:
    name = "torch"

    def __init__(self, model: PreTrainedModel, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.inference_mode()
    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        tensors = {k: torch.from_numpy(v).to(self.device) for k, v in inputs.items()}
        return self.model(**tensors).logits.float().cpu().numpy()


class OpenVinoBackend:
    """model.xml (IR) или model.onnx через OpenVINO; все размерности динамические."""

    name = "openvino"

    def __init__(self, model_path: Path, device: str = "CPU", num_threads: int | None = None) -> None:
        import openvino as ov

        core = ov.Core()
        config: dict[str, Any] = {}
        if num_threads is not None and device.upper().startswith("CPU"):
            config["INFERENCE_NUM_THREADS"] = int(num_threads)
        self.compiled = core.compile_model(core.read_model(str(model_path)), device, config)
        self.input_names = [port.get_any_name() for port in self.compiled.inputs]
        self.output = self.compiled.output(0)
        self.device = device

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        feed = {name: np.ascontiguousarray(inputs[name]) for name in self.input_names}
        return np.asarray(self.compiled(feed)[self.output], dtype=np.float32)


class OnnxRuntimeBackend:
    """ONNX Runtime на GPU: CUDAExecutionProvider, опционально TensorRT (fp16, кэш движков рядом с
    моделью). По замерам (RTX 3090, mmBERT): CUDA EP fp32 равен torch fp32 и в 3–5 раз медленнее torch
    bf16 + flash-attention; TensorRT fp16 быстрее только на коротких батчах, на длинных документах
    медленнее и теряет точность. Поэтому на CUDA при наличии HF-весов auto выбирает torch, а этот
    бэкенд — для директорий, где есть только экспорт. Для CPU предпочтителен OpenVinoBackend."""

    name = "onnxruntime"

    def __init__(self, model_path: Path, device: torch.device, use_tensorrt: bool = False, fp16: bool = True) -> None:
        import onnxruntime as ort

        try:
            ort.preload_dlls()  # CUDA/cuDNN/TensorRT из pip-пакетов nvidia-* и tensorrt_libs
        except Exception:  # noqa: BLE001 — старые версии без preload_dlls
            pass
        device_id = device.index or 0
        providers: list[Any] = []
        available = set(ort.get_available_providers())
        if use_tensorrt and "TensorrtExecutionProvider" in available:
            cache = model_path.parent / "trt_cache"
            cache.mkdir(exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": device_id,
                        "trt_fp16_enable": fp16,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache),
                        "trt_timing_cache_enable": True,
                        "trt_timing_cache_path": str(cache),
                    },
                )
            )
        providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), options, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.providers = self.session.get_providers()
        self.device = device

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        feed = {name: np.ascontiguousarray(inputs[name]) for name in self.input_names}
        return np.asarray(self.session.run(None, feed)[0], dtype=np.float32)


def find_exported_model(model_dir: str | Path) -> Path | None:
    directory = Path(model_dir)
    for name in (OPENVINO_FILE, ONNX_FILE):
        if (directory / name).is_file():
            return directory / name
    return None


@dataclass
class LoadedModel:
    name: str
    backend: ForwardBackend
    tokenizer: PreTrainedTokenizerBase
    head: HeadConfig
    schema: LabelSchema
    max_positions: int  # max_position_embeddings: длиннее — отдельный батч с масштабированным RoPE
    weight: float = 1.0

    @property
    def torch_model(self) -> PreTrainedModel | None:
        return self.backend.model if isinstance(self.backend, TorchBackend) else None


@dataclass
class ModelDocOutput:
    """Выход одной модели на одном документе (numpy float32)."""

    offsets: Offsets
    probs: np.ndarray | None = None  # BIO: (T, num_tags)
    band: np.ndarray | None = None  # span: (W, K, C)
    words: list[Word] | None = None  # span: слова модели


class NerPredictor:
    """Тексты -> сущности с координатами. Одна модель или ансамбль, длинные документы без окон."""

    def __init__(
        self,
        models: Sequence[LoadedModel],
        device: torch.device,
        batch_size: int = 16,
        max_length: int | None = None,
        span_decoding: SpanDecoding = "greedy",
        none_scale: float = 1.0,
        rope_scaling_threshold: float | None = 2.0,
        max_batch_tokens: int = 8192,
        type_scales: dict[str, float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required")
        types = {tuple(m.schema.entity_types) for m in models}
        if len(types) != 1:
            raise ValueError(f"Models have different entity types: {types}")
        self.models = list(models)
        self.device = device
        self.batch_size = batch_size
        # бюджет батча в токенах (число документов x длина с паддингом): длинные документы идут мелкими
        # батчами, иначе разложенный attention в ONNX Runtime/OpenVINO материализует B x heads x T x T
        self.max_batch_tokens = max_batch_tokens
        self.max_length = max_length or _DEFAULT_MAX_LENGTH_FACTOR * min(m.max_positions for m in models)
        self.span_decoding: SpanDecoding = span_decoding
        self.none_scale = none_scale  # дисконт класса "нет сущности" в span-декодере (см. span_decoding.py)
        self.type_scales = dict(type_scales or {})  # множители на типы при декодировании ({"ORG": 1.2})
        # dynamic-NTK RoPE включается на батч, если документ длиннее threshold x max_position_embeddings; None — никогда
        self.rope_scaling_threshold = rope_scaling_threshold
        self.schema = self.models[0].schema
        self.transitions = BioTransitions(self.schema)

    @classmethod
    def from_pretrained(
        cls,
        model_dirs: str | Path | Sequence[str | Path],
        device: str = "auto",
        dtype: str = "auto",
        weights: Sequence[float] | None = None,
        batch_size: int = 16,
        max_length: int | None = None,
        span_decoding: SpanDecoding = "greedy",
        none_scale: float = 1.0,
        rope_scaling_threshold: float | None = 2.0,
        rope_scaling_factor: float = 1.0,
        max_batch_tokens: int = 8192,
        type_scales: dict[str, float] | None = None,
        backend: Backend = "auto",
        openvino_device: str = "CPU",
        openvino_threads: int | None = None,
        use_tensorrt: bool = False,
    ) -> "NerPredictor":
        """model_dirs — HF-директории (config.json + веса + токенизатор), директории экспорта
        (export_to_onnx.py) или имена репозиториев на HF Hub. backend: auto — если в директории есть
        model.xml/model.onnx, то OpenVINO на CPU и ONNX Runtime (TensorRT, иначе CUDA EP) на CUDA; без
        экспорта — torch. rope_scaling_threshold — с какой длины (в контекстах модели)
        на батч включается dynamic-NTK RoPE с множителем rope_scaling_factor; None — никогда; только torch."""
        dirs = [str(model_dirs)] if isinstance(model_dirs, (str, Path)) else [str(d) for d in model_dirs]
        if weights is not None and len(weights) != len(dirs):
            raise ValueError("weights must match model_dirs")
        torch_device = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, torch_device)
        loaded: list[LoadedModel] = []
        for index, model_dir in enumerate(dirs):
            head = head_from_config(model_dir, span_decoding)
            config, _ = PreTrainedConfig.get_config_dict(model_dir)
            exported = find_exported_model(model_dir) if backend != "torch" else None
            if backend in ("openvino", "onnxruntime") and exported is None:
                raise FileNotFoundError(f"{model_dir}: no {OPENVINO_FILE}/{ONNX_FILE}; run export_to_onnx.py first")
            forward: ForwardBackend
            # auto: экспорт есть -> на CPU OpenVINO, на CUDA ONNX Runtime (TensorRT | CUDA EP); иначе torch
            use_ort = exported is not None and (
                backend == "onnxruntime" or (backend == "auto" and torch_device.type == "cuda")
            )
            if use_ort and exported is not None:
                onnx_path = next(
                    (exported.parent / n for n in (ORT_ONNX_FILE, ONNX_FILE) if (exported.parent / n).is_file()), None
                )
                if onnx_path is None:
                    raise FileNotFoundError(f"{model_dir}: ONNX Runtime needs {ORT_ONNX_FILE} or {ONNX_FILE} (export_to_onnx.py --ort-opset 18)")
                ort_backend = OnnxRuntimeBackend(onnx_path, torch_device, use_tensorrt=use_tensorrt)
                forward = ort_backend
                description = f"ONNX Runtime[{'/'.join(ort_backend.providers)}] {onnx_path.name}"
            elif exported is not None:
                forward = OpenVinoBackend(exported, openvino_device, openvino_threads)
                description = f"OpenVINO[{openvino_device}{f', {openvino_threads} threads' if openvino_threads else ''}] {exported.name}"
            else:
                model = load_model(model_dir, head, torch_device, torch_dtype, rope_scaling_factor)
                forward = TorchBackend(model, torch_device)
                description = f"torch {model.config._attn_implementation} {torch_dtype} on {torch_device}"
            tokenizer = load_tokenizer(model_dir)
            loaded.append(
                LoadedModel(
                    name=model_dir,
                    backend=forward,
                    tokenizer=tokenizer,
                    head=head,
                    schema=LabelSchema.from_label2id(config["label2id"]),
                    max_positions=int(config["max_position_embeddings"]),
                    weight=float(weights[index]) if weights is not None else 1.0,
                )
            )
            logger.info(
                f"Loaded {model_dir}: {config.get('architectures')}, head={head.type}, {description}, "
                f"max_positions={loaded[-1].max_positions}, weight={loaded[-1].weight}"
            )
        return cls(
            loaded,
            torch_device,
            batch_size=batch_size,
            max_length=max_length,
            span_decoding=span_decoding,
            none_scale=none_scale,
            rope_scaling_threshold=rope_scaling_threshold,
            max_batch_tokens=max_batch_tokens,
            type_scales=type_scales,
        )

    # --- публичный API --------------------------------------------------------------

    def predict(self, texts: Sequence[str]) -> list[list[Entity]]:
        """Сущности для каждого текста, отсортированные по началу; координаты — в исходном тексте."""
        if not texts:
            return []
        normalized = [prepare_input(text) for text in texts]
        per_model = [self._run_model(loaded, normalized) for loaded in self.models]
        results: list[list[Entity]] = []
        for doc, text in enumerate(normalized):
            outputs = [per_model[m][doc] for m in range(len(self.models))]
            if len(self.models) == 1:
                entities = self._decode_single(self.models[0], outputs[0], text)
            else:
                entities = self._decode_ensemble(outputs, text)
            results.append(sorted(entities, key=lambda e: (e["start"], e["end"])))
        return results

    # --- forward ----------------------------------------------------------------------

    def _encode(self, loaded: LoadedModel, texts: Sequence[str]) -> list[dict[str, Any]]:
        encoded = loaded.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        examples: list[dict[str, Any]] = []
        for index, (text, input_ids, raw_offsets) in enumerate(
            zip(texts, encoded["input_ids"], encoded["offset_mapping"], strict=True)
        ):
            offsets = trim_offsets(text, raw_offsets)
            covered = max((end for _, end in offsets), default=0)
            if covered < len(text.rstrip()):
                logger.warning(
                    f"[{loaded.name}] document {index} truncated to {self.max_length} tokens "
                    f"({covered}/{len(text)} chars covered)"
                )
            examples.append(
                {
                    "index": index,
                    "hash": str(index),
                    "text": text,
                    "entities": [],
                    "input_ids": list(input_ids),
                    "labels": [-100] * len(input_ids),
                    "offsets": offsets,
                    "n_tokens": len(input_ids),
                }
            )
        return examples

    def _batches(self, loaded: LoadedModel, examples: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        """Документы по убыванию длины; батч ограничен batch_size и бюджетом max_batch_tokens
        (первый, самый длинный документ задаёт длину паддинга). Длинные (> max_positions) — по одному,
        чтобы RoPE масштабировался только для них."""
        order = sorted(examples, key=lambda e: e["n_tokens"], reverse=True)
        batch: list[dict[str, Any]] = []
        for example in order:
            if example["n_tokens"] > loaded.max_positions:
                yield [example]
                continue
            padded = batch[0]["n_tokens"] if batch else example["n_tokens"]
            if batch and (len(batch) >= self.batch_size or (len(batch) + 1) * padded > self.max_batch_tokens):
                yield batch
                batch = []
            batch.append(example)
        if batch:
            yield batch

    @torch.inference_mode()
    def _run_model(self, loaded: LoadedModel, texts: Sequence[str]) -> list[ModelDocOutput]:
        examples = self._encode(loaded, texts)
        collator = NerCollator(
            loaded.tokenizer, loaded.schema, self.max_length, pad_to_multiple_of=8, seed=0, head=loaded.head
        )
        outputs: list[ModelDocOutput | None] = [None] * len(examples)
        for chunk in self._batches(loaded, examples):
            batch = collator(chunk)
            n_tokens = chunk[0]["n_tokens"]
            is_long = n_tokens > loaded.max_positions
            torch_model = loaded.torch_model
            use_dynamic = (
                torch_model is not None
                and self.rope_scaling_threshold is not None
                and n_tokens > self.rope_scaling_threshold * loaded.max_positions
            )
            if is_long:
                logger.info(
                    f"[{loaded.name}] long document: {n_tokens} > {loaded.max_positions} tokens, "
                    f"rope={'dynamic' if use_dynamic else 'default'}"
                )
            if use_dynamic and torch_model is not None:
                set_rope_mode(torch_model, dynamic=True)
            try:
                probs = self._forward(loaded, batch)
            finally:
                if use_dynamic and torch_model is not None:
                    set_rope_mode(torch_model, dynamic=False)
            for row, example in enumerate(chunk):
                offsets = batch["offsets"][row]
                if isinstance(loaded.head, SpanHeadConfig):
                    words = group_words(example["text"], offsets)
                    outputs[example["index"]] = ModelDocOutput(
                        offsets=offsets, band=probs[row, : len(words)], words=words
                    )
                else:
                    outputs[example["index"]] = ModelDocOutput(offsets=offsets, probs=probs[row, : len(offsets)])
        return cast(list[ModelDocOutput], outputs)

    def _forward(self, loaded: LoadedModel, batch: NerBatch) -> np.ndarray:
        """Вероятности: BIO (B, T, tags), span (B, W, K, C). Входы одинаковы для torch и OpenVINO."""
        inputs: dict[str, np.ndarray] = {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
        }
        if isinstance(loaded.head, SpanHeadConfig):
            num_words = batch["num_words"]
            max_words = int(batch["span_labels"].shape[1])
            inputs["word_index"] = batch["word_index"].numpy()
            inputs["word_mask"] = (torch.arange(max_words)[None, :] < num_words[:, None]).numpy()
        logits = loaded.backend(inputs)
        return torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()

    # --- декодирование ----------------------------------------------------------------

    def _decode_single(self, loaded: LoadedModel, output: ModelDocOutput, text: str) -> list[Entity]:
        """Ровно тот декодер, что на валидации при обучении."""
        if isinstance(loaded.head, SpanHeadConfig):
            assert output.band is not None and output.words is not None
            words = [(w.start, w.end) for w in output.words]
            return decode_span_probs(
                output.band, words, text, self.schema, self.span_decoding, self.none_scale, self.type_scales
            )
        assert output.probs is not None
        return decode_probs(output.probs, output.offsets, text, self.schema, self.transitions, self.type_scales)

    def _decode_ensemble(self, outputs: Sequence[ModelDocOutput], text: str) -> list[Entity]:
        words = canonical_words(text)
        if not words:
            return []
        total_weight = sum(m.weight for m in self.models)
        span_widths = [m.head.max_span_width for m in self.models if isinstance(m.head, SpanHeadConfig)]

        if not span_widths:  # все BIO: усредняем эмиссии по словам, один Viterbi
            emissions = sum(
                m.weight * bio_word_emissions(cast(np.ndarray, o.probs), o.offsets, words, len(text), self.transitions)
                for m, o in zip(self.models, outputs, strict=True)
            )
            return decode_word_emissions(
                cast(np.ndarray, emissions) / total_weight, words, text, self.schema, self.transitions, self.type_scales
            )

        max_width = max(span_widths)
        band = None
        for loaded, output in zip(self.models, outputs, strict=True):
            if isinstance(loaded.head, SpanHeadConfig):
                assert output.band is not None and output.words is not None
                model_band = span_band_to_canonical(output.band, output.words, words, len(text), max_width)
            else:
                assert output.probs is not None
                emissions = bio_word_emissions(output.probs, output.offsets, words, len(text), self.transitions)
                model_band = emissions_to_band(emissions, self.transitions, max_width)
            band = loaded.weight * model_band if band is None else band + loaded.weight * model_band
        assert band is not None
        return decode_span_probs(
            band / total_weight, words, text, self.schema, self.span_decoding, self.none_scale, self.type_scales
        )


# --- CLI ----------------------------------------------------------------------------------


def parse_type_scales(raw: Sequence[str]) -> dict[str, float]:
    """["ORG=1.2", "GEO=0.9"] -> {"ORG": 1.2, "GEO": 0.9}."""
    scales: dict[str, float] = {}
    for item in raw:
        entity_type, _, value = item.partition("=")
        if not entity_type or not value:
            raise click.BadParameter(f"expected TYPE=FACTOR, got {item!r}", param_hint="--type-scale")
        scales[entity_type.strip()] = float(value)
    return scales



@click.command(context_settings={"show_default": True})
@click.option("--test-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option(
    "--model-dir",
    "model_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    multiple=True,
    help="HF-директория с чекпоинтом и токенизатором; повторить для ансамбля.",
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--weights", type=str, default=None, help="Веса моделей через запятую (по умолчанию равные).")
@click.option("--batch-size", type=int, default=16)
@click.option("--max-batch-tokens", type=int, default=8192, help="Бюджет батча: документов x длина с паддингом (у ONNX Runtime память attention ~ B x T^2).")
@click.option("--max-length", type=int, default=None, help="Предел токенов; по умолчанию 4 x max_position_embeddings.")
@click.option("--device", type=str, default="auto")
@click.option("--dtype", type=click.Choice(["auto", "float32", "bfloat16", "float16"]), default="auto")
@click.option("--span-decoding", type=click.Choice(["greedy", "dp"]), default="greedy", help="Декодер span-модели и ансамбля со span-моделью.")
@click.option("--none-scale", type=float, default=1.0, help="Дисконт класса 'нет сущности' в span-декодере (<1 повышает recall).")
@click.option("--type-scale", "type_scales_raw", multiple=True, help="Множитель на тип при декодировании, напр. ORG=1.2; повторяемо.")
@click.option("--rope-scaling-threshold", type=float, default=2.0, help="dynamic-NTK RoPE включается на батч, если документ длиннее threshold x контекст; 0 = никогда (только torch).")
@click.option("--rope-scaling-factor", type=float, default=1.0, help="Множитель dynamic-NTK (factor).")
@click.option("--backend", type=click.Choice(["auto", "torch", "openvino", "onnxruntime"]), default="auto", help="auto: при наличии экспорта OpenVINO на CPU / ONNX Runtime на CUDA, иначе torch.")
@click.option("--tensorrt/--no-tensorrt", "use_tensorrt", default=False, help="ONNX Runtime: TensorRT EP (fp16) перед CUDA EP; нужна группа зависимостей tensorrt.")
@click.option("--openvino-device", default="CPU", help="Устройство OpenVINO (CPU, GPU, AUTO).")
@click.option("--openvino-threads", type=int, default=None, help="INFERENCE_NUM_THREADS для OpenVINO CPU; по умолчанию решает OpenVINO.")
@click.option("--limit", type=int, default=None, help="Обработать только первые N записей.")
def main(
    test_path: Path,
    model_dirs: tuple[Path, ...],
    output: Path,
    weights: str | None,
    batch_size: int,
    max_batch_tokens: int,
    max_length: int | None,
    device: str,
    dtype: str,
    span_decoding: Literal["greedy", "dp"],
    none_scale: float,
    type_scales_raw: tuple[str, ...],
    rope_scaling_threshold: float,
    rope_scaling_factor: float,
    backend: Literal["auto", "torch", "openvino", "onnxruntime"],
    use_tensorrt: bool,
    openvino_device: str,
    openvino_threads: int | None,
    limit: int | None,
) -> None:
    started = time.monotonic()
    predictor = NerPredictor.from_pretrained(
        list(model_dirs),
        device=device,
        dtype=dtype,
        weights=[float(w) for w in weights.split(",")] if weights else None,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_length=max_length,
        span_decoding=span_decoding,
        none_scale=none_scale,
        type_scales=parse_type_scales(type_scales_raw),
        rope_scaling_threshold=rope_scaling_threshold if rope_scaling_threshold > 0 else None,
        rope_scaling_factor=rope_scaling_factor,
        backend=backend,
        openvino_device=openvino_device,
        openvino_threads=openvino_threads,
        use_tensorrt=use_tensorrt,
    )
    records = read_records(test_path, require_entities=False, limit=limit)
    predictions = predictor.predict([record["text"] for record in records])
    write_jsonl(
        output,
        [
            {"hash": record["hash"], "text": record["text"], "entities": [dict(e) for e in entities]}
            for record, entities in zip(records, predictions, strict=True)
        ],
    )
    by_type = Counter(e["label"] for entities in predictions for e in entities)
    logger.info(
        f"Wrote {len(records)} records, {sum(by_type.values())} entities {dict(by_type)} "
        f"to {output} in {time.monotonic() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
