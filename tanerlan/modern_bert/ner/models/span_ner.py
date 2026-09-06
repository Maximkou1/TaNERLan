"""ModernBERT + span-голова: классификация спанов по словам в ленточной раскладке.

Тушка (`model`) и `head` (dense -> GELU -> LayerNorm) совпадают с
ModernBertForTokenClassification, поэтому веса грузятся из тех же MLM- и
NER-чекпоинтов; всё, что ниже, — своё:

  токены -(pooling по word_index)-> слова (B, W, H)
  start = FFN(слова), end = FFN(слова)                       (B, W, d)
  feats[i, k] = score(start_i, end_{i+k}) + emb(k)           (B, W, K, S)   K = max_width
  [CNN-блоки 3x3 по сетке спанов (i, j), в ленте — скошенное ядро]
  logits = Linear(dropout(LayerNorm(feats)))                 (B, W, K, C)   C = типы + 1

score — affine (Linear([start; end])) или biaffine (+ start^T U end, U блочно-
диагональная по num_heads головам). Ячейки с i + k >= W обнуляются до и после
каждого слоя, чтобы паддинг не протекал через свёртки.
"""

from dataclasses import dataclass
from typing import Any, cast, override

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from transformers.modeling_outputs import ModelOutput
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertConfig,
    ModernBertModel,
    ModernBertPredictionHead,
    ModernBertPreTrainedModel,
)


class ModernBertSpanNerConfig(ModernBertConfig):
    model_type = "modernbert_span_ner"

    def __init__(
        self,
        span_scorer: str = "biaffine",
        span_proj_size: int = 256,
        span_hidden_size: int = 150,
        span_num_heads: int = 1,
        span_cnn_depth: int = 0,
        span_cnn_kernel_size: int = 3,
        span_max_width: int = 24,
        span_word_pooling: str = "first",
        span_dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.span_scorer = span_scorer
        self.span_proj_size = span_proj_size
        self.span_hidden_size = span_hidden_size
        self.span_num_heads = span_num_heads
        self.span_cnn_depth = span_cnn_depth
        self.span_cnn_kernel_size = span_cnn_kernel_size
        self.span_max_width = span_max_width
        self.span_word_pooling = span_word_pooling
        self.span_dropout = span_dropout


@dataclass
class SpanNerOutput(ModelOutput):
    logits: torch.Tensor | None = None  # (B, W, K, C)
    word_mask: torch.Tensor | None = None  # (B, W)


def band_gather(x: torch.Tensor, max_width: int) -> torch.Tensor:
    """(B, W, D) -> (B, W, K, D): out[b, i, k] = x[b, i + k], нули за концом (view через unfold)."""
    padded = F.pad(x, (0, 0, 0, max_width - 1))
    return padded.unfold(1, max_width, 1).transpose(-1, -2)


def band_mask(word_mask: torch.Tensor, max_width: int) -> torch.Tensor:
    """(B, W) -> (B, W, K): спан (i, i + k) допустим, если оба слова настоящие."""
    return word_mask[:, :, None] & band_gather(word_mask[..., None], max_width)[..., 0]


def pool_words(
    hidden: torch.Tensor, word_index: torch.Tensor, num_words: int, pooling: str
) -> torch.Tensor:
    """(B, T, H) токены -> (B, W, H) слова по word_index (-1 = токен вне слов)."""
    batch, seq_len, dim = hidden.shape
    slots = num_words + 1  # слот 0 — мусорный, для токенов вне слов
    flat_index = (word_index + 1) + torch.arange(batch, device=hidden.device)[:, None] * slots
    flat_index = flat_index.reshape(-1)
    if pooling == "first":
        positions = torch.arange(seq_len, device=hidden.device).repeat(batch)
        first = torch.full((batch * slots,), seq_len, dtype=torch.long, device=hidden.device)
        first.scatter_reduce_(0, flat_index, positions, reduce="amin", include_self=True)
        first = first.view(batch, slots)[:, 1:].clamp(max=seq_len - 1)
        return torch.gather(hidden, 1, first[..., None].expand(-1, -1, dim))
    if pooling == "mean":
        sums = torch.zeros((batch * slots, dim), dtype=hidden.dtype, device=hidden.device)
        sums.index_add_(0, flat_index, hidden.reshape(-1, dim))
        counts = torch.zeros((batch * slots,), dtype=hidden.dtype, device=hidden.device)
        counts.index_add_(0, flat_index, torch.ones_like(flat_index, dtype=hidden.dtype))
        words = sums / counts.clamp(min=1.0)[:, None]
        return words.view(batch, slots, dim)[:, 1:]
    raise ValueError(f"Unknown word pooling: {pooling}")


class MultiHeadBilinear(nn.Module):
    """start^T U end с блочно-диагональной U: num_heads блоков (d/h, S, d/h), сумма по головам."""

    def __init__(self, proj_size: int, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if proj_size % num_heads != 0:
            raise ValueError("proj_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = proj_size // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, self.head_dim, hidden_size, self.head_dim))

    @override
    def forward(self, start: torch.Tensor, end_band: torch.Tensor) -> torch.Tensor:
        """start (B, W, d), end_band (B, W, K, d) -> (B, W, K, S)."""
        batch, n_words, _ = start.shape
        start_h = start.view(batch, n_words, self.num_heads, self.head_dim)
        end_h = end_band.view(batch, n_words, end_band.shape[2], self.num_heads, self.head_dim)
        projected = torch.einsum("bwhx,hxsy->bwhsy", start_h, self.weight.to(start.dtype))
        return torch.einsum("bwhsy,bwkhy->bwks", projected, end_h)


class SkewedConvBlock(nn.Module):
    """Свёртка kxk по сетке спанов (i, j), выполненная на ленте (i, k = j - i).

    Сосед (i + di, j + dj) в ленте лежит в (i + di, k + dj - di), поэтому ядро kxk
    по сетке — это ядро (k, 2k-1) по ленте с маской |dk + di| <= r; маска фиксирована,
    веса вне её не участвуют. Pre-norm остаток: x + drop(GELU(conv(LN(x)))).
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        radius = kernel_size // 2
        self.padding = (radius, 2 * radius)
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv2d(channels, channels, (kernel_size, 2 * kernel_size - 1), padding=self.padding)
        self.drop = nn.Dropout(dropout)
        # Маска — python-константа, а не буфер: transformers собирает модель на meta-устройстве,
        # и непersistent-буфер после from_pretrained остаётся неинициализированной памятью.
        self.mask_taps: list[list[float]] = [
            [1.0 if abs(dk + di) <= radius else 0.0 for dk in range(-2 * radius, 2 * radius + 1)]
            for di in range(-radius, radius + 1)
        ]

    def mask(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.mask_taps, dtype=like.dtype, device=like.device)

    @override
    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x (B, W, K, S), valid (B, W, K, 1)."""
        h = (self.norm(x) * valid).permute(0, 3, 1, 2)  # (B, S, W, K)
        weight = self.conv.weight * self.mask(self.conv.weight)
        h = F.conv2d(h, weight, self.conv.bias, padding=self.padding)
        h = F.gelu(h.permute(0, 2, 3, 1))
        return (x + self.drop(h)) * valid


class SpanHead(nn.Module):
    def __init__(self, config: ModernBertSpanNerConfig) -> None:
        super().__init__()
        hidden, proj, size = config.hidden_size, config.span_proj_size, config.span_hidden_size
        dropout = config.span_dropout
        self.max_width = config.span_max_width
        self.start_proj = nn.Sequential(nn.Linear(hidden, proj), nn.GELU(), nn.Dropout(dropout))
        self.end_proj = nn.Sequential(nn.Linear(hidden, proj), nn.GELU(), nn.Dropout(dropout))
        # Linear([start; end]) = start_affine(start) + end_affine(end): считаем по словам, а не по спанам
        self.start_affine = nn.Linear(proj, size)
        self.end_affine = nn.Linear(proj, size, bias=False)
        self.width_embedding = nn.Embedding(self.max_width, size)
        self.bilinear: MultiHeadBilinear | None = (
            MultiHeadBilinear(proj, size, config.span_num_heads) if config.span_scorer == "biaffine" else None
        )
        self.blocks = nn.ModuleList(
            [SkewedConvBlock(size, config.span_cnn_kernel_size, dropout) for _ in range(config.span_cnn_depth)]
        )
        self.norm = nn.LayerNorm(size)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(size, cast(int, config.num_labels))

    @override
    def forward(self, words: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        """words (B, W, H), word_mask (B, W) -> logits (B, W, K, C)."""
        start = self.start_proj(words)
        end = self.end_proj(words)
        width = torch.arange(self.max_width, device=words.device)
        feats = (
            self.start_affine(start)[:, :, None, :]
            + band_gather(self.end_affine(end), self.max_width)
            + self.width_embedding(width)[None, None]
        )
        if self.bilinear is not None:
            feats = feats + self.bilinear(start, band_gather(end, self.max_width))
        valid = band_mask(word_mask, self.max_width)[..., None].to(feats.dtype)
        feats = F.gelu(feats) * valid
        for block in self.blocks:
            feats = block(feats, valid)
        return self.classifier(self.drop(self.norm(feats)))


class ModernBertForSpanNer(ModernBertPreTrainedModel):
    config_class = ModernBertSpanNerConfig

    def __init__(self, config: ModernBertSpanNerConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(config.classifier_dropout)
        self.span_head = SpanHead(config)
        self.post_init()

    @override
    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, MultiHeadBilinear):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_index: torch.Tensor,
        num_words: torch.Tensor | None = None,
        word_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SpanNerOutput:
        """word_index (B, T): индекс слова токена или -1. Число слов задаётся либо num_words (B,),
        либо word_mask (B, W) — второй вариант для экспорта в ONNX, где W должно быть входной
        размерностью, а не значением из данных."""
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]
        hidden = self.drop(self.head(hidden))
        if word_mask is None:
            if num_words is None:
                raise ValueError("either num_words or word_mask is required")
            max_words = max(int(num_words.max()) if num_words.numel() else 0, 1)
            word_mask = torch.arange(max_words, device=hidden.device)[None, :] < num_words[:, None]
        words = pool_words(hidden, word_index, word_mask.shape[1], self.config.span_word_pooling)
        logits = self.span_head(words, word_mask.bool())
        return SpanNerOutput(logits=logits, word_mask=word_mask)
