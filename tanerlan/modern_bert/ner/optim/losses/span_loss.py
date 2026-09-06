"""Cross-entropy по спанам в ленточной раскладке (B, W, K, C) с boundary smoothing.

Boundary smoothing (Zhu & Li, "Boundary Smoothing for Named Entity Recognition",
ACL 2022): gold-спан (i, j) типа t отдаёт долю epsilon своей вероятности соседям —
спанам (i', j') с 0 < |i - i'| + |j - j'| <= distance, поровну между допустимыми,
тоже на класс t. Остаток вероятности каждой ячейки уходит в класс 0 (нет сущности).
Сглаживание по границам, а не по классам: модель перестаёт быть слишком уверенной
в точной позиции границы, где разметка шумит сильнее всего.

В ленте спан (i, j) лежит в ячейке [i, k = j - i], поэтому сдвиг (di, dj) по сетке
спанов — это сдвиг (di, dk = dj - di) по ленте.
"""

import torch
import torch.nn.functional as F

from tanerlan.modern_bert.ner.data.dataset_preparation import IGNORE_INDEX


def band_shifts(distance: int) -> list[tuple[int, int]]:
    """Сдвиги (di, dk) по ленте для соседей на манхэттенском расстоянии 1..distance по сетке (i, j)."""
    shifts: list[tuple[int, int]] = []
    for di in range(-distance, distance + 1):
        for dj in range(-distance, distance + 1):
            if 0 < abs(di) + abs(dj) <= distance:
                shifts.append((di, dj - di))
    return shifts


def shift_band(x: torch.Tensor, di: int, dk: int) -> torch.Tensor:
    """out[..., i, k] = x[..., i - di, k - dk], нули за краем. Работает по двум последним осям."""
    n_i, n_k = x.shape[-2], x.shape[-1]
    if abs(di) >= n_i or abs(dk) >= n_k:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    src_i = slice(max(0, -di), n_i - max(0, di))
    dst_i = slice(max(0, di), n_i - max(0, -di))
    src_k = slice(max(0, -dk), n_k - max(0, dk))
    dst_k = slice(max(0, dk), n_k - max(0, -dk))
    out[..., dst_i, dst_k] = x[..., src_i, src_k]
    return out


def boundary_smoothed_targets(
    span_labels: torch.Tensor,
    num_classes: int,
    epsilon: float,
    distance: int,
) -> torch.Tensor:
    """(B, W, K) метки -> (B, W, K, C) распределения; недопустимые ячейки — нули."""
    valid = span_labels != IGNORE_INDEX
    labels = span_labels.clamp(min=0)
    onehot = F.one_hot(labels, num_classes).to(torch.float32) * valid[..., None]
    entity_mass = onehot[..., 1:].permute(0, 3, 1, 2)  # (B, C-1, W, K)
    valid_f = valid[:, None].to(entity_mass.dtype)  # (B, 1, W, K)

    shifts = band_shifts(distance)
    # число допустимых соседей у каждой ячейки-источника: сумма valid по целям (i + di, k + dk)
    n_neighbors = torch.zeros_like(valid_f)
    for di, dk in shifts:
        n_neighbors += shift_band(valid_f, -di, -dk)
    given = entity_mass * epsilon / n_neighbors.clamp(min=1.0)
    received = torch.zeros_like(entity_mass)
    for di, dk in shifts:
        received += shift_band(given, di, dk)
    received = received * valid_f

    kept = entity_mass * (1.0 - epsilon) * (n_neighbors > 0) + entity_mass * (n_neighbors == 0)
    entity = (kept + received).permute(0, 2, 3, 1)  # (B, W, K, C-1)
    none = (1.0 - entity.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    target = torch.cat([none, entity], dim=-1) * valid[..., None]
    return target / target.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def span_cross_entropy(
    logits: torch.Tensor,
    span_labels: torch.Tensor,
    epsilon: float | None = None,
    distance: int = 1,
) -> torch.Tensor:
    """logits (B, W, K, C) в float32; среднее по допустимым спанам."""
    valid = span_labels != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0.0
    if epsilon is None:
        return F.cross_entropy(logits[valid], span_labels[valid])
    target = boundary_smoothed_targets(span_labels, logits.shape[-1], epsilon, distance)
    log_probs = F.log_softmax(logits, dim=-1)
    per_span = -(target * log_probs).sum(dim=-1)
    return per_span[valid].mean()
