"""Exact-span метрики NER (совпадение label/start/end), синхронизируемые torchmetrics.

Считает precision/recall/f1 по каждому типу, micro и macro, а также
sentence accuracy — долю документов, где множество предсказанных spans
полностью совпало с разметкой (строгая глобальная метрика качества).
Схема подсчёта совпадает с evaluation/core.py.
"""

from collections.abc import Sequence
from typing import Any, no_type_check, override

import torch
from torchmetrics import Metric

from tanerlan.modern_bert.ner.decoding import EntityKey


def _prf(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp, fp, fn = tp.double(), fp.double(), fn.double()
    precision = torch.where(
        tp + fp > 0, tp / (tp + fp).clamp(min=1), torch.zeros_like(tp)
    )
    recall = torch.where(tp + fn > 0, tp / (tp + fn).clamp(min=1), torch.zeros_like(tp))
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0,
        2 * precision * recall / denominator.clamp(min=1e-12),
        torch.zeros_like(tp),
    )
    return precision, recall, f1


class SpanMetrics(Metric):
    full_state_update = False

    def __init__(self, entity_types: Sequence[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entity_types = list(entity_types)
        self._type_index = {t: i for i, t in enumerate(self.entity_types)}
        n = len(self.entity_types)
        self.add_state(
            "tp", default=torch.zeros(n, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "fp", default=torch.zeros(n, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "fn", default=torch.zeros(n, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_docs", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_correct_docs",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    @override
    @no_type_check
    def update(
        self, preds: Sequence[set[EntityKey]], golds: Sequence[set[EntityKey]]
    ) -> None:
        tp = torch.zeros_like(self.tp)
        fp = torch.zeros_like(self.fp)
        fn = torch.zeros_like(self.fn)
        n_correct = 0
        for pred, gold in zip(preds, golds, strict=True):
            n_correct += int(pred == gold)
            for key in pred & gold:
                tp[self._type_index[key[0]]] += 1
            for key in pred - gold:
                fp[self._type_index[key[0]]] += 1
            for key in gold - pred:
                fn[self._type_index[key[0]]] += 1
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.n_docs += len(preds)
        self.n_correct_docs += n_correct

    @override
    @no_type_check
    def compute(self) -> dict[str, torch.Tensor]:
        precision, recall, f1 = _prf(self.tp, self.fp, self.fn)
        micro_p, micro_r, micro_f1 = _prf(self.tp.sum(), self.fp.sum(), self.fn.sum())
        n_docs = self.n_docs.double().clamp(min=1)
        result: dict[str, torch.Tensor] = {
            "micro/precision": micro_p,
            "micro/recall": micro_r,
            "micro/f1": micro_f1,
            "macro/precision": precision.mean(),
            "macro/recall": recall.mean(),
            "macro/f1": f1.mean(),
            "sentence_accuracy": self.n_correct_docs.double() / n_docs,
            "counts/tp": self.tp.sum().double(),
            "counts/fp": self.fp.sum().double(),
            "counts/fn": self.fn.sum().double(),
        }
        for entity_type, idx in self._type_index.items():
            result[f"{entity_type}/precision"] = precision[idx]
            result[f"{entity_type}/recall"] = recall[idx]
            result[f"{entity_type}/f1"] = f1[idx]
        return result
