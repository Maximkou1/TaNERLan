"""Сэмплер с заданными долями источников в эпохе, совместимый с DDP.

Наследует DistributedSampler, поэтому Lightning не подменяет его своим и
estimate_total_steps считает длину лоадера как per-rank. Один и тот же
список индексов эпохи строится на всех ранках из seed + epoch и режется
между ранками с шагом world_size; Lightning вызывает set_epoch каждую эпоху.
"""

import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def check_sources(sources: Sequence[str | None], weights: Mapping[str, float]) -> Counter[str]:
    """Источники в данных и в конфиге должны совпадать один в один; иначе ValueError."""
    if any(source is None for source in sources):
        n_missing = sum(1 for source in sources if source is None)
        raise ValueError(
            f"{n_missing} records have no `source` field, but data.source_weights is set"
        )
    counts = Counter(str(source) for source in sources)
    missing_in_data = sorted(set(weights) - set(counts))
    missing_in_config = sorted(set(counts) - set(weights))
    if missing_in_data or missing_in_config:
        raise ValueError(
            "source_weights and train data disagree: "
            f"in config but not in data={missing_in_data}, "
            f"in data but not in config={missing_in_config} "
            "(exclude a source explicitly with weight 0.0)"
        )
    return counts


def _draws_per_source(weights: Mapping[str, float], epoch_size: int) -> dict[str, int]:
    """Целые квоты по источникам; остаток от округления уходит самому весомому."""
    draws = {source: int(round(weight * epoch_size)) for source, weight in weights.items()}
    largest = max(weights, key=lambda source: weights[source])
    draws[largest] += epoch_size - sum(draws.values())
    return draws


class WeightedSourceSampler(DistributedSampler):  # ty: ignore[missing-type-argument]
    def __init__(
        self,
        sources: Sequence[str | None],
        weights: Mapping[str, float],
        seed: int,
        epoch_size: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        distributed = dist.is_available() and dist.is_initialized()
        if num_replicas is None:
            num_replicas = dist.get_world_size() if distributed else 1
        if rank is None:
            rank = dist.get_rank() if distributed else 0
        self.counts = check_sources(sources, weights)
        self.weights = dict(weights)
        self.indices_by_source: dict[str, np.ndarray] = {
            source: np.flatnonzero(np.array([s == source for s in sources])) for source in weights
        }
        self.epoch_size = epoch_size if epoch_size is not None else len(sources)
        self.draws = _draws_per_source(weights, self.epoch_size)
        # DistributedSampler ждёт dataset ради len(); num_samples/total_size переопределяем сами
        super().__init__(
            range(self.epoch_size), num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed
        )
        self.num_samples = math.ceil(self.epoch_size / num_replicas)
        self.total_size = self.num_samples * num_replicas

    @staticmethod
    def _draw(rng: np.random.Generator, indices: np.ndarray, k: int) -> np.ndarray:
        """k индексов без повторов, пока хватает; сверх размера — полные проходы плюс остаток."""
        if k == 0 or len(indices) == 0:
            return np.empty(0, dtype=np.int64)
        full, remainder = divmod(k, len(indices))
        parts = [rng.permutation(indices) for _ in range(full)]
        if remainder:
            parts.append(rng.permutation(indices)[:remainder])
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def epoch_indices(self) -> np.ndarray:
        """Полный список индексов эпохи (одинаковый на всех ранках)."""
        rng = np.random.default_rng(self.seed + self.epoch)
        parts = [
            self._draw(rng, self.indices_by_source[source], self.draws[source]) for source in self.weights
        ]
        indices = np.concatenate(parts)
        rng.shuffle(indices)
        if len(indices) < self.total_size:  # добивка, чтобы у всех ранков было поровну
            indices = np.concatenate([indices, indices[: self.total_size - len(indices)]])
        return indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.epoch_indices()[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def describe(self) -> str:
        parts = []
        for source, weight in self.weights.items():
            n, k = self.counts[source], self.draws[source]
            parts.append(f"{source}: {k} draws/epoch of {n} records ({weight:.0%}, coverage x{k / max(n, 1):.2f})")
        return "; ".join(parts)
