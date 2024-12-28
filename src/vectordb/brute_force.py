"""Exact linear-scan nearest-neighbor search."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

from .distance import Vector, distance, normalize_vector, validate_dimensions, validate_metric


Label = Hashable
SearchResult = Tuple[Label, float]


class BruteForceIndex:
    """A simple exact index useful for small datasets and recall baselines."""

    def __init__(self, dimensions: int, *, metric: str = "euclidean") -> None:
        validate_dimensions(dimensions)
        validate_metric(metric)
        self.dimensions = dimensions
        self.metric = metric
        self._vectors: Dict[Label, Vector] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, label: object) -> bool:
        return label in self._vectors

    def add(self, label: Label, vector: Sequence[float]) -> None:
        if label in self._vectors:
            raise ValueError(f"label already exists: {label!r}")
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError("label must be hashable") from exc
        self._vectors[label] = normalize_vector(vector, self.dimensions, self.metric)

    def add_many(self, items: Iterable[Tuple[Label, Sequence[float]]]) -> None:
        for label, vector in items:
            self.add(label, vector)

    def search(self, vector: Sequence[float], k: int = 10, **_: object) -> List[SearchResult]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        query = normalize_vector(vector, self.dimensions, self.metric)
        results = [
            (label, distance(query, point, self.metric)) for label, point in self._vectors.items()
        ]
        results.sort(key=lambda result: result[1])
        return results[:k]

    def remove(self, label: Label) -> None:
        try:
            del self._vectors[label]
        except KeyError:
            raise KeyError(label) from None

    def get_vector(self, label: Label) -> Vector:
        try:
            return self._vectors[label]
        except KeyError:
            raise KeyError(label) from None
