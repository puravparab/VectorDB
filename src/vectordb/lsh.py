"""Random-hyperplane locality-sensitive hashing."""

from __future__ import annotations

import itertools
import random
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from .distance import Vector, distance, normalize_vector, validate_dimensions, validate_metric


Label = Hashable
SearchResult = Tuple[Label, float]


class LSHIndex:
    """Approximate search using several random-hyperplane hash tables."""

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "cosine",
        tables: int = 8,
        hash_size: int = 12,
        probe_radius: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        validate_dimensions(dimensions)
        validate_metric(metric)
        if tables <= 0 or hash_size <= 0:
            raise ValueError("tables and hash_size must be positive")
        if probe_radius < 0 or probe_radius > hash_size:
            raise ValueError("probe_radius must be between zero and hash_size")
        self.dimensions = dimensions
        self.metric = metric
        self.tables = tables
        self.hash_size = hash_size
        self.probe_radius = probe_radius
        randomizer = random.Random(seed)
        self._planes = [
            [tuple(randomizer.gauss(0.0, 1.0) for _ in range(dimensions)) for _ in range(hash_size)]
            for _ in range(tables)
        ]
        self._buckets: List[Dict[int, Set[Label]]] = [{} for _ in range(tables)]
        self._vectors: Dict[Label, Vector] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, label: object) -> bool:
        return label in self._vectors

    def add(self, label: Label, vector: Sequence[float]) -> None:
        if label in self._vectors:
            raise ValueError(f"label already exists: {label!r}")
        point = normalize_vector(vector, self.dimensions, self.metric)
        self._vectors[label] = point
        for table in range(self.tables):
            signature = self._signature(point, table)
            self._buckets[table].setdefault(signature, set()).add(label)

    def add_many(self, items: Iterable[Tuple[Label, Sequence[float]]]) -> None:
        for label, vector in items:
            self.add(label, vector)

    def remove(self, label: Label) -> None:
        try:
            point = self._vectors.pop(label)
        except KeyError:
            raise KeyError(label) from None
        for table in range(self.tables):
            signature = self._signature(point, table)
            bucket = self._buckets[table][signature]
            bucket.remove(label)
            if not bucket:
                del self._buckets[table][signature]

    def get_vector(self, label: Label) -> Vector:
        try:
            return self._vectors[label]
        except KeyError:
            raise KeyError(label) from None

    def search(self, vector: Sequence[float], k: int = 10, **_: object) -> List[SearchResult]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        query = normalize_vector(vector, self.dimensions, self.metric)
        candidates: Set[Label] = set()
        for table in range(self.tables):
            signature = self._signature(query, table)
            for nearby in self._nearby_signatures(signature):
                candidates.update(self._buckets[table].get(nearby, ()))
        if len(candidates) < k:
            candidates.update(self._vectors)
        results = [
            (label, distance(query, self._vectors[label], self.metric)) for label in candidates
        ]
        results.sort(key=lambda result: result[1])
        return results[:k]

    def _signature(self, vector: Sequence[float], table: int) -> int:
        signature = 0
        for bit, plane in enumerate(self._planes[table]):
            if sum(a * b for a, b in zip(vector, plane)) >= 0:
                signature |= 1 << bit
        return signature

    def _nearby_signatures(self, signature: int) -> Iterable[int]:
        yield signature
        bits = range(self.hash_size)
        for radius in range(1, self.probe_radius + 1):
            for flipped in itertools.combinations(bits, radius):
                mask = 0
                for bit in flipped:
                    mask |= 1 << bit
                yield signature ^ mask
