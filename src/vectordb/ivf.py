"""Inverted-file (IVF-flat) approximate nearest-neighbor search."""

from __future__ import annotations

import random
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from .distance import Vector, distance, normalize_vector, validate_dimensions, validate_metric


Label = Hashable
SearchResult = Tuple[Label, float]


class IVFFlatIndex:
    """Cluster vectors and search only the closest inverted lists."""

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "euclidean",
        n_lists: int = 16,
        n_probe: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        validate_dimensions(dimensions)
        validate_metric(metric)
        if isinstance(n_lists, bool) or not isinstance(n_lists, int) or n_lists <= 0:
            raise ValueError("n_lists must be a positive integer")
        if isinstance(n_probe, bool) or not isinstance(n_probe, int) or n_probe <= 0:
            raise ValueError("n_probe must be a positive integer")
        self.dimensions = dimensions
        self.metric = metric
        self.n_lists = n_lists
        self.n_probe = n_probe
        self._rng = random.Random(seed)
        self._vectors: Dict[Label, Vector] = {}
        self._centroids: List[Vector] = []
        self._buckets: List[Set[Label]] = []

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, label: object) -> bool:
        return label in self._vectors

    def add(self, label: Label, vector: Sequence[float]) -> None:
        if label in self._vectors:
            raise ValueError(f"label already exists: {label!r}")
        point = normalize_vector(vector, self.dimensions, self.metric)
        self._vectors[label] = point
        if self._centroids:
            self._buckets[self._closest_centroid(point)].add(label)

    def add_many(self, items: Iterable[Tuple[Label, Sequence[float]]]) -> None:
        for label, vector in items:
            self.add(label, vector)

    def train(self, *, iterations: int = 12) -> None:
        """Train coarse centroids with k-means and rebuild inverted lists."""
        if not self._vectors:
            self._centroids = []
            self._buckets = []
            return
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        count = min(self.n_lists, len(self._vectors))
        points = list(self._vectors.values())
        centroids = list(self._rng.sample(points, count))
        assignments = [0] * len(points)

        for _ in range(iterations):
            new_assignments = [
                min(range(count), key=lambda cluster: distance(point, centroids[cluster], self.metric))
                for point in points
            ]
            if new_assignments == assignments and _ > 0:
                break
            assignments = new_assignments
            for cluster in range(count):
                members = [point for point, assigned in zip(points, assignments) if assigned == cluster]
                if not members:
                    centroids[cluster] = self._rng.choice(points)
                    continue
                center = tuple(
                    sum(point[coordinate] for point in members) / len(members)
                    for coordinate in range(self.dimensions)
                )
                if self.metric == "cosine" and not any(center):
                    center = members[0]
                centroids[cluster] = center

        self._centroids = centroids
        self._buckets = [set() for _ in centroids]
        for label, point in self._vectors.items():
            self._buckets[self._closest_centroid(point)].add(label)

    def search(self, vector: Sequence[float], k: int = 10, **_: object) -> List[SearchResult]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        query = normalize_vector(vector, self.dimensions, self.metric)
        if not self._vectors:
            return []
        if not self._centroids:
            self.train()
        ranked_lists = sorted(
            range(len(self._centroids)),
            key=lambda cluster: distance(query, self._centroids[cluster], self.metric),
        )
        candidates: Set[Label] = set()
        for position, cluster in enumerate(ranked_lists):
            if position >= self.n_probe and len(candidates) >= k:
                break
            candidates.update(self._buckets[cluster])
        results = [
            (label, distance(query, self._vectors[label], self.metric)) for label in candidates
        ]
        results.sort(key=lambda result: result[1])
        return results[:k]

    def remove(self, label: Label) -> None:
        try:
            point = self._vectors.pop(label)
        except KeyError:
            raise KeyError(label) from None
        if self._centroids:
            self._buckets[self._closest_centroid(point)].discard(label)

    def get_vector(self, label: Label) -> Vector:
        try:
            return self._vectors[label]
        except KeyError:
            raise KeyError(label) from None

    def _closest_centroid(self, point: Sequence[float]) -> int:
        return min(
            range(len(self._centroids)),
            key=lambda cluster: distance(point, self._centroids[cluster], self.metric),
        )
