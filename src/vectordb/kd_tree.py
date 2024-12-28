"""Exact k-dimensional tree search for low-dimensional Euclidean vectors."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from .distance import Vector, normalize_vector, squared_euclidean, validate_dimensions


Label = Hashable
SearchResult = Tuple[Label, float]


@dataclass
class _Node:
    label: Label
    axis: int
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class KDTreeIndex:
    """A balanced, lazily rebuilt KD-tree with exact nearest-neighbor search."""

    metric = "euclidean"

    def __init__(self, dimensions: int) -> None:
        validate_dimensions(dimensions)
        self.dimensions = dimensions
        self._vectors: Dict[Label, Vector] = {}
        self._root: Optional[_Node] = None
        self._dirty = False

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
        self._dirty = True

    def add_many(self, items: Iterable[Tuple[Label, Sequence[float]]]) -> None:
        for label, vector in items:
            self.add(label, vector)

    def remove(self, label: Label) -> None:
        try:
            del self._vectors[label]
        except KeyError:
            raise KeyError(label) from None
        self._dirty = True

    def get_vector(self, label: Label) -> Vector:
        try:
            return self._vectors[label]
        except KeyError:
            raise KeyError(label) from None

    def search(self, vector: Sequence[float], k: int = 10, **_: object) -> List[SearchResult]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        query = normalize_vector(vector, self.dimensions, self.metric)
        self._rebuild_if_needed()
        best: List[Tuple[float, int, Label]] = []
        counter = 0

        def visit(node: Optional[_Node]) -> None:
            nonlocal counter
            if node is None:
                return
            point = self._vectors[node.label]
            distance_squared = squared_euclidean(query, point)
            heapq.heappush(best, (-distance_squared, counter, node.label))
            counter += 1
            if len(best) > k:
                heapq.heappop(best)

            delta = query[node.axis] - point[node.axis]
            near, far = (node.left, node.right) if delta < 0 else (node.right, node.left)
            visit(near)
            worst = -best[0][0] if len(best) == k else float("inf")
            if delta * delta <= worst:
                visit(far)

        visit(self._root)
        results = [(label, (-negative) ** 0.5) for negative, _, label in best]
        results.sort(key=lambda result: result[1])
        return results

    def _rebuild_if_needed(self) -> None:
        if not self._dirty:
            return

        def build(labels: List[Label], depth: int) -> Optional[_Node]:
            if not labels:
                return None
            axis = depth % self.dimensions
            labels.sort(key=lambda label: self._vectors[label][axis])
            middle = len(labels) // 2
            return _Node(
                labels[middle],
                axis,
                build(labels[:middle], depth + 1),
                build(labels[middle + 1 :], depth + 1),
            )

        self._root = build(list(self._vectors), 0)
        self._dirty = False
