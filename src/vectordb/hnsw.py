"""A dependency-free implementation of Hierarchical Navigable Small Worlds.

HNSW builds a hierarchy of proximity graphs.  Searches begin in the sparse
upper layers and use those layers to find a good entry point into the denser
bottom layer.  The implementation favors readability and a small API over
micro-optimizations, making it suitable as the in-memory index for this
project and as a reference implementation.
"""

from __future__ import annotations

import heapq
import json
import math
import random
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple, Union


Label = Hashable
SearchResult = Tuple[Label, float]


class HNSWIndex:
    """An in-memory approximate nearest-neighbor index.

    Args:
        dimensions: Number of coordinates in every vector.
        m: Maximum number of links per node on upper layers.  Layer zero can
            contain up to ``2 * m`` links.
        ef_construction: Candidate-list size used while inserting vectors.
        metric: Either ``"euclidean"`` or ``"cosine"``.
        seed: Seed used for deterministic level assignment.

    Labels must be unique hashable values.  Search returns ``(label,
    distance)`` pairs ordered from nearest to farthest.
    """

    _FORMAT_VERSION = 1

    def __init__(
        self,
        dimensions: int,
        *,
        m: int = 16,
        ef_construction: int = 200,
        metric: str = "euclidean",
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if isinstance(m, bool) or not isinstance(m, int) or m < 2:
            raise ValueError("m must be an integer greater than one")
        if (
            isinstance(ef_construction, bool)
            or not isinstance(ef_construction, int)
            or ef_construction < m
        ):
            raise ValueError("ef_construction must be an integer greater than or equal to m")
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")

        self.dimensions = dimensions
        self.m = m
        self.ef_construction = ef_construction
        self.metric = metric
        self._rng = random.Random(seed)
        self._level_multiplier = 1.0 / math.log(m)
        self._vectors: Dict[Label, Tuple[float, ...]] = {}
        self._levels: Dict[Label, int] = {}
        self._links: Dict[Label, List[Set[Label]]] = {}
        self._entry_point: Optional[Label] = None
        self._max_level = -1

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, label: object) -> bool:
        return label in self._vectors

    def add(self, label: Label, vector: Sequence[float]) -> None:
        """Insert one vector under a unique label."""
        if label in self._vectors:
            raise ValueError(f"label already exists: {label!r}")
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError("label must be hashable") from exc

        point = self._validate_vector(vector)
        new_level = self._random_level()
        self._vectors[label] = point
        self._levels[label] = new_level
        self._links[label] = [set() for _ in range(new_level + 1)]

        if self._entry_point is None:
            self._entry_point = label
            self._max_level = new_level
            return

        entry = self._entry_point
        assert entry is not None

        # Greedily descend through layers that the new node does not occupy.
        for level in range(self._max_level, new_level, -1):
            entry = self._greedy_closest(point, entry, level)

        # Search, connect, and descend through layers shared by both nodes.
        for level in range(min(new_level, self._max_level), -1, -1):
            candidates = self._search_layer(point, [entry], self.ef_construction, level)
            neighbors = self._select_neighbors(candidates, self._max_connections(level))
            self._links[label][level].update(neighbors)
            for neighbor in neighbors:
                self._links[neighbor][level].add(label)
                self._prune(neighbor, level)
            if candidates:
                entry = candidates[0][1]

        if new_level > self._max_level:
            self._entry_point = label
            self._max_level = new_level

    def add_many(self, items: Iterable[Tuple[Label, Sequence[float]]]) -> None:
        """Insert ``(label, vector)`` pairs in iteration order."""
        for label, vector in items:
            self.add(label, vector)

    def search(self, vector: Sequence[float], k: int = 10, *, ef: Optional[int] = None) -> List[SearchResult]:
        """Return up to ``k`` approximate nearest neighbors.

        ``ef`` controls the query-time accuracy/speed tradeoff.  It must be at
        least ``k``; larger values generally improve recall.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if ef is None:
            ef = max(50, k)
        if isinstance(ef, bool) or not isinstance(ef, int) or ef < k:
            raise ValueError("ef must be an integer greater than or equal to k")
        point = self._validate_vector(vector)
        if self._entry_point is None:
            return []

        entry = self._entry_point
        for level in range(self._max_level, 0, -1):
            entry = self._greedy_closest(point, entry, level)

        results = self._search_layer(point, [entry], ef, 0)
        return [(label, self._external_distance(distance)) for distance, label in results[:k]]

    def remove(self, label: Label) -> None:
        """Remove a label and all graph edges incident to it."""
        if label not in self._vectors:
            raise KeyError(label)
        for level, neighbors in enumerate(self._links[label]):
            # The removed node may be the bridge between two graph regions.
            # Join its former neighbors before pruning their link lists.
            former_neighbors = list(neighbors)
            for neighbor in neighbors:
                self._links[neighbor][level].discard(label)
                self._links[neighbor][level].update(
                    candidate for candidate in former_neighbors if candidate != neighbor
                )
            for neighbor in former_neighbors:
                self._prune(neighbor, level)
        del self._links[label]
        del self._levels[label]
        del self._vectors[label]

        if not self._vectors:
            self._entry_point = None
            self._max_level = -1
            return
        if label == self._entry_point:
            self._max_level = max(self._levels.values())
            self._entry_point = next(
                node for node, level in self._levels.items() if level == self._max_level
            )

    def get_vector(self, label: Label) -> Tuple[float, ...]:
        """Return the stored vector for ``label``."""
        try:
            return self._vectors[label]
        except KeyError:
            raise KeyError(label) from None

    def save(self, path: Union[str, Path]) -> None:
        """Save the complete index as portable JSON.

        JSON persistence supports string, integer, float, boolean, and null
        labels.  Tuple or application-specific labels should be mapped to one
        of those types before saving.
        """
        nodes = []
        for label, vector in self._vectors.items():
            if not self._is_json_scalar(label):
                raise TypeError("save supports only JSON scalar labels")
            nodes.append(
                {
                    "label": label,
                    "vector": vector,
                    "level": self._levels[label],
                    "links": [list(neighbors) for neighbors in self._links[label]],
                }
            )
        data = {
            "format_version": self._FORMAT_VERSION,
            "dimensions": self.dimensions,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "entry_point": self._entry_point,
            "max_level": self._max_level,
            "rng_state": self._state_to_json(self._rng.getstate()),
            "nodes": nodes,
        }
        Path(path).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HNSWIndex":
        """Load an index written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != cls._FORMAT_VERSION:
            raise ValueError("unsupported HNSW index format")
        index = cls(
            data["dimensions"],
            m=data["m"],
            ef_construction=data["ef_construction"],
            metric=data["metric"],
        )
        for node in data["nodes"]:
            label = node["label"]
            index._vectors[label] = tuple(node["vector"])
            index._levels[label] = node["level"]
            index._links[label] = [set(neighbors) for neighbors in node["links"]]
        index._entry_point = data["entry_point"]
        index._max_level = data["max_level"]
        index._rng.setstate(index._state_from_json(data["rng_state"]))
        index._validate_loaded_graph()
        return index

    def _validate_vector(self, vector: Sequence[float]) -> Tuple[float, ...]:
        try:
            point = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise ValueError("vector coordinates must be real numbers") from exc
        if len(point) != self.dimensions:
            raise ValueError(f"expected {self.dimensions} dimensions, got {len(point)}")
        if not all(math.isfinite(value) for value in point):
            raise ValueError("vector coordinates must be finite")
        if self.metric == "cosine" and not any(point):
            raise ValueError("cosine distance is undefined for a zero vector")
        return point

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        if self.metric == "euclidean":
            # Squared distance preserves ordering and avoids square roots while
            # traversing the graph.  Results are converted at the API boundary.
            return sum((a - b) ** 2 for a, b in zip(left, right))
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return 1.0 - dot / (left_norm * right_norm)

    def _external_distance(self, distance: float) -> float:
        return math.sqrt(distance) if self.metric == "euclidean" else distance

    def _random_level(self) -> int:
        # random() can return 0.0, so protect log from that rare endpoint.
        sample = max(self._rng.random(), 1e-300)
        return int(-math.log(sample) * self._level_multiplier)

    def _greedy_closest(self, query: Sequence[float], entry: Label, level: int) -> Label:
        best = entry
        best_distance = self._distance(query, self._vectors[entry])
        improved = True
        while improved:
            improved = False
            for neighbor in self._links[best][level]:
                distance = self._distance(query, self._vectors[neighbor])
                if distance < best_distance:
                    best = neighbor
                    best_distance = distance
                    improved = True
                    break
        return best

    def _search_layer(
        self, query: Sequence[float], entries: Iterable[Label], ef: int, level: int
    ) -> List[Tuple[float, Label]]:
        visited: Set[Label] = set()
        candidates: List[Tuple[float, int, Label]] = []
        # The counter prevents Python from comparing unlike label types when
        # two distances are equal.
        counter = 0
        best: List[Tuple[float, int, Label]] = []
        for entry in entries:
            if entry in visited:
                continue
            visited.add(entry)
            distance = self._distance(query, self._vectors[entry])
            heapq.heappush(candidates, (distance, counter, entry))
            heapq.heappush(best, (-distance, counter, entry))
            counter += 1

        while candidates:
            candidate_distance, _, candidate = heapq.heappop(candidates)
            worst_distance = -best[0][0]
            if len(best) >= ef and candidate_distance > worst_distance:
                break
            for neighbor in self._links[candidate][level]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                distance = self._distance(query, self._vectors[neighbor])
                if len(best) < ef or distance < -best[0][0]:
                    heapq.heappush(candidates, (distance, counter, neighbor))
                    heapq.heappush(best, (-distance, counter, neighbor))
                    counter += 1
                    if len(best) > ef:
                        heapq.heappop(best)

        results = [(-negative_distance, label) for negative_distance, _, label in best]
        return sorted(results, key=lambda result: result[0])

    @staticmethod
    def _select_neighbors(candidates: List[Tuple[float, Label]], limit: int) -> List[Label]:
        return [label for _, label in candidates[:limit]]

    def _max_connections(self, level: int) -> int:
        return self.m * 2 if level == 0 else self.m

    def _prune(self, label: Label, level: int) -> None:
        links = self._links[label][level]
        limit = self._max_connections(level)
        if len(links) <= limit:
            return
        origin = self._vectors[label]
        nearest = sorted(links, key=lambda neighbor: self._distance(origin, self._vectors[neighbor]))[
            :limit
        ]
        removed = links.difference(nearest)
        self._links[label][level] = set(nearest)
        for neighbor in removed:
            self._links[neighbor][level].discard(label)

    def _validate_loaded_graph(self) -> None:
        labels = set(self._vectors)
        if set(self._levels) != labels or set(self._links) != labels:
            raise ValueError("invalid HNSW index: inconsistent node sets")
        if bool(labels) != (self._entry_point is not None):
            raise ValueError("invalid HNSW index: bad entry point")
        if self._entry_point is not None and self._entry_point not in labels:
            raise ValueError("invalid HNSW index: missing entry point")
        for label in labels:
            if len(self._links[label]) != self._levels[label] + 1:
                raise ValueError("invalid HNSW index: bad level data")
            for level, neighbors in enumerate(self._links[label]):
                for neighbor in neighbors:
                    if neighbor not in labels or self._levels[neighbor] < level:
                        raise ValueError("invalid HNSW index: bad link")

    @staticmethod
    def _is_json_scalar(value: object) -> bool:
        return (
            value is None
            or isinstance(value, (str, int, bool))
            or (isinstance(value, float) and math.isfinite(value))
        )

    @classmethod
    def _state_to_json(cls, value: object) -> object:
        if isinstance(value, tuple):
            return [cls._state_to_json(item) for item in value]
        return value

    @classmethod
    def _state_from_json(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(cls._state_from_json(item) for item in value)
        return value
