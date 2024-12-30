"""High-level vector database interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence

from .brute_force import BruteForceIndex
from .hnsw import HNSWIndex
from .ivf import IVFFlatIndex
from .kd_tree import KDTreeIndex
from .lsh import LSHIndex
from .filters import Filter


Label = Hashable


@dataclass(frozen=True)
class SearchResult:
    label: Label
    distance: float
    metadata: Mapping[str, Any]


class VectorDatabase:
    """Store vectors and metadata behind a selectable search index."""

    _INDEXES = {
        "brute_force": BruteForceIndex,
        "hnsw": HNSWIndex,
        "ivf": IVFFlatIndex,
        "kd_tree": KDTreeIndex,
        "lsh": LSHIndex,
    }

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "euclidean",
        index: str = "hnsw",
        index_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if index not in self._INDEXES:
            choices = ", ".join(sorted(self._INDEXES))
            raise ValueError(f"index must be one of: {choices}")
        if index == "kd_tree" and metric != "euclidean":
            raise ValueError("kd_tree supports only euclidean distance")
        options = dict(index_options or {})
        constructor = self._INDEXES[index]
        if index == "kd_tree":
            self._index = constructor(dimensions, **options)
        else:
            self._index = constructor(dimensions, metric=metric, **options)
        self.dimensions = dimensions
        self.metric = metric
        self.index_name = index
        self._metadata: Dict[Label, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, label: object) -> bool:
        return label in self._index

    def add(
        self,
        label: Label,
        vector: Sequence[float],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._index.add(label, vector)
        self._metadata[label] = dict(metadata or {})

    def remove(self, label: Label) -> None:
        self._index.remove(label)
        del self._metadata[label]

    def get_vector(self, label: Label):
        return self._index.get_vector(label)

    def get_metadata(self, label: Label) -> Dict[str, Any]:
        if label not in self._metadata:
            raise KeyError(label)
        return dict(self._metadata[label])

    def update_metadata(self, label: Label, values: Mapping[str, Any]) -> None:
        if label not in self._metadata:
            raise KeyError(label)
        self._metadata[label].update(values)

    def search(
        self,
        vector: Sequence[float],
        k: int = 10,
        *,
        where: Optional[Filter] = None,
        **search_options: Any,
    ) -> List[SearchResult]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        candidate_count = max(k, len(self)) if where is not None else k
        if self.index_name == "hnsw":
            search_options.setdefault("ef", max(50, candidate_count))
        candidates = self._index.search(vector, k=candidate_count, **search_options)
        results = []
        for label, score in candidates:
            metadata = self._metadata[label]
            if where is None or where(metadata):
                results.append(SearchResult(label, score, dict(metadata)))
                if len(results) == k:
                    break
        return results
