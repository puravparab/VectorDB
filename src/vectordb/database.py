"""High-level vector database interface."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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

    _FORMAT_VERSION = 1
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
        self.index_options = options
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
        record = dict(metadata or {})
        self._index.add(label, vector)
        self._metadata[label] = record

    def add_many(
        self,
        items: Iterable[Tuple[Label, Sequence[float], Optional[Mapping[str, Any]]]],
    ) -> None:
        """Add records atomically, rolling back the batch if one is invalid."""
        added: List[Label] = []
        try:
            for label, vector, metadata in items:
                self.add(label, vector, metadata)
                added.append(label)
        except Exception:
            for label in reversed(added):
                self.remove(label)
            raise

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

    def search_many(
        self,
        vectors: Iterable[Sequence[float]],
        k: int = 10,
        *,
        where: Optional[Filter] = None,
        **search_options: Any,
    ) -> List[List[SearchResult]]:
        return [
            self.search(vector, k=k, where=where, **search_options) for vector in vectors
        ]

    def save(self, path: Union[str, Path]) -> None:
        """Persist vectors, metadata, and index configuration as JSON."""
        records = []
        for label, metadata in self._metadata.items():
            if not self._is_json_label(label):
                raise TypeError("save supports only JSON scalar labels")
            records.append(
                {"label": label, "vector": self.get_vector(label), "metadata": metadata}
            )
        document = {
            "format_version": self._FORMAT_VERSION,
            "dimensions": self.dimensions,
            "metric": self.metric,
            "index": self.index_name,
            "index_options": self.index_options,
            "records": records,
        }
        destination = Path(path)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorDatabase":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format_version") != cls._FORMAT_VERSION:
            raise ValueError("unsupported vector database format")
        database = cls(
            document["dimensions"],
            metric=document["metric"],
            index=document["index"],
            index_options=document["index_options"],
        )
        database.add_many(
            (record["label"], record["vector"], record["metadata"])
            for record in document["records"]
        )
        return database

    @staticmethod
    def _is_json_label(value: object) -> bool:
        return (
            value is None
            or isinstance(value, (str, int, bool))
            or (isinstance(value, float) and math.isfinite(value))
        )
