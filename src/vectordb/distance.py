"""Vector validation and distance functions shared by search indexes."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple


Vector = Tuple[float, ...]
SUPPORTED_METRICS = frozenset({"euclidean", "cosine"})


def validate_dimensions(dimensions: int) -> None:
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")


def validate_metric(metric: str) -> None:
    if metric not in SUPPORTED_METRICS:
        raise ValueError("metric must be 'euclidean' or 'cosine'")


def normalize_vector(vector: Iterable[float], dimensions: int, metric: str) -> Vector:
    try:
        point = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise ValueError("vector coordinates must be real numbers") from exc
    if len(point) != dimensions:
        raise ValueError(f"expected {dimensions} dimensions, got {len(point)}")
    if not all(math.isfinite(value) for value in point):
        raise ValueError("vector coordinates must be finite")
    if metric == "cosine" and not any(point):
        raise ValueError("cosine distance is undefined for a zero vector")
    return point


def squared_euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(squared_euclidean(left, right))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine distance is undefined for a zero vector")
    return 1.0 - dot / (left_norm * right_norm)


def distance(left: Sequence[float], right: Sequence[float], metric: str) -> float:
    if metric == "euclidean":
        return euclidean(left, right)
    if metric == "cosine":
        return cosine(left, right)
    raise ValueError("metric must be 'euclidean' or 'cosine'")
