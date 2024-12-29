"""Composable predicates for metadata filtering."""

from __future__ import annotations

from typing import Any, Callable, Mapping


Metadata = Mapping[str, Any]
Filter = Callable[[Metadata], bool]
_MISSING = object()


def _read(metadata: Metadata, path: str) -> Any:
    value: Any = metadata
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def equals(path: str, expected: Any) -> Filter:
    return lambda metadata: _read(metadata, path) == expected


def not_equals(path: str, expected: Any) -> Filter:
    return lambda metadata: (
        (value := _read(metadata, path)) is not _MISSING and value != expected
    )


def greater_than(path: str, expected: Any) -> Filter:
    def predicate(metadata: Metadata) -> bool:
        value = _read(metadata, path)
        try:
            return value is not _MISSING and value > expected
        except TypeError:
            return False

    return predicate


def less_than(path: str, expected: Any) -> Filter:
    def predicate(metadata: Metadata) -> bool:
        value = _read(metadata, path)
        try:
            return value is not _MISSING and value < expected
        except TypeError:
            return False

    return predicate


def contains(path: str, expected: Any) -> Filter:
    def predicate(metadata: Metadata) -> bool:
        value = _read(metadata, path)
        try:
            return value is not _MISSING and expected in value
        except TypeError:
            return False

    return predicate


def all_of(*predicates: Filter) -> Filter:
    return lambda metadata: all(predicate(metadata) for predicate in predicates)


def any_of(*predicates: Filter) -> Filter:
    return lambda metadata: any(predicate(metadata) for predicate in predicates)


def negate(predicate: Filter) -> Filter:
    return lambda metadata: not predicate(metadata)
