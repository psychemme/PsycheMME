"""Shared deterministic contract for explicit structured rationale coverage."""

from __future__ import annotations

from typing import Any


ANSWER_FIELD = "answer"
REASONING_FIELDS = ("G", "W", "N", "T", "S", "A", "R", "M")
OUTPUT_FIELDS = (ANSWER_FIELD, *REASONING_FIELDS)
PROTOCOL_VERSION = "psychebench-reasoning-verified-v3-explicit-rationale-coverage"

_NULL_LIKE = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "not applicable",
    "not used",
    "not relevant",
    "unused",
}


def informative_reasoning_slots(prediction: Any) -> tuple[str, ...]:
    """Return slots containing explicit, non-placeholder rationale text."""
    if not isinstance(prediction, dict):
        return ()
    present = []
    for field in REASONING_FIELDS:
        value = prediction.get(field)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().rstrip(".")
        if normalized not in _NULL_LIKE:
            present.append(field)
    return tuple(present)


def reasoning_coverage(prediction: Any) -> str:
    return "present" if informative_reasoning_slots(prediction) else "missing"
