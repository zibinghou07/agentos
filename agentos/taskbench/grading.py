"""Programmatic graders. No model judges: answers are matched against computed gold."""

from __future__ import annotations

import math
import re
import string
from collections import Counter

from agentos.taskbench.schema import CheckResult, DbCheck, NumberCheck, TaskSpec, TextCheck
from agentos.world.store import World

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?|(?<![\w.])-?\.\d+")
# Dates and identifiers (o042, INC-201, W3) must not leak digits that could match gold values.
_NOT_A_QUANTITY = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b[A-Za-z]+-?\d+\b")


def extract_numbers(text: str) -> list[float]:
    values = []
    for match in _NUMBER.findall(_NOT_A_QUANTITY.sub(" ", text or "")):
        cleaned = match.replace(",", "")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def normalize_tokens(text: str) -> list[str]:
    """SQuAD/HotpotQA-style normalisation: lowercase, drop punctuation and articles."""
    lowered = (text or "").lower()
    lowered = "".join(" " if ch in string.punctuation else ch for ch in lowered)
    return [token for token in lowered.split() if token not in {"a", "an", "the"}]


def contains_phrase(answer: str, phrase: str) -> bool:
    needle, haystack = normalize_tokens(phrase), normalize_tokens(answer)
    if not needle:
        return False
    width = len(needle)
    return any(haystack[i : i + width] == needle for i in range(len(haystack) - width + 1))


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize_tokens(prediction), normalize_tokens(gold)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if not pred or not ref or overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _number_matches(check: NumberCheck, values: list[float]) -> bool:
    return any(
        math.isclose(value, check.value, rel_tol=check.rel_tol, abs_tol=check.abs_tol)
        for value in values
    )


def grade(task: TaskSpec, answer: str | None, world: World) -> list[CheckResult]:
    text = answer or ""
    numbers = extract_numbers(text)
    results = []
    for check in task.checks:
        if isinstance(check, NumberCheck):
            passed = _number_matches(check, numbers)
            detail = f"expected {check.value}; found {numbers[:12]}"
        elif isinstance(check, TextCheck):
            passed = any(contains_phrase(text, alias) for alias in check.aliases)
            detail = f"expected one of {check.aliases}"
        else:
            assert isinstance(check, DbCheck)
            with world.connect(read_only=True) as conn:
                observed = float(conn.execute(check.sql, tuple(check.params)).fetchone()[0])
            passed = abs(observed - check.expected) <= check.tolerance
            detail = f"expected {check.expected}; observed {observed}"
        results.append(CheckResult(label=check.label, kind=check.kind, passed=passed, detail=detail))
    return results
