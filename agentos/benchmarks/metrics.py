from __future__ import annotations

from statistics import mean
from typing import Iterable

import numpy as np
from scipy.stats import chi2

from agentos.benchmarks.schema import RunRecord


def summarize(records: Iterable[RunRecord]) -> dict[str, float]:
    rows = list(records)
    latencies = [row.latency_ms for row in rows]
    eligible = [row for row in rows if row.recovery_eligible]
    return {
        "tasks": len(rows), "task_success_rate": mean(row.success for row in rows),
        "step_success_rate": mean(row.step_success for row in rows),
        "tool_selection_accuracy": mean(row.tool_selection_correct for row in rows),
        "recovery_success_rate": mean(row.recovered for row in eligible) if eligible else 0.0,
        "avg_tool_calls": mean(row.tool_calls for row in rows), "avg_input_tokens": mean(row.input_tokens for row in rows),
        "avg_output_tokens": mean(row.output_tokens for row in rows), "avg_cost": mean(row.cost for row in rows),
        "p50_latency_ms": float(np.percentile(latencies, 50)), "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def bootstrap_ci(values: list[float], confidence: float = 0.95, samples: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    estimates = [rng.choice(array, len(array), replace=True).mean() for _ in range(samples)]
    tail = (1 - confidence) / 2
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))


def paired_bootstrap_difference(left: list[float], right: list[float], samples: int = 2000, seed: int = 42) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("paired samples require equal lengths")
    diffs = np.asarray(left) - np.asarray(right)
    return bootstrap_ci(diffs.tolist(), samples=samples, seed=seed)


def mcnemar(success_a: list[bool], success_b: list[bool]) -> dict[str, float]:
    if len(success_a) != len(success_b):
        raise ValueError("paired samples require equal lengths")
    b = sum(a and not c for a, c in zip(success_a, success_b))
    c = sum(not a and c for a, c in zip(success_a, success_b))
    statistic = ((abs(b - c) - 1) ** 2 / (b + c)) if b + c else 0.0
    return {"discordant_a_only": b, "discordant_b_only": c, "chi_square": statistic, "p_value": float(chi2.sf(statistic, 1)) if b + c else 1.0}
