from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

import numpy as np

from agentos.benchmarks.metrics import bootstrap_ci, mcnemar, paired_bootstrap_difference
from agentos.taskbench.schema import RunRecord


def _summary(rows: list[RunRecord]) -> dict[str, Any]:
    latencies = [row.latency_ms for row in rows]
    success = [float(row.success) for row in rows]
    low, high = bootstrap_ci(success) if len(success) > 1 else (mean(success), mean(success))
    summary = {
        "runs": len(rows),
        "task_success_rate": mean(success),
        "task_success_ci95": [low, high],
        "avg_check_score": mean(row.score for row in rows),
        "avg_llm_calls": mean(row.llm_calls for row in rows),
        "avg_input_tokens": mean(row.input_tokens for row in rows),
        "avg_output_tokens": mean(row.output_tokens for row in rows),
        "avg_total_tokens": mean(row.input_tokens + row.output_tokens for row in rows),
        "avg_reasoning_tokens": mean(row.reasoning_tokens for row in rows),
        "avg_tool_calls": mean(row.tool_calls for row in rows),
        "avg_tool_errors": mean(row.tool_errors for row in rows),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "crashed_or_timed_out": sum(row.error is not None for row in rows),
        "usage_estimated_runs": sum(row.usage_estimated for row in rows),
    }
    hotpot = [row for row in rows if row.family == "hotpotqa_agentic"]
    if hotpot:
        summary["hotpot_answer_f1"] = mean(row.extra.get("answer_f1", 0.0) for row in hotpot)
    actions = [row for row in rows if row.family == "policy_action"]
    if actions:
        # Over-triggering: a refund or ticket beyond what the policy required.
        summary["side_effect_violations"] = sum(
            any(c.kind == "db" and not c.passed for c in row.checks) for row in actions
        )
    agentos = [row for row in rows if row.extra.get("plan_shape")]
    if agentos:
        summary["avg_plan_nodes"] = mean(row.extra["plan_shape"]["nodes"] for row in agentos)
        summary["avg_plan_parallelism"] = mean(row.extra["plan_shape"]["parallelism"] for row in agentos)
        summary["avg_replans"] = mean(row.extra.get("replans", 0) for row in agentos)
        summary["avg_node_reexecutions"] = mean(row.extra.get("node_reexecutions", 0) for row in agentos)
    return summary


def summarize(records: list[RunRecord]) -> dict[str, Any]:
    by_method: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_method[record.method].append(record)
    families = sorted({record.family for record in records})
    result: dict[str, Any] = {"methods": {}, "by_family": {}, "paired_tests": {}}
    for method, rows in sorted(by_method.items()):
        result["methods"][method] = _summary(rows)
        result["by_family"][method] = {
            family: {
                "runs": len(subset),
                "task_success_rate": mean(r.success for r in subset),
                "avg_total_tokens": mean(r.input_tokens + r.output_tokens for r in subset),
            }
            for family in families
            if (subset := [r for r in rows if r.family == family])
        }
    if "agentos" in by_method:
        reference = {(r.run_id, r.task_id): r for r in by_method["agentos"]}
        for method, rows in sorted(by_method.items()):
            if method == "agentos":
                continue
            pairs = [(reference[key], r) for r in rows if (key := (r.run_id, r.task_id)) in reference]
            if len(pairs) < 2:
                continue
            ours = [float(a.success) for a, _ in pairs]
            theirs = [float(b.success) for _, b in pairs]
            tokens_ours = [a.input_tokens + a.output_tokens for a, _ in pairs]
            tokens_theirs = [b.input_tokens + b.output_tokens for _, b in pairs]
            result["paired_tests"][f"agentos_vs_{method}"] = {
                "pairs": len(pairs),
                "success_diff": mean(ours) - mean(theirs),
                "success_diff_ci95": list(paired_bootstrap_difference(ours, theirs)),
                "mcnemar": mcnemar([bool(x) for x in ours], [bool(x) for x in theirs]),
                "token_ratio": (sum(tokens_ours) / sum(tokens_theirs)) if sum(tokens_theirs) else None,
            }
    return result
