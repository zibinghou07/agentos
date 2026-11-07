"""Run a chat model through SearchEnv and summarise the episodes.

Works with any OpenAI-compatible endpoint: a vLLM server on the training machine
(base or LoRA-merged policy) or a hosted API (the prompt-only large-model baseline).
Tools are sent through the native ``tools`` field, so the model sees the same
rendering as during verl training (Qwen chat template, Hermes tool calls).
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentos.llm.openai_compatible import OpenAICompatibleProvider
from agentos.rl.env import SearchEnv


@dataclass
class EpisodeResult:
    trajectory: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


async def run_episode(
    env: SearchEnv,
    llm: OpenAICompatibleProvider,
    qid: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> EpisodeResult:
    obs = env.reset(qid, seed=seed)
    prompt_tokens = completion_tokens = 0
    done, error = False, None
    while not done:
        kwargs: dict[str, Any] = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens}
        if obs["tools"]:
            kwargs["tools"] = obs["tools"]
        try:
            result = await llm.chat(obs["messages"], **kwargs)
        except Exception as exc:  # recorded, never silently dropped
            error = f"{type(exc).__name__}: {exc}"
            break
        prompt_tokens += result.usage.input_tokens
        completion_tokens += result.usage.output_tokens
        message = result.raw["choices"][0]["message"]
        obs, _, done, _ = env.step({"content": message.get("content") or "", "tool_calls": message.get("tool_calls")})
    return EpisodeResult(env.trajectory(model=llm.model, error=error), prompt_tokens, completion_tokens, error)


async def run_split(
    make_env,
    llm: OpenAICompatibleProvider,
    qids: Sequence[str],
    seed: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
) -> list[EpisodeResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(qid: str) -> EpisodeResult:
        async with semaphore:
            return await run_episode(make_env(), llm, qid, seed, temperature, max_tokens)

    return await asyncio.gather(*(one(qid) for qid in qids))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarize(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Metrics for one (split, seed) run. Errored episodes count as failures and are reported."""
    trajectories = [r.trajectory for r in results]
    n = len(trajectories)
    rewards = [t["reward"] or {} for t in trajectories]
    stats = [t["stats"] for t in trajectories]
    em = [r.get("em", 0.0) for r in rewards]
    tools = Counter()
    for s in stats:
        tools.update(s["tool_counts"])
    by_type: dict[str, list[float]] = defaultdict(list)
    for t, value in zip(trajectories, em):
        by_type[t["qtype"]].append(value)
    return {
        "n": n,
        "em": _mean(em),
        "em_count": f"{int(sum(em))}/{n}",
        "f1": _mean([r.get("f1", 0.0) for r in rewards]),
        "reward": _mean([r.get("total", 0.0) for r in rewards]),
        "avg_tool_calls": _mean([s["tool_calls"] for s in stats]),
        "avg_obs_tokens": _mean([s["obs_tokens"] for s in stats]),
        "avg_turns": _mean([s["assistant_turns"] for s in stats]),
        # attempted calls = executed (incl. rejected) + malformed ones that never ran
        "valid_call_rate": 1 - sum(s["invalid_calls"] for s in stats)
        / max(sum(s["tool_calls"] + s["malformed_calls"] for s in stats), 1),
        "no_answer_rate": _mean([float(t["answer"] is None) for t in trajectories]),
        "done_reasons": dict(Counter(t["done_reason"] or "error" for t in trajectories)),
        "tool_usage": dict(tools),
        "em_by_qtype": {k: {"em": _mean(v), "count": f"{int(sum(v))}/{len(v)}"} for k, v in sorted(by_type.items())},
        "api_errors": sum(r.error is not None for r in results),
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "completion_tokens": sum(r.completion_tokens for r in results),
    }


def aggregate(per_seed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample std across seeds for every numeric metric."""
    keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float)) and k != "n"]
    out: dict[str, Any] = {"seeds": len(per_seed), "n": per_seed[0]["n"]}
    for key in keys:
        values = [run[key] for run in per_seed]
        out[key] = {"mean": _mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    out["em_count_per_seed"] = [run["em_count"] for run in per_seed]
    return out
