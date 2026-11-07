"""verl custom reward: scores a decoded multi-turn rollout with the shared reward.

Config (hydra overrides):
    reward.custom_reward_function.path=agentos/rl/verl_reward.py
    reward.custom_reward_function.name=compute_score
    ++reward.custom_reward_function.reward_kwargs.preset=full
    ++reward.custom_reward_function.reward_kwargs.max_turns=8
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agentos.rl.reward import RewardConfig, compute_reward, stats_from_transcript
from agentos.rl.tools import TOOL_NAMES


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    preset: str = "full",
    soft_turns: int | None = None,
    max_turns: int | None = None,
    **_: Any,
) -> dict[str, float]:
    extra_info = extra_info or {}
    answers = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    stats = stats_from_transcript(solution_str)
    # Parquet hands lists back as numpy arrays, so never test them for truthiness.
    support = extra_info.get("support_doc_ids")
    support = [] if support is None else [str(doc) for doc in support]
    config = RewardConfig.preset(preset)
    # Keep the overlong penalty aligned with rollout.multi_turn.max_assistant_turns.
    config = replace(config, soft_turns=soft_turns or config.soft_turns, max_turns=max_turns or config.max_turns)
    breakdown = compute_reward(stats, [str(a) for a in answers], config, support)
    result = {"score": breakdown.total, **{f"r_{k}": v for k, v in breakdown.as_dict().items() if k != "total"}}
    result.update(
        tool_calls=float(stats.tool_calls),
        invalid_calls=float(stats.invalid_calls),
        obs_tokens=float(stats.obs_tokens),
        assistant_turns=float(stats.assistant_turns),
        no_answer=float(stats.answer is None),
        # Fixed keys: verl stacks these per batch, so every sample must report all of them.
        **{f"calls_{name}": float(stats.tool_counts.get(name, 0)) for name in TOOL_NAMES},
    )
    return result
