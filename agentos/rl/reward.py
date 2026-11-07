"""Configurable episode reward for the search agent.

R = outcome + format + cost + overlong (+ process, ablation only)

* outcome  : exact match (or F1) against any gold answer.
* format   : penalty per invalid tool call, and for ending without <answer>.
* cost     : charged only on correct episodes, capped, so a correct answer always
             beats a wrong one and "answer without searching" is never the best policy.
* overlong : soft penalty for assistant turns beyond ``soft_turns`` (DAPO-style).
* process  : share of gold supporting paragraphs the agent retrieved, capped.

The same function scores local episodes (from ``EpisodeStats``) and verl rollouts
(from the decoded response text via ``stats_from_transcript``).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

from agentos.rl.actions import ANSWER, TOOL_CALL
from agentos.rl.data import normalize_answer
from agentos.rl.tools import DOC_ID_IN_TEXT, approx_tokens

TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
# Error prefixes produced by our tools and by verl's ToolAgentLoop.
TOOL_ERROR_PREFIXES = ("Error:", "Unknown function", "Invalid JSON in arguments", "Error executing tool")


@dataclass(frozen=True)
class RewardConfig:
    use_f1: bool = False
    invalid_call_penalty: float = 0.1
    max_format_penalty: float = 0.3
    missing_answer_penalty: float = 0.2
    cost_enabled: bool = True
    free_calls: int = 1
    call_cost: float = 0.05
    obs_token_cost_per_1k: float = 0.02
    max_cost_penalty: float = 0.3
    cost_only_when_correct: bool = True
    soft_turns: int = 6
    max_turns: int = 8
    max_overlong_penalty: float = 0.2
    process_weight: float = 0.0

    @classmethod
    def preset(cls, name: str) -> RewardConfig:
        presets = {
            "full": cls(),
            "outcome_only": cls(cost_enabled=False, max_overlong_penalty=0.0),
            "no_cost": cls(cost_enabled=False),
            "with_process": cls(process_weight=0.1),
            "no_overlong": cls(max_overlong_penalty=0.0),
            # v1.1: the first run collapsed to answering without search, so useful
            # retrieval earns a small capped bonus and the first two calls are free.
            "v11": cls(process_weight=0.1, free_calls=2),
            "v11_no_cost": cls(process_weight=0.1, free_calls=2, cost_enabled=False),
        }
        return presets[name]


@dataclass
class EpisodeStats:
    answer: str | None = None
    assistant_turns: int = 0
    tool_calls: int = 0
    invalid_calls: int = 0  # malformed calls + calls the tool rejected
    malformed_calls: int = 0
    obs_tokens: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    total: float
    outcome: float
    format: float
    cost: float
    overlong: float
    process: float
    em: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def exact_match(prediction: str | None, answers: Iterable[str]) -> float:
    if prediction is None:
        return 0.0
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(answer) for answer in answers))


def f1_score(prediction: str | None, answers: Iterable[str]) -> float:
    if prediction is None:
        return 0.0
    best = 0.0
    special = {"yes", "no", "noanswer"}
    pred_text = normalize_answer(prediction)
    pred = pred_text.split()
    for answer in answers:
        gold_text = normalize_answer(answer)
        # Official HotpotQA rule: yes/no/noanswer only score on an exact match.
        if pred_text in special or gold_text in special:
            best = max(best, float(pred_text == gold_text))
            continue
        gold = gold_text.split()
        common = sum((Counter(pred) & Counter(gold)).values())
        if common:
            precision, recall = common / len(pred), common / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def compute_reward(
    stats: EpisodeStats,
    answers: Sequence[str],
    config: RewardConfig = RewardConfig(),
    support_doc_ids: Sequence[str] = (),
) -> RewardBreakdown:
    em, f1 = exact_match(stats.answer, answers), f1_score(stats.answer, answers)
    outcome = f1 if config.use_f1 else em

    format_penalty = min(stats.invalid_calls * config.invalid_call_penalty, config.max_format_penalty)
    if stats.answer is None:
        format_penalty += config.missing_answer_penalty

    cost = 0.0
    if config.cost_enabled and (outcome > 0 or not config.cost_only_when_correct):
        raw = config.call_cost * max(stats.tool_calls - config.free_calls, 0)
        raw += config.obs_token_cost_per_1k * stats.obs_tokens / 1000
        cost = min(raw, config.max_cost_penalty)

    overlong = 0.0
    if config.max_overlong_penalty and stats.assistant_turns > config.soft_turns:
        span = max(config.max_turns - config.soft_turns, 1)
        overlong = config.max_overlong_penalty * min((stats.assistant_turns - config.soft_turns) / span, 1.0)

    process = 0.0
    if config.process_weight and support_doc_ids:
        found = len(set(support_doc_ids) & set(stats.retrieved_doc_ids))
        process = config.process_weight * found / len(set(support_doc_ids))

    total = outcome - format_penalty - cost - overlong + process
    return RewardBreakdown(total, outcome, -format_penalty, -cost, -overlong, process, em, f1)


def stats_from_transcript(response: str) -> EpisodeStats:
    """Rebuild episode statistics from a decoded multi-turn rollout (verl ``solution_str``).

    Tool responses appear as <tool_response>...</tool_response> blocks; the answer is
    the last <answer> written after the final tool response.
    """
    stats = EpisodeStats()
    responses = TOOL_RESPONSE.findall(response)
    stats.tool_calls = len(responses)
    for block in responses:
        stats.obs_tokens += approx_tokens(block)
        if block.startswith(TOOL_ERROR_PREFIXES):
            stats.invalid_calls += 1
        stats.retrieved_doc_ids.extend(DOC_ID_IN_TEXT.findall(block))
    for payload in TOOL_CALL.findall(response):
        try:
            call = json.loads(payload)
            ok = isinstance(call, dict) and isinstance(call.get("name"), str) and "arguments" in call
        except json.JSONDecodeError:
            ok = False
        if ok:
            stats.tool_counts[call["name"]] = stats.tool_counts.get(call["name"], 0) + 1
        else:
            stats.invalid_calls += 1
            stats.malformed_calls += 1
    last = response.rfind("</tool_response>")
    tail = response[last:] if last >= 0 else response
    answers = ANSWER.findall(tail)
    stats.answer = answers[-1].strip() if answers else None
    # One assistant turn per executed tool round, plus the final turn.
    stats.assistant_turns = stats.tool_calls + 1
    return stats
