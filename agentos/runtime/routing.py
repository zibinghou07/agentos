from __future__ import annotations

import math
import random
import re
from collections import Counter

from agentos.models import AgentProfile, RouteDecision, ToolMetadata


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"[\w一-鿿]+", text.lower().replace("_", " ")))


def lexical_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def beta_success_rate(successes: float, failures: float, a: float = 1.0, b: float = 1.0) -> float:
    """Posterior mean of a Beta(a, b) prior; 0.5 before any evidence."""
    return (successes + a) / (successes + failures + a + b)


class AdaptiveRouter:
    """Score = alpha*relevance + beta*success - gamma*cost - delta*latency + eta*planner_hint.

    Success, cost (LLM tokens per node) and latency come from persisted runtime
    outcomes (``agent_stats``), so routing changes as agents succeed or fail.
    """

    def __init__(
        self,
        alpha: float = 0.55,
        beta: float = 0.30,
        gamma: float = 0.08,
        delta: float = 0.07,
        eta: float = 0.15,
        epsilon: float = 0.05,
        seed: int = 42,
    ):
        self.alpha, self.beta, self.gamma, self.delta, self.eta = alpha, beta, gamma, delta, eta
        self.epsilon = epsilon
        self.random = random.Random(seed)

    @staticmethod
    def success_rate(profile: AgentProfile, a: float = 1.0, b: float = 1.0) -> float:
        return beta_success_rate(profile.successes, profile.failures, a, b)

    def route(
        self,
        task: str,
        agents: list[AgentProfile],
        budget: float | None = None,
        hint: str | None = None,
    ) -> RouteDecision:
        eligible = [a for a in agents if a.enabled and (budget is None or a.average_cost <= budget)]
        if not eligible:
            raise ValueError("no eligible agent")
        max_cost = max((a.average_cost for a in eligible), default=1.0) or 1.0
        max_latency = max((a.average_latency_ms for a in eligible), default=1.0) or 1.0
        scores = {}
        for agent in eligible:
            relevance = max(
                (lexical_similarity(task, cap) for cap in agent.capabilities), default=0.0
            )
            scores[agent.name] = (
                self.alpha * relevance
                + self.beta * self.success_rate(agent)
                - self.gamma * agent.average_cost / max_cost
                - self.delta * agent.average_latency_ms / max_latency
                + (self.eta if hint and agent.name == hint else 0.0)
            )
        exploring = len(eligible) > 1 and self.random.random() < self.epsilon
        selected = self.random.choice(eligible).name if exploring else max(scores, key=scores.get)
        confidence = 1 / (1 + math.exp(-4 * scores[selected]))
        return RouteDecision(selected_agent=selected, reason="epsilon exploration" if exploring else "adaptive score maximum", confidence=confidence, scores=scores)


def rank_tools(
    query: str,
    tools: list[ToolMetadata],
    stats: dict[str, dict[str, float]] | None = None,
    preferred: list[str] | None = None,
    *,
    relevance_weight: float = 0.6,
    success_weight: float = 0.25,
    latency_weight: float = 0.15,
    preferred_bonus: float = 0.5,
) -> list[tuple[ToolMetadata, float]]:
    """Rank candidate tools by relevance, observed reliability and latency.

    The planner's preferred tool gets a bonus rather than a hard pin, so a tool
    that keeps failing at runtime can fall below a healthy alternative.
    """
    stats = stats or {}
    preferred = preferred or []
    latencies = {
        tool.name: stats[tool.name]["total_latency_ms"] / stats[tool.name]["calls"]
        for tool in tools
        if tool.name in stats and stats[tool.name]["calls"]
    }
    max_latency = max(latencies.values(), default=0.0) or 1.0
    scored = []
    for tool in tools:
        record = stats.get(tool.name)
        success = (
            beta_success_rate(record["successes"], record["calls"] - record["successes"])
            if record
            else 0.5
        )
        relevance = lexical_similarity(query, f"{tool.name} {tool.description}")
        score = (
            relevance_weight * relevance
            + success_weight * success
            - latency_weight * latencies.get(tool.name, 0.0) / max_latency
            # Trust the planner's choice in proportion to the tool's observed reliability.
            + (preferred_bonus * min(1.0, 2 * success) if tool.name in preferred else 0.0)
        )
        scored.append((tool, score))
    return sorted(scored, key=lambda item: (-item[1], item[0].name))
