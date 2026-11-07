"""Gym-style multi-turn search environment.

    env = SearchEnv(records, corpus, embeddings)
    obs = env.reset(qid)                     # {"messages": [...], "tools": [...]}
    obs, reward, done, info = env.step(turn) # turn: assistant text or API message dict

An action is one assistant turn: a tool call or a final <answer>. The reward is
0 until the episode ends, then the full breakdown from ``agentos.rl.reward``.
Each ``reset`` builds a fresh searcher and statistics, so episodes share no state.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from agentos.rl.actions import Action, parse_action
from agentos.rl.reward import EpisodeStats, RewardConfig, compute_reward
from agentos.rl.tools import Corpus, EmbeddingStore, PoolSearcher, approx_tokens, prompt_tool_schemas

SYSTEM_PROMPT = (
    "You answer multi-hop questions using a private collection of Wikipedia paragraphs. "
    "Search with the tools to find evidence, calling at most one tool per turn; most "
    "questions need a search for each entity they mention. When you have the evidence, "
    "your final reply must contain only the answer inside answer tags, for example "
    "<answer>Paris</answer>. Always finish this way, also right after reading tool "
    "results, and give your best guess if you are unsure. The answer must be a short span: "
    "an entity, a date, a number, or yes/no. Do not explain."
)
CLOSED_BOOK_PROMPT = (
    "Answer the multi-hop question from your own knowledge. "
    "Reply with only the final answer inside <answer></answer>. "
    "The answer must be a short span: an entity, a date, a number, or yes/no. Do not explain."
)


def initial_messages(question: str, closed_book: bool = False) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": CLOSED_BOOK_PROMPT if closed_book else SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]


class SearchEnv:
    def __init__(
        self,
        records: dict[str, dict[str, Any]],
        corpus: Corpus,
        embeddings: EmbeddingStore,
        reward_config: RewardConfig = RewardConfig(),
        closed_book: bool = False,
    ):
        self.records, self.corpus, self.embeddings = records, corpus, embeddings
        self.reward_config, self.closed_book = reward_config, closed_book
        self.record: dict[str, Any] | None = None
        self.done = True

    @property
    def tools(self) -> list[dict[str, Any]]:
        # Same tool text as the verl training prompt; arguments are still validated in full.
        return [] if self.closed_book else prompt_tool_schemas()

    def reset(self, qid: str, seed: int | None = None) -> dict[str, Any]:
        self.record = self.records[qid]
        self.seed = seed
        self.searcher = PoolSearcher(self.corpus, self.record["pool"], self.embeddings)
        self.stats = EpisodeStats()
        self.messages = initial_messages(self.record["question"], self.closed_book)
        self.turns: list[dict[str, Any]] = []
        self.done = False
        self.done_reason: str | None = None
        return {"messages": list(self.messages), "tools": self.tools}

    def support_doc_ids(self) -> list[str]:
        assert self.record is not None
        support = set(self.record["support_titles"])
        return [doc_id for doc_id in self.record["own_doc_ids"] if self.corpus[doc_id].title in support]

    def _assistant_message(self, turn: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(turn, str):
            return {"role": "assistant", "content": turn}
        message = {"role": "assistant", "content": turn.get("content") or ""}
        if turn.get("tool_calls"):
            # Keep only the executed call: APIs require a tool reply for every listed call.
            message["tool_calls"] = turn["tool_calls"][:1]
        return message

    def step(self, turn: str | dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode is finished; call reset() first")
        stats, config = self.stats, self.reward_config
        action: Action = parse_action(turn)
        stats.assistant_turns += 1
        stats.invalid_calls += action.malformed_calls
        stats.malformed_calls += action.malformed_calls
        assistant = self._assistant_message(turn)
        self.messages.append(assistant)
        record: dict[str, Any] = {"assistant": assistant, "action": action.kind}
        reason = None

        if action.kind == "tool_call":
            if self.closed_book:
                result_text, ok, doc_ids, error_kind = "Error: no tools are available.", False, (), "no_tools"
            else:
                result = self.searcher.call(action.name or "", action.arguments)
                result_text, ok, doc_ids, error_kind = result.text, result.ok, result.doc_ids, result.error_kind
            stats.tool_calls += 1
            stats.obs_tokens += approx_tokens(result_text)
            stats.invalid_calls += 0 if ok else 1
            stats.retrieved_doc_ids.extend(doc_ids)
            stats.tool_counts[action.name or "?"] = stats.tool_counts.get(action.name or "?", 0) + 1
            tool_message: dict[str, Any] = {"role": "tool", "content": result_text}
            if assistant.get("tool_calls"):
                tool_message["tool_call_id"] = assistant["tool_calls"][0].get("id", "call_0")
            self.messages.append(tool_message)
            record.update(
                tool_call={"name": action.name, "arguments": action.arguments},
                observation=result_text,
                tool_ok=ok,
                error_kind=error_kind,
            )
            if stats.assistant_turns >= config.max_turns:
                reason = "max_turns"
        elif action.kind == "answer":
            stats.answer = action.answer
            reason = "answer"
        else:  # malformed tool call or plain text: the episode ends without an answer
            reason = action.kind

        info: dict[str, Any] = {"action": action.kind, "error": action.error}
        reward = 0.0
        if reason is not None:
            self.done, self.done_reason = True, reason
            breakdown = compute_reward(stats, self.record["answers"], config, self.support_doc_ids())
            reward = breakdown.total
            info.update(done_reason=reason, reward=breakdown.as_dict(), stats=asdict(stats))
        record.update(step_reward=reward, done=self.done)
        self.turns.append(record)
        return {"messages": list(self.messages), "tools": self.tools}, reward, self.done, info

    def trajectory(self, **extra: Any) -> dict[str, Any]:
        """One JSONL line: messages, per-turn tool calls/observations, final reward."""
        assert self.record is not None
        return {
            "qid": self.record["qid"],
            "source": self.record["source"],
            "qtype": self.record["qtype"],
            "question": self.record["question"],
            "gold": self.record["answers"],
            "seed": self.seed,
            "closed_book": self.closed_book,
            "messages": self.messages,
            "turns": self.turns,
            "answer": self.stats.answer,
            "done": self.done,
            "reward": compute_reward(self.stats, self.record["answers"], self.reward_config, self.support_doc_ids()).as_dict()
            if self.done
            else None,
            "stats": asdict(self.stats),
            "done_reason": self.done_reason,
            **extra,
        }


def dumps(trajectory: dict[str, Any]) -> str:
    return json.dumps(trajectory, ensure_ascii=False, default=str)
