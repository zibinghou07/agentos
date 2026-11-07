import json

import pytest

from agentos.rl.actions import parse_action
from agentos.rl.data import Paragraph, QAItem, attach_pools, doc_id, leakage_report, stratified_sample
from agentos.rl.env import SearchEnv
from agentos.rl.reward import (
    EpisodeStats,
    RewardConfig,
    compute_reward,
    exact_match,
    f1_score,
    stats_from_transcript,
)
from agentos.rl.tools import Corpus, Document, EmbeddingStore, HashingEncoder


def call(name, **arguments):
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


@pytest.fixture
def env_parts():
    docs = {
        "d0000000001": Document("d0000000001", "Scott Derrickson", "Scott Derrickson is an American director."),
        "d0000000002": Document("d0000000002", "Ed Wood", "Edward Davis Wood Jr. was an American filmmaker."),
        "d0000000003": Document("d0000000003", "Tyler Bates", "Tyler Bates is an American musician."),
        "d0000000004": Document("d0000000004", "Paris", "Paris is the capital of France."),
    }
    records = {
        "q1": {
            "qid": "q1", "source": "hotpotqa", "qtype": "comparison",
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "answers": ["yes"], "support_titles": ["Scott Derrickson", "Ed Wood"],
            "own_doc_ids": ["d0000000001", "d0000000002", "d0000000003"],
            "pool": ["d0000000001", "d0000000002", "d0000000003", "d0000000004"],
        },
        "q2": {
            "qid": "q2", "source": "hotpotqa", "qtype": "bridge",
            "question": "What is the capital of France?", "answers": ["Paris"],
            "support_titles": ["Paris"], "own_doc_ids": ["d0000000004"], "pool": ["d0000000004", "d0000000003"],
        },
    }
    return records, Corpus(docs), EmbeddingStore(HashingEncoder())


# ---------------------------------------------------------------- actions


def test_parse_hermes_tool_call_and_answer():
    action = parse_action(call("bm25_search", query="Ed Wood", k=2))
    assert (action.kind, action.name, action.arguments) == ("tool_call", "bm25_search", {"query": "Ed Wood", "k": 2})
    assert parse_action("<answer> yes </answer>").answer == "yes"
    assert parse_action("I think it is yes.").kind == "no_action"


def test_malformed_calls_are_dropped_like_verl():
    assert parse_action("<tool_call>{not json}</tool_call>").kind == "malformed_tool_call"
    assert parse_action('<tool_call>{"name": "bm25_search"}</tool_call>').kind == "malformed_tool_call"
    mixed = parse_action("<tool_call>{bad}</tool_call>" + call("get_document", doc_id="d0000000001"))
    assert mixed.kind == "tool_call" and mixed.malformed_calls == 1
    two = parse_action(call("bm25_search", query="a") + call("dense_search", query="b"))
    assert two.name == "bm25_search" and two.extra_tool_calls == 1
    assert parse_action(call("bm25_search", query="x") + "<answer>no</answer>").kind == "tool_call"


def test_structured_api_tool_calls():
    turn = {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "bm25_search", "arguments": '{"query": "Paris"}'}}]}
    action = parse_action(turn)
    assert action.kind == "tool_call" and action.arguments == {"query": "Paris"}
    broken = {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "bm25_search", "arguments": "{oops"}}]}
    assert parse_action(broken).kind == "malformed_tool_call"


# ---------------------------------------------------------------- rewards


def test_hotpot_answer_metrics():
    assert exact_match("The Paris", ["paris"]) == 1.0
    assert f1_score("yes indeed", ["yes"]) == 0.0  # yes/no only score on exact match
    assert f1_score("New York City", ["New York"]) == pytest.approx(0.8)


def test_cost_is_charged_only_on_correct_answers():
    heavy = EpisodeStats(answer="Paris", assistant_turns=6, tool_calls=5, obs_tokens=3000)
    right = compute_reward(heavy, ["Paris"])
    assert right.cost == pytest.approx(-(0.05 * 4 + 0.02 * 3))
    wrong = compute_reward(EpisodeStats(answer="Lyon", assistant_turns=6, tool_calls=5, obs_tokens=3000), ["Paris"])
    assert wrong.cost == 0.0 and wrong.total == 0.0


def test_correct_answer_always_beats_wrong_answer():
    worst_correct = EpisodeStats(answer="Paris", assistant_turns=8, tool_calls=7, invalid_calls=9, obs_tokens=50_000)
    best_wrong = EpisodeStats(answer="Lyon", assistant_turns=1)
    assert compute_reward(worst_correct, ["Paris"]).total > compute_reward(best_wrong, ["Paris"]).total


def test_format_overlong_and_process_terms():
    config = RewardConfig(process_weight=0.1)
    stats = EpisodeStats(answer=None, assistant_turns=8, tool_calls=7, invalid_calls=5, retrieved_doc_ids=["a"])
    breakdown = compute_reward(stats, ["Paris"], config, support_doc_ids=["a", "b"])
    assert breakdown.format == pytest.approx(-(0.3 + 0.2))  # capped invalid calls + missing answer
    assert breakdown.overlong == pytest.approx(-0.2)
    assert breakdown.process == pytest.approx(0.05)
    assert RewardConfig.preset("outcome_only").cost_enabled is False
    v11 = RewardConfig.preset("v11")
    assert v11.process_weight == 0.1 and v11.free_calls == 2
    two_calls = EpisodeStats(answer="Paris", assistant_turns=3, tool_calls=2)
    assert compute_reward(two_calls, ["Paris"], v11).cost == 0.0


# ---------------------------------------------------------------- environment


def test_episode_runs_and_scores(env_parts):
    env = SearchEnv(*env_parts)
    obs = env.reset("q1", seed=0)
    assert obs["messages"][1]["content"].endswith("same nationality?") and len(obs["tools"]) == 3
    obs, reward, done, info = env.step(call("bm25_search", query="Scott Derrickson nationality", k=1))
    assert not done and reward == 0.0 and "[d0000000001]" in obs["messages"][-1]["content"]
    _, reward, done, info = env.step("<answer>yes</answer>")
    assert done and info["done_reason"] == "answer" and info["reward"]["em"] == 1.0
    assert reward == pytest.approx(1.0 - 0.02 * env.stats.obs_tokens / 1000)
    with pytest.raises(RuntimeError):
        env.step("<answer>no</answer>")


def test_reset_isolates_episodes(env_parts):
    first, second = SearchEnv(*env_parts), SearchEnv(*env_parts)
    first.reset("q1")
    second.reset("q2")
    first.step(call("bm25_search", query="Ed Wood"))
    assert second.stats.tool_calls == 0 and first.stats.tool_calls == 1
    _, _, _, info = second.step(call("get_document", doc_id="d0000000001"))  # q1's paragraph
    assert info["action"] == "tool_call" and second.stats.invalid_calls == 1
    first.reset("q1")
    assert first.stats.tool_calls == 0 and len(first.messages) == 2 and first.turns == []


def test_invalid_actions_are_handled(env_parts):
    env = SearchEnv(*env_parts)
    env.reset("q1")
    obs, _, done, _ = env.step(call("web_search", query="x"))
    assert not done and obs["messages"][-1]["content"].startswith("Error: unknown tool")
    obs, _, done, _ = env.step(call("bm25_search", query="x", k=99))
    assert not done and "invalid arguments" in obs["messages"][-1]["content"]
    _, reward, done, info = env.step("<tool_call>{broken</tool_call>")
    assert done and info["done_reason"] == "malformed_tool_call" and env.stats.invalid_calls == 3
    assert reward == pytest.approx(-(0.3 + 0.2))


def test_max_turns_truncates(env_parts):
    env = SearchEnv(*env_parts, reward_config=RewardConfig(soft_turns=1, max_turns=2))
    env.reset("q2")
    env.step(call("bm25_search", query="capital"))
    _, _, done, info = env.step(call("bm25_search", query="France"))
    assert done and info["done_reason"] == "max_turns" and env.stats.answer is None


def test_closed_book_has_no_tools(env_parts):
    env = SearchEnv(*env_parts, closed_book=True)
    assert env.reset("q2")["tools"] == []
    _, _, _, info = env.step(call("bm25_search", query="x"))
    assert env.stats.invalid_calls == 1
    _, _, done, info = env.step("<answer>Paris</answer>")
    assert done and info["reward"]["em"] == 1.0 and info["reward"]["format"] == pytest.approx(-0.1)


def test_transcript_stats_match_environment(env_parts):
    """verl scores the decoded rollout text; it must agree with the environment."""
    env = SearchEnv(*env_parts)
    env.reset("q1")
    turns = [call("bm25_search", query="Ed Wood", k=2), call("nope", query="x"), "<answer>yes</answer>"]
    transcript = []
    for turn in turns:
        obs, reward, done, _ = env.step(turn)
        transcript.append(turn)
        if not done:
            transcript.append(f"user\n<tool_response>\n{obs['messages'][-1]['content']}\n</tool_response>")
    stats = stats_from_transcript("\n".join(transcript))
    for field in ("answer", "tool_calls", "invalid_calls", "malformed_calls", "obs_tokens", "assistant_turns"):
        assert getattr(stats, field) == getattr(env.stats, field), field
    assert compute_reward(stats, ["yes"]).total == pytest.approx(reward)


# ---------------------------------------------------------------- data


def _item(qid, qtype="bridge", question=None):
    return QAItem(qid, "hotpotqa", question or f"question {qid}?", ["a"], qtype, "hard", ["T"],
                  [Paragraph(doc_id("hotpotqa", qid, i), f"T{i}", "text") for i in range(3)])


def test_stratified_sampling_is_deterministic_and_proportional():
    items = [_item(str(i), "comparison" if i % 4 == 0 else "bridge") for i in range(400)]
    first, again = stratified_sample(items, 100, seed=1), stratified_sample(items, 100, seed=1)
    assert [i.qid for i in first] == [i.qid for i in again]
    assert sum(i.qtype == "comparison" for i in first) == 25


def test_pools_contain_own_paragraphs_and_opaque_ids():
    items = [_item(f"q{i}") for i in range(50)]  # "q" never occurs in a hex digest
    attach_pools(items, seed=3, pool_size=20)
    for item in items:
        assert len(item.pool) == len(set(item.pool)) == 20
        assert {p.doc_id for p in item.paragraphs} <= set(item.pool)
        assert all(item.qid not in doc for doc in item.pool)


def test_leakage_report_flags_duplicates():
    train = [_item("t1", question="Who wrote Hamlet?")]
    report = leakage_report({"train": train, "test": [_item("x1", question="who wrote hamlet")]})
    assert report["test"]["exact_question_in_train"] == 1


def test_verl_reward_hook_accepts_parquet_arrays():
    import numpy as np

    from agentos.rl.verl_reward import compute_score

    rollout = (
        call("bm25_search", query="Ed Wood")
        + "user\n<tool_response>\n1. [d0000000002] Ed Wood | American filmmaker\n</tool_response>"
        + "<answer>yes</answer>"
    )
    result = compute_score(
        "hotpotqa", rollout, np.array(["yes"], dtype=object),
        {"support_doc_ids": np.array(["d0000000002", "d0000000001"], dtype=object)}, preset="with_process",
    )
    assert result["r_em"] == 1.0 and result["tool_calls"] == 1.0 and result["calls_bm25_search"] == 1.0
    assert result["r_process"] == pytest.approx(0.05) and result["calls_get_document"] == 0.0
    assert compute_score("hotpotqa", "no tags here", np.array(["yes"], dtype=object), None)["no_answer"] == 1.0


def test_prompt_tool_schemas_match_verl_rendering():
    from agentos.rl.tools import prompt_tool_schemas

    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in prompt_tool_schemas()}
    assert schemas["get_document"] == {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}
    k = schemas["bm25_search"]["properties"]["k"]
    assert set(k) == {"type", "description"} and "additionalProperties" not in schemas["bm25_search"]
