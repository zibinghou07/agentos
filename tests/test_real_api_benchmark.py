from agentos.benchmarks.real_api import evaluate_api_result, openai_tools
from agentos.benchmarks.real_data import OpenSourceTask, SourceInfo
from agentos.llm.base import ChatResult, Usage


def _source(dataset: str = "bfcl_v4_live_multiple") -> SourceInfo:
    return SourceInfo(
        dataset=dataset,
        source_id="source-1",
        source_partition="test",
        source_url="https://example.test/data",
        license="Apache-2.0",
        raw_sha256="abc",
    )


def _usage() -> Usage:
    return Usage(
        request_id="request-1",
        model_name="qwen-test",
        input_tokens=10,
        output_tokens=2,
        latency_ms=12,
    )


def test_bfcl_tool_names_are_api_safe_and_scored_against_original():
    task = OpenSourceTask(
        task_id="real-bfcl-1",
        split="test",
        task_type="tool_calling",
        prompt="find a movie",
        tools=[
            {
                "name": "Movies.Find",
                "description": "find movies",
                "parameters": {"type": "dict", "properties": {}},
            }
        ],
        ground_truth={"calls": [], "function_names": ["Movies.Find"]},
        source=_source(),
    )
    tools, mapping = openai_tools(task)
    assert tools[0]["function"]["name"] == "Movies_Find"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    result = ChatResult(
        content="",
        usage=_usage(),
        raw={
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "Movies_Find", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
    )
    report = evaluate_api_result(task, result, mapping)
    assert report["prediction"] == ["Movies.Find"]
    assert report["score"] == 1


def test_hotpot_answer_em_and_f1():
    task = OpenSourceTask(
        task_id="real-hotpot-1",
        split="test",
        task_type="multi_hop_qa",
        prompt="Where?",
        context=[{"title": "Ada", "sentences": ["Ada was born in London."]}],
        ground_truth={"answer": "London", "supporting_facts": []},
        source=_source("hotpotqa_distractor"),
    )
    result = ChatResult(content="The London", usage=_usage())
    report = evaluate_api_result(task, result)
    assert report["exact_match"] == 1
    assert report["score"] == 1
