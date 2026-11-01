from pathlib import Path

from agentos.benchmarks.real_data import convert_bfcl_rows, convert_hotpot_rows, read_tasks
from agentos.benchmarks.real_runner import evaluate_real_task


def test_convert_bfcl_preserves_provenance_and_ground_truth():
    prompts = [
        {
            "id": "b1",
            "question": [[{"role": "user", "content": "weather in Paris"}]],
            "function": [
                {
                    "name": "weather",
                    "description": "get weather for a city",
                    "parameters": {},
                }
            ],
        }
    ]
    answers = [{"id": "b1", "ground_truth": [{"weather": {"city": ["Paris"]}}]}]
    task = convert_bfcl_rows(
        prompts,
        answers,
        count=1,
        dev_count=1,
        source_url="https://example.test/bfcl",
        raw_sha256="abc",
    )[0]
    assert task.synthetic is False
    assert task.source.source_id == "b1"
    assert task.ground_truth["function_names"] == ["weather"]
    assert evaluate_real_task(task).value == 1


def test_convert_hotpot_and_retrieval_recall():
    rows = [
        {
            "_id": "h1",
            "question": "Where was Ada born?",
            "answer": "London",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": [["Ada", 0], ["London", 0]],
            "context": [
                ["Ada", ["Ada was born in London."]],
                ["London", ["London is a city."]],
                ["Noise", ["Unrelated material."]],
            ],
        }
    ]
    task = convert_hotpot_rows(
        rows,
        count=1,
        dev_count=0,
        source_url="https://example.test/hotpot",
        raw_sha256="def",
    )[0]
    record = evaluate_real_task(task)
    assert task.split == "test" and task.synthetic is False
    assert record.metric == "supporting_title_recall@2"
    assert record.value == 1


def test_checked_in_real_dataset_integrity():
    tasks = read_tasks(Path("benchmarks/real_data/all.jsonl"))
    assert len(tasks) == 200
    assert len({task.task_id for task in tasks}) == 200
    assert sum(task.split == "dev" for task in tasks) == 150
    assert sum(task.split == "test" for task in tasks) == 50
    assert all(task.synthetic is False for task in tasks)
    assert {task.source.dataset for task in tasks} == {
        "bfcl_v4_live_multiple",
        "hotpotqa_distractor",
    }
