from __future__ import annotations

from dataclasses import asdict, dataclass

from agentos.benchmarks.real_data import OpenSourceTask
from agentos.rag.retrieval import BM25Retriever


@dataclass(frozen=True)
class RealBaselineRecord:
    task_id: str
    dataset: str
    split: str
    metric: str
    value: float
    prediction: list[str]
    expected: list[str]


def _tool_text(tool: dict) -> str:
    parameters = tool.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    property_text = " ".join(
        f"{name} {spec.get('description', '') if isinstance(spec, dict) else ''}"
        for name, spec in properties.items()
    )
    return f"{tool.get('name', '')} {tool.get('description', '')} {property_text}"


def evaluate_real_task(task: OpenSourceTask) -> RealBaselineRecord:
    if task.task_type == "tool_calling":
        documents = {
            str(tool.get("name", index)): _tool_text(tool)
            for index, tool in enumerate(task.tools)
        }
        prediction = [result.document_id for result in BM25Retriever(documents).search(task.prompt, 1)]
        expected = [str(name) for name in task.ground_truth["function_names"]]
        value = float(bool(prediction) and prediction[0] in expected)
        metric = "tool_name_accuracy@1"
    else:
        documents = {
            str(item["title"]): " ".join(item["sentences"])
            for item in task.context
        }
        prediction = [result.document_id for result in BM25Retriever(documents).search(task.prompt, 2)]
        expected = sorted(
            {str(item["title"]) for item in task.ground_truth["supporting_facts"]}
        )
        value = len(set(prediction) & set(expected)) / max(len(set(expected)), 1)
        metric = "supporting_title_recall@2"
    return RealBaselineRecord(
        task_id=task.task_id,
        dataset=task.source.dataset,
        split=task.split,
        metric=metric,
        value=value,
        prediction=prediction,
        expected=expected,
    )


def evaluate_real_dataset(tasks: list[OpenSourceTask]) -> tuple[list[dict], dict]:
    records = [asdict(evaluate_real_task(task)) for task in tasks]
    groups: dict[str, list[float]] = {}
    for record in records:
        groups.setdefault(record["metric"], []).append(record["value"])
    summary = {
        "task_count": len(records),
        "split": sorted({record["split"] for record in records}),
        "metrics": {
            metric: {"mean": sum(values) / len(values), "count": len(values)}
            for metric, values in sorted(groups.items())
        },
        "note": (
            "Local retrieval/routing baseline only; this is not the official BFCL AST "
            "score or HotpotQA answer/joint score."
        ),
    }
    return records, summary
