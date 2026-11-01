from __future__ import annotations

import re
import string
from typing import Any

from agentos.benchmarks.real_data import OpenSourceTask
from agentos.llm.base import ChatResult


def _json_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _json_schema(item) for key, item in value.items()}
    type_map = {
        "dict": "object",
        "float": "number",
        "list": "array",
        "tuple": "array",
        "bool": "boolean",
    }
    if isinstance(result.get("type"), str):
        result["type"] = type_map.get(result["type"], result["type"])
    return result


def openai_tools(task: OpenSourceTask) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tools, safe_to_original = [], {}
    for index, tool in enumerate(task.tools):
        original = str(tool.get("name", f"tool_{index}"))
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", original)[:64] or f"tool_{index}"
        while safe in safe_to_original:
            safe = f"{safe[:58]}_{index}"
        safe_to_original[safe] = original
        parameters = _json_schema(tool.get("parameters") or {})
        if not parameters:
            parameters = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": safe,
                    "description": str(tool.get("description", "Select this function when relevant.")),
                    "parameters": parameters,
                },
            }
        )
    return tools, safe_to_original


def bfcl_request(task: OpenSourceTask) -> tuple[list[dict[str, str]], dict[str, Any]]:
    tools, mapping = openai_tools(task)
    messages = [
        {
            "role": "system",
            "content": (
                "Select and call the function that best satisfies the user. "
                "Do not invent unavailable functions."
            ),
        },
        {"role": "user", "content": task.prompt},
    ]
    return messages, {"tools": tools, "tool_choice": "auto", "_name_mapping": mapping}


def hotpot_request(task: OpenSourceTask) -> tuple[list[dict[str, str]], dict[str, Any]]:
    passages = []
    for item in task.context:
        passages.append(f"[{item['title']}]\n" + " ".join(item["sentences"]))
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the supplied passages. "
                "Return only the short answer, without explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {task.prompt}\n\nPassages:\n" + "\n\n".join(passages),
        },
    ]
    return messages, {}


def build_request(task: OpenSourceTask) -> tuple[list[dict[str, str]], dict[str, Any]]:
    return bfcl_request(task) if task.task_type == "tool_calling" else hotpot_request(task)


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _answer_f1(prediction: str, expected: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    expected_tokens = _normalize_answer(expected).split()
    if not prediction_tokens or not expected_tokens:
        return float(prediction_tokens == expected_tokens)
    common = 0
    remaining = expected_tokens.copy()
    for token in prediction_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_api_result(
    task: OpenSourceTask,
    result: ChatResult,
    name_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    message = result.raw.get("choices", [{}])[0].get("message", {})
    if task.task_type == "tool_calling":
        mapping = name_mapping or {}
        tool_calls = message.get("tool_calls") or []
        predicted = [
            mapping.get(str(call.get("function", {}).get("name", "")), str(call.get("function", {}).get("name", "")))
            for call in tool_calls
        ]
        expected = [str(name) for name in task.ground_truth["function_names"]]
        score = float(set(predicted) == set(expected) and len(predicted) == len(expected))
        metric = "function_selection_exact_match"
        output: Any = tool_calls if tool_calls else result.content
    else:
        predicted = result.content.strip()
        expected = str(task.ground_truth["answer"])
        exact_match = float(_normalize_answer(predicted) == _normalize_answer(expected))
        score = _answer_f1(predicted, expected)
        metric = "answer_f1"
        output = result.content
    return {
        "task_id": task.task_id,
        "source": task.source.dataset,
        "model": result.usage.model_name,
        "prompt": task.prompt,
        "model_output": output,
        "prediction": predicted,
        "expected": expected,
        "metric": metric,
        "score": score,
        **({"exact_match": exact_match} if task.task_type == "multi_hop_qa" else {}),
        "usage": result.usage.model_dump(),
    }
