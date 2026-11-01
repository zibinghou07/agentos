from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from agentos.models import TaskNode


class VerificationResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    action: str = "none"


class Verifier:
    """Programmatic acceptance checks; model self-evaluation is never sufficient."""

    def verify(
        self,
        result: dict[str, Any],
        required_keys: list[str] | None = None,
        expected: Any = None,
        output_schema: dict[str, Any] | None = None,
    ) -> VerificationResult:
        failures: list[str] = []
        output = result.get("output")
        list_contract = (output_schema or {}).get("type") == "array"
        if output in (None, "", {}) or (output == [] and not list_contract):
            failures.append("empty_output")
        if required_keys:
            if not isinstance(output, dict):
                failures.append("output_not_object")
            else:
                for key in required_keys:
                    if key not in output:
                        failures.append(f"missing:{key}")
        if expected is not None and output != expected:
            failures.append("expected_value_mismatch")
        if output_schema:
            errors = sorted(
                Draft202012Validator(output_schema).iter_errors(output),
                key=lambda error: list(error.path),
            )
            failures.extend(f"output_schema:{error.message}" for error in errors)
        if not failures:
            return VerificationResult(passed=True)
        if any(item.startswith(("missing:", "output_schema:")) for item in failures):
            action = "repair"
        elif "empty_output" in failures:
            action = "retry"
        else:
            action = "replan"
        return VerificationResult(passed=False, failures=failures, action=action)

    def verify_node(self, node: TaskNode, result: dict[str, Any]) -> VerificationResult:
        tool_schema = result.get("tool_output_schema") or {}
        tool = result.get("tool")
        # The node contract was grounded on the planner's preferred tool. When the
        # executor legitimately chose an alternative (e.g. a conditional action),
        # verify against the contract of the tool that actually ran.
        if tool and node.preferred_tools and tool not in node.preferred_tools:
            return self.verify(result, output_schema=tool_schema)
        required = node.required_output_keys
        if tool_schema and required:
            # Only enforce keys that the tool's published contract can actually produce.
            required = [key for key in required if key in tool_schema.get("properties", {})]
        return self.verify(
            result,
            required_keys=required,
            output_schema=node.expected_output_schema or tool_schema,
        )
