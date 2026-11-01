import time

from fastapi.testclient import TestClient

from agentos.agents import AgentExecutor, Planner, Verifier
from agentos.api import create_app
from agentos.config import Settings
from agentos.llm.mock import MockProvider
from agentos.mcp.catalog import build_local_registry
from agentos.mcp.provider import LocalToolProvider
from agentos.runtime.checkpoint import CheckpointStore
from agentos.runtime.recovery import RecoveryManager
from agentos.runtime.service import AgentRuntimeService


def _runtime(tmp_path):
    plan = {
        "goal": "search checkpoint docs",
        "subtasks": [
            {
                "task_id": "search",
                "description": "search local documents for checkpoint",
                "agent_type": "researcher",
                "preferred_tools": ["search_documents"],
                "expected_output_schema": {"type": "array", "minItems": 1},
            }
        ],
    }
    decision = {
        "tool_name": "search_documents",
        "arguments": {"query": "checkpoint", "top_k": 1},
        "reason": "relevant",
    }
    llm = MockProvider([plan, decision], latency_ms=0)
    provider = LocalToolProvider(build_local_registry(":memory:"))
    recovery = RecoveryManager(base_delay=0, jitter=0)
    return AgentRuntimeService(
        Planner(llm),
        AgentExecutor(provider, llm, recovery),
        Verifier(),
        CheckpointStore(tmp_path / "api.db"),
    )


def test_health_and_goal_only_task_lifecycle(tmp_path):
    runtime = _runtime(tmp_path)
    settings = Settings(_env_file=None, provider="mock", database=tmp_path / "unused.db")
    app = create_app(runtime, settings)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["runtime"] == "agent-runtime-service"
        chat = client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hello"}]}
        )
        assert chat.status_code == 200
        assert chat.json()["usage"]["model_name"] == "mock-v1"
        response = client.post(
            "/api/tasks", json={"goal": "search checkpoint docs"}
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        record = None
        for _ in range(100):
            record = client.get(f"/api/tasks/{task_id}").json()
            if record["status"] in {"SUCCESS", "FAILED"}:
                break
            time.sleep(0.01)
        assert record and record["status"] == "SUCCESS"
        assert record["final_result"]["nodes"]["search"]["verification"]["passed"]
        trace = client.get(f"/api/tasks/{task_id}/trace").json()["events"]
        assert "PLAN_CREATED" in {event["event"] for event in trace}
        assert "NODE_VERIFIED" in {event["event"] for event in trace}
        operations = client.get(f"/api/tasks/{task_id}/operations").json()
        assert operations["operations"][0]["status"] == "SUCCESS"
        assert client.post(f"/api/tasks/{task_id}/resume").status_code == 409


def test_unknown_task_is_404(tmp_path):
    app = create_app(
        _runtime(tmp_path),
        Settings(_env_file=None, provider="mock", database=tmp_path / "unused.db"),
    )
    with TestClient(app) as client:
        assert client.get("/api/tasks/missing").status_code == 404
