
import pytest
import httpx

from agentos.models import AgentProfile
from agentos.runtime.recovery import (
    ErrorType,
    RecoveryExhausted,
    RecoveryManager,
    classify_error,
)
from agentos.runtime.routing import AdaptiveRouter


def test_beta_binomial_cold_start_and_budget():
    router = AdaptiveRouter(epsilon=0)
    agents = [AgentProfile(name="fast", capabilities=["search documents"], average_cost=0.1, successes=0, failures=0), AgentProfile(name="costly", capabilities=["search documents"], average_cost=2.0, successes=100, failures=0)]
    assert router.success_rate(agents[0]) == 0.5
    assert router.route("search documents", agents, budget=0.5).selected_agent == "fast"


async def test_repair_changes_arguments():
    seen = []

    async def call(arguments):
        seen.append(arguments.copy())
        if "id" not in arguments:
            raise ValueError("invalid argument id")
        return arguments["id"]

    result, trace = await RecoveryManager(retries=0, repairs=1).execute(call, {"wrong": 1}, repair=lambda args, exc: {"id": 7})
    assert result == 7 and seen == [{"wrong": 1}, {"id": 7}]
    assert trace[0].startswith("REPAIR")


async def test_retry_budget_is_finite():
    calls = 0

    async def call(arguments):
        nonlocal calls
        calls += 1
        raise TimeoutError()

    with pytest.raises(RecoveryExhausted):
        await RecoveryManager(
            retries=2, repairs=0, replans=0, base_delay=0, jitter=0
        ).execute(call, {})
    assert calls == 3


def test_protocol_disconnect_is_server_error():
    assert classify_error(httpx.RemoteProtocolError("disconnect")) == ErrorType.SERVER_ERROR

