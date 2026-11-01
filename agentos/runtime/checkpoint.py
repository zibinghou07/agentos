from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from agentos.models import ExecutionEvent, TaskPlan, TaskRecord, TaskStatus, utc_now


class CheckpointStore:
    """SQLite task state, event stream, runner leases, and operation ledger.

    Confirmed successes are not replayed. SQLite and external systems cannot be
    committed atomically, so an interrupted non-idempotent RUNNING operation is
    deliberately treated as ambiguous rather than exactly-once.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                  checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  task_id TEXT NOT NULL,
                  thread_id TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  current_node TEXT,
                  task_status TEXT NOT NULL,
                  state_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoint_task
                  ON checkpoints(task_id, checkpoint_id DESC);
                CREATE TABLE IF NOT EXISTS tasks (
                  task_id TEXT PRIMARY KEY,
                  goal TEXT NOT NULL,
                  status TEXT NOT NULL,
                  plan_version INTEGER NOT NULL,
                  cancel_requested INTEGER NOT NULL DEFAULT 0,
                  final_result_json TEXT,
                  error_json TEXT,
                  runner_id TEXT,
                  lease_expires_at REAL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  task_id TEXT NOT NULL,
                  node_id TEXT,
                  event TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_task
                  ON events(task_id, event_id);
                CREATE TABLE IF NOT EXISTS operations (
                  operation_id TEXT PRIMARY KEY,
                  task_id TEXT NOT NULL,
                  node_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  result_json TEXT,
                  error_json TEXT,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_operations_task
                  ON operations(task_id, node_id);
                CREATE TABLE IF NOT EXISTS agent_stats (
                  agent_name TEXT PRIMARY KEY,
                  successes INTEGER NOT NULL DEFAULT 0,
                  failures INTEGER NOT NULL DEFAULT 0,
                  total_cost REAL NOT NULL DEFAULT 0,
                  total_latency_ms REAL NOT NULL DEFAULT 0,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_stats (
                  tool_name TEXT PRIMARY KEY,
                  calls INTEGER NOT NULL DEFAULT 0,
                  successes INTEGER NOT NULL DEFAULT 0,
                  total_latency_ms REAL NOT NULL DEFAULT 0,
                  updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(operations)")
            }
            if "error_json" not in columns:
                conn.execute("ALTER TABLE operations ADD COLUMN error_json TEXT")

    def save(
        self,
        plan: TaskPlan,
        thread_id: str,
        current_node: str | None,
        status: str,
    ) -> int:
        payload = plan.model_dump_json()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO checkpoints(task_id,thread_id,version,current_node,"
                "task_status,state_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (plan.task_id, thread_id, plan.version, current_node, status, payload, utc_now()),
            )
            return int(cursor.lastrowid)

    async def asave(
        self,
        plan: TaskPlan,
        thread_id: str,
        current_node: str | None,
        status: str,
    ) -> int:
        return await asyncio.to_thread(self.save, plan, thread_id, current_node, status)

    def latest(self, task_id: str) -> TaskPlan | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM checkpoints WHERE task_id=? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        return TaskPlan.model_validate_json(row["state_json"]) if row else None

    async def alatest(self, task_id: str) -> TaskPlan | None:
        return await asyncio.to_thread(self.latest, task_id)

    def upsert_task(
        self,
        plan: TaskPlan,
        status: TaskStatus,
        *,
        final_result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(task_id,goal,status,plan_version,final_result_json,
                  error_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET goal=excluded.goal,
                  status=excluded.status,plan_version=excluded.plan_version,
                  final_result_json=COALESCE(excluded.final_result_json,tasks.final_result_json),
                  error_json=excluded.error_json,updated_at=excluded.updated_at
                """,
                (
                    plan.task_id,
                    plan.goal,
                    status,
                    plan.version,
                    json.dumps(final_result, ensure_ascii=False) if final_result is not None else None,
                    json.dumps(error, ensure_ascii=False) if error is not None else None,
                    now,
                    now,
                ),
            )

    async def aupsert_task(
        self,
        plan: TaskPlan,
        status: TaskStatus,
        *,
        final_result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.upsert_task, plan, status, final_result=final_result, error=error
        )

    def task_record(self, task_id: str) -> TaskRecord | None:
        plan = self.latest(task_id)
        if plan is None:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            return None
        return TaskRecord(
            task_id=row["task_id"],
            goal=row["goal"],
            status=TaskStatus(row["status"]),
            plan_version=row["plan_version"],
            cancel_requested=bool(row["cancel_requested"]),
            final_result=json.loads(row["final_result_json"]) if row["final_result_json"] else None,
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            plan=plan,
        )

    async def atask_record(self, task_id: str) -> TaskRecord | None:
        return await asyncio.to_thread(self.task_record, task_id)

    def append_event(self, event: ExecutionEvent) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events(task_id,node_id,event,payload_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    event.task_id,
                    event.node_id,
                    event.event,
                    json.dumps(event.payload, ensure_ascii=False),
                    event.timestamp,
                ),
            )
            return int(cursor.lastrowid)

    async def aappend_event(self, event: ExecutionEvent) -> int:
        return await asyncio.to_thread(self.append_event, event)

    def events(self, task_id: str) -> list[ExecutionEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE task_id=? ORDER BY event_id", (task_id,)
            ).fetchall()
        return [
            ExecutionEvent(
                event_id=row["event_id"],
                event=row["event"],
                task_id=row["task_id"],
                node_id=row["node_id"],
                timestamp=row["created_at"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    async def aevents(self, task_id: str) -> list[ExecutionEvent]:
        return await asyncio.to_thread(self.events, task_id)

    def record_operation(
        self,
        operation_id: str,
        task_id: str,
        node_id: str,
        status: str,
        result: Any = None,
        error: Any = None,
    ) -> None:
        encoded = json.dumps(result, ensure_ascii=False) if result is not None else None
        encoded_error = json.dumps(error, ensure_ascii=False) if error is not None else None
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO operations(operation_id,task_id,node_id,status,result_json,
                  error_json,updated_at) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(operation_id) DO UPDATE SET status=excluded.status,
                  result_json=excluded.result_json,error_json=excluded.error_json,
                  updated_at=excluded.updated_at
                """,
                (operation_id, task_id, node_id, status, encoded, encoded_error, utc_now()),
            )

    async def arecord_operation(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.to_thread(self.record_operation, *args, **kwargs)

    def operation(self, operation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "operation_id": row["operation_id"],
            "task_id": row["task_id"],
            "node_id": row["node_id"],
            "status": row["status"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "updated_at": row["updated_at"],
        }

    async def aoperation(self, operation_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.operation, operation_id)

    def completed_operation(self, operation_id: str) -> Any | None:
        operation = self.operation(operation_id)
        return operation["result"] if operation and operation["status"] == "SUCCESS" else None

    def operations(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            ids = [
                row["operation_id"]
                for row in conn.execute(
                    "SELECT operation_id FROM operations WHERE task_id=? ORDER BY updated_at",
                    (task_id,),
                )
            ]
        return [operation for operation_id in ids if (operation := self.operation(operation_id))]

    async def aoperations(self, task_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.operations, task_id)

    def request_cancel(self, task_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE tasks SET cancel_requested=1,updated_at=? WHERE task_id=?",
                (utc_now(), task_id),
            )
            return cursor.rowcount > 0

    async def arequest_cancel(self, task_id: str) -> bool:
        return await asyncio.to_thread(self.request_cancel, task_id)

    def cancel_requested(self, task_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    async def acancel_requested(self, task_id: str) -> bool:
        return await asyncio.to_thread(self.cancel_requested, task_id)

    def acquire_lease(self, task_id: str, runner_id: str, ttl_seconds: float = 30) -> bool:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT runner_id,lease_expires_at FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                return False
            current, expires = row["runner_id"], row["lease_expires_at"] or 0
            if current and current != runner_id and expires > now:
                return False
            conn.execute(
                "UPDATE tasks SET runner_id=?,lease_expires_at=?,updated_at=? WHERE task_id=?",
                (runner_id, now + ttl_seconds, utc_now(), task_id),
            )
            return True

    async def aacquire_lease(
        self, task_id: str, runner_id: str, ttl_seconds: float = 30
    ) -> bool:
        return await asyncio.to_thread(self.acquire_lease, task_id, runner_id, ttl_seconds)

    def release_lease(self, task_id: str, runner_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET runner_id=NULL,lease_expires_at=NULL,updated_at=? "
                "WHERE task_id=? AND runner_id=?",
                (utc_now(), task_id, runner_id),
            )

    async def arelease_lease(self, task_id: str, runner_id: str) -> None:
        await asyncio.to_thread(self.release_lease, task_id, runner_id)

    def recoverable_task_ids(self) -> list[str]:
        terminal = (TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED)
        placeholders = ",".join("?" for _ in terminal)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT task_id FROM tasks WHERE status NOT IN ({placeholders}) "
                "AND cancel_requested=0 ORDER BY created_at",
                tuple(terminal),
            ).fetchall()
        return [row["task_id"] for row in rows]

    async def arecoverable_task_ids(self) -> list[str]:
        return await asyncio.to_thread(self.recoverable_task_ids)

    def task_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status,COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        counts["TOTAL"] = sum(counts.values())
        return counts

    async def atask_counts(self) -> dict[str, int]:
        return await asyncio.to_thread(self.task_counts)

    # Routing feedback: outcomes observed at runtime feed the next routing decision.

    def record_agent_outcome(
        self, agent_name: str, success: bool, cost: float, latency_ms: float
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_stats(agent_name,successes,failures,total_cost,
                  total_latency_ms,updated_at) VALUES(?,?,?,?,?,?)
                ON CONFLICT(agent_name) DO UPDATE SET
                  successes=agent_stats.successes+excluded.successes,
                  failures=agent_stats.failures+excluded.failures,
                  total_cost=agent_stats.total_cost+excluded.total_cost,
                  total_latency_ms=agent_stats.total_latency_ms+excluded.total_latency_ms,
                  updated_at=excluded.updated_at
                """,
                (agent_name, int(success), int(not success), cost, latency_ms, utc_now()),
            )

    async def arecord_agent_outcome(
        self, agent_name: str, success: bool, cost: float, latency_ms: float
    ) -> None:
        await asyncio.to_thread(self.record_agent_outcome, agent_name, success, cost, latency_ms)

    def agent_stats(self) -> dict[str, dict[str, float]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM agent_stats").fetchall()
        return {
            row["agent_name"]: {
                "successes": row["successes"],
                "failures": row["failures"],
                "total_cost": row["total_cost"],
                "total_latency_ms": row["total_latency_ms"],
            }
            for row in rows
        }

    async def aagent_stats(self) -> dict[str, dict[str, float]]:
        return await asyncio.to_thread(self.agent_stats)

    def record_tool_outcome(self, tool_name: str, success: bool, latency_ms: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_stats(tool_name,calls,successes,total_latency_ms,updated_at)
                VALUES(?,1,?,?,?)
                ON CONFLICT(tool_name) DO UPDATE SET calls=tool_stats.calls+1,
                  successes=tool_stats.successes+excluded.successes,
                  total_latency_ms=tool_stats.total_latency_ms+excluded.total_latency_ms,
                  updated_at=excluded.updated_at
                """,
                (tool_name, int(success), latency_ms, utc_now()),
            )

    async def arecord_tool_outcome(self, tool_name: str, success: bool, latency_ms: float) -> None:
        await asyncio.to_thread(self.record_tool_outcome, tool_name, success, latency_ms)

    def tool_stats(self) -> dict[str, dict[str, float]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tool_stats").fetchall()
        return {
            row["tool_name"]: {
                "calls": row["calls"],
                "successes": row["successes"],
                "total_latency_ms": row["total_latency_ms"],
            }
            for row in rows
        }

    async def atool_stats(self) -> dict[str, dict[str, float]]:
        return await asyncio.to_thread(self.tool_stats)
