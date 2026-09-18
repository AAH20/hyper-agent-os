"""
Durable Execution Engine (Temporal.io Pattern).

Provides checkpointed, fault-tolerant execution for 24/7 background agent
workflows. Steps are journaled idempotently to SQLite; if a host or daemon crashes,
the workflow replays past steps from the log and resumes exactly where it stopped.
"""

from __future__ import annotations
import enum
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class WorkflowStatus(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


@dataclass
class WorkflowStep:
    """A discrete, idempotent activity within a durable workflow."""
    name: str
    action: Callable[..., Any]
    max_retries: int = 3
    retry_delay_seconds: float = 0.5


class DurableEngine:
    """
    SQLite-backed journal engine for workflow state persistence and replay.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return self._conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    workflow_name TEXT,
                    status TEXT,
                    state_json TEXT,
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS step_journal (
                    step_id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    step_name TEXT,
                    result_json TEXT,
                    status TEXT,
                    completed_at REAL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id)
                )
                """
            )
            conn.commit()

    def create_workflow(self, workflow_name: str, initial_state: Dict[str, Any]) -> str:
        wf_id = f"wf_{uuid.uuid4().hex[:8]}"
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO workflows (workflow_id, workflow_name, status, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (wf_id, workflow_name, WorkflowStatus.PENDING.value, json.dumps(initial_state), now, now),
            )
            conn.commit()
        return wf_id

    def get_step_result(self, workflow_id: str, step_name: str) -> Optional[Any]:
        """Check if step has already been successfully executed and recorded."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT result_json FROM step_journal WHERE workflow_id = ? AND step_name = ? AND status = 'COMPLETED'",
                (workflow_id, step_name),
            )
            row = cur.fetchone()
            if row:
                return json.loads(row["result_json"])
        return None

    def record_step_result(self, workflow_id: str, step_name: str, result: Any) -> None:
        """Atomically record step completion to journal."""
        step_id = f"step_{uuid.uuid4().hex[:8]}"
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO step_journal (step_id, workflow_id, step_name, result_json, status, completed_at)
                VALUES (?, ?, ?, ?, 'COMPLETED', ?)
                """,
                (step_id, workflow_id, step_name, json.dumps(result), now),
            )
            conn.commit()

    def update_workflow_status(
        self, workflow_id: str, status: WorkflowStatus, state: Optional[Dict[str, Any]] = None
    ) -> None:
        now = time.time()
        with self._get_connection() as conn:
            if state is not None:
                conn.execute(
                    "UPDATE workflows SET status = ?, state_json = ?, updated_at = ? WHERE workflow_id = ?",
                    (status.value, json.dumps(state), now, workflow_id),
                )
            else:
                conn.execute(
                    "UPDATE workflows SET status = ?, updated_at = ? WHERE workflow_id = ?",
                    (status.value, now, workflow_id),
                )
            conn.commit()

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,))
            row = cur.fetchone()
            if row:
                return {
                    "workflow_id": row["workflow_id"],
                    "workflow_name": row["workflow_name"],
                    "status": row["status"],
                    "state": json.loads(row["state_json"]),
                    "updated_at": row["updated_at"],
                }
        return None


class DurableWorkflow:
    """
    Orchestrates a sequence of resilient steps with replay guarantees.
    """

    def __init__(self, name: str, engine: DurableEngine):
        self.name = name
        self.engine = engine
        self.steps: List[WorkflowStep] = []

    def add_step(
        self,
        name: str,
        action: Callable[..., Any],
        max_retries: int = 3,
        retry_delay_seconds: float = 0.2,
    ) -> DurableWorkflow:
        self.steps.append(
            WorkflowStep(
                name=name,
                action=action,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        )
        return self

    def execute(
        self, initial_state: Dict[str, Any], workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute or resume workflow. If workflow_id is provided, replays cached steps.
        """
        wf_id = workflow_id or self.engine.create_workflow(self.name, initial_state)
        current_state = dict(initial_state)

        self.engine.update_workflow_status(wf_id, WorkflowStatus.RUNNING)

        for step in self.steps:
            # 1. Check if step was already completed in previous run (Replay)
            cached_result = self.engine.get_step_result(wf_id, step.name)
            if cached_result is not None:
                current_state[step.name] = cached_result
                continue

            # 2. Execute step with retry loop
            attempts = 0
            success = False
            last_error = None
            step_result = None

            while attempts < step.max_retries:
                attempts += 1
                try:
                    step_result = step.action(current_state)
                    self.engine.record_step_result(wf_id, step.name, step_result)
                    current_state[step.name] = step_result
                    self.engine.update_workflow_status(wf_id, WorkflowStatus.RUNNING, current_state)
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    time.sleep(step.retry_delay_seconds)

            if not success:
                self.engine.update_workflow_status(wf_id, WorkflowStatus.FAILED, current_state)
                raise RuntimeError(
                    f"Workflow '{self.name}' [{wf_id}] step '{step.name}' failed after {step.max_retries} attempts: {last_error}"
                )

        self.engine.update_workflow_status(wf_id, WorkflowStatus.COMPLETED, current_state)
        return {
            "workflow_id": wf_id,
            "status": WorkflowStatus.COMPLETED.value,
            "state": current_state,
        }
