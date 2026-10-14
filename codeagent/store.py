import json
import secrets
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .budget import Budget, BudgetExhausted
from .model import ToolCall
from .plan import Check, CheckKind, Plan, PlanStatus, PlanVersion, Step, Task, TaskStatus
from .records import (
    Attempt,
    AttemptStatus,
    Call,
    CallStatus,
    FileChange,
    Mode,
    Phase,
    Run,
    RunStatus,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    task TEXT NOT NULL,
    mode TEXT NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    workspace TEXT NOT NULL,
    policy_path TEXT NOT NULL,
    model TEXT NOT NULL,
    policy_text TEXT NOT NULL,
    limit_model_requests INTEGER NOT NULL,
    limit_tool_calls INTEGER NOT NULL,
    limit_side_effects INTEGER NOT NULL,
    limit_output_bytes INTEGER NOT NULL,
    used_model_requests INTEGER NOT NULL DEFAULT 0,
    used_tool_calls INTEGER NOT NULL DEFAULT 0,
    used_side_effects INTEGER NOT NULL DEFAULT 0,
    used_output_bytes INTEGER NOT NULL DEFAULT 0,
    baseline INTEGER,
    active_task TEXT,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS messages (
    run_id TEXT NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS calls (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    call_id TEXT NOT NULL,
    batch INTEGER NOT NULL,
    phase TEXT NOT NULL,
    task_id TEXT,
    name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    status TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    replayed INTEGER NOT NULL DEFAULT 0,
    changes TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS attempts (
    call_seq INTEGER NOT NULL REFERENCES calls(seq),
    number INTEGER NOT NULL,
    status TEXT NOT NULL,
    pre_snapshot INTEGER,
    diff TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    PRIMARY KEY (call_seq, number)
);
CREATE TABLE IF NOT EXISTS decisions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    run_id TEXT NOT NULL REFERENCES runs(id),
    version INTEGER NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL,
    feedback TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, version)
);
CREATE TABLE IF NOT EXISTS tasks (
    run_id TEXT NOT NULL REFERENCES runs(id),
    id TEXT NOT NULL,
    position INTEGER NOT NULL,
    description TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    check_kind TEXT NOT NULL,
    check_target TEXT NOT NULL,
    status TEXT NOT NULL,
    evidence TEXT,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS blobs (
    hash TEXT PRIMARY KEY,
    content BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    manifest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS file_changes (
    run_id TEXT NOT NULL REFERENCES runs(id),
    path TEXT NOT NULL,
    original_hash TEXT,
    final_hash TEXT,
    PRIMARY KEY (run_id, path)
);
"""

_UNCHANGED: Any = object()


class UnknownRun(LookupError):
    pass


class Store:
    """SQLite persistence for runs; every method is atomic, and `atomic()` makes a group of them one transaction."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA synchronous = FULL")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._depth = 0

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def atomic(self) -> Iterator[None]:
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self._db.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self._depth = 0
            self._db.execute("ROLLBACK")
            raise
        self._depth = 0
        self._db.execute("COMMIT")

    def create_run(
        self,
        *,
        task: str,
        mode: Mode,
        phase: Phase,
        workspace: Path,
        policy_path: Path,
        model: str,
        policy_text: str,
        limits: Budget,
    ) -> str:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        with self.atomic():
            self._db.execute(
                "INSERT INTO runs (id, created_at, task, mode, phase, status, workspace, policy_path, model,"
                " policy_text, limit_model_requests, limit_tool_calls, limit_side_effects, limit_output_bytes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    _now(),
                    task,
                    mode,
                    phase,
                    RunStatus.RUNNING,
                    str(workspace),
                    str(policy_path),
                    model,
                    policy_text,
                    *(getattr(limits, name) for name in Budget.counters()),
                ),
            )
        return run_id

    def run(self, run_id: str) -> Run:
        row = self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise UnknownRun(f"unknown run: {run_id}")
        return _run(row)

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus = _UNCHANGED,
        phase: Phase = _UNCHANGED,
        note: str = _UNCHANGED,
        active_task: str | None = _UNCHANGED,
        baseline: int = _UNCHANGED,
    ) -> None:
        changes = {
            column: value
            for column, value in (
                ("status", status),
                ("phase", phase),
                ("note", note),
                ("active_task", active_task),
                ("baseline", baseline),
            )
            if value is not _UNCHANGED
        }
        assignments = ", ".join(f"{column} = ?" for column in changes)
        with self.atomic():
            self._db.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*changes.values(), run_id))

    def charge(self, run_id: str, need: Budget) -> None:
        """Spend `need`, or spend nothing and raise BudgetExhausted if any limit cannot cover it."""
        with self.atomic():
            run = self.run(run_id)
            missing = run.limits.shortfall(run.used, need)
            if missing is not None:
                raise BudgetExhausted(missing)
            assignments = ", ".join(f"used_{name} = used_{name} + ?" for name in Budget.counters())
            self._db.execute(
                f"UPDATE runs SET {assignments} WHERE id = ?",
                (*(getattr(need, name) for name in Budget.counters()), run_id),
            )

    def append_message(self, run_id: str, body: Mapping[str, Any]) -> None:
        with self.atomic():
            self._db.execute(
                "INSERT INTO messages (run_id, seq, body)"
                " VALUES (?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE run_id = ?), ?)",
                (run_id, run_id, json.dumps(body, ensure_ascii=False)),
            )

    def messages(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT body FROM messages WHERE run_id = ? ORDER BY seq", (run_id,))
        return [json.loads(row["body"]) for row in rows]

    def record_reply(
        self,
        run_id: str,
        message: Mapping[str, Any],
        tool_calls: Sequence[ToolCall],
        *,
        phase: Phase,
        task_id: str | None,
    ) -> None:
        with self.atomic():
            self.append_message(run_id, message)
            batch = self._db.execute(
                "SELECT COALESCE(MAX(batch), 0) + 1 FROM calls WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            self._db.executemany(
                "INSERT INTO calls (run_id, call_id, batch, phase, task_id, name, arguments, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (run_id, call.id, batch, phase, task_id, call.name, call.arguments, CallStatus.PENDING)
                    for call in tool_calls
                ],
            )

    def call(self, seq: int) -> Call:
        return self._calls("seq = ?", (seq,))[0]

    def calls(self, run_id: str) -> list[Call]:
        return self._calls("run_id = ?", (run_id,))

    def open_calls(self, run_id: str) -> list[Call]:
        return self._calls("run_id = ? AND status != ?", (run_id, CallStatus.DONE))

    def calls_with_status(self, run_id: str, status: CallStatus) -> list[Call]:
        return self._calls("run_id = ? AND status = ?", (run_id, status))

    def prior_call(self, call: Call) -> Call | None:
        calls = self._calls("run_id = ? AND call_id = ? AND seq < ?", (call.run_id, call.call_id, call.seq))
        return calls[0] if calls else None

    def task_evidence(self, run_id: str, task_id: str) -> list[Call]:
        """Calls of a task that actually ran; answers replayed for a reused tool_call_id prove nothing new."""
        return self._calls(
            "run_id = ? AND task_id = ? AND status = ? AND replayed = 0", (run_id, task_id, CallStatus.DONE)
        )

    def set_call_status(self, call: Call, status: CallStatus, *, approved: bool | None = None) -> None:
        with self.atomic():
            self._db.execute("UPDATE calls SET status = ? WHERE seq = ?", (status, call.seq))
            if approved is not None:
                self._db.execute("UPDATE calls SET approved = ? WHERE seq = ?", (int(approved), call.seq))

    def finish_call(self, call: Call, result: str, *, replayed: bool = False) -> None:
        """Store the result and hand it to the model as the role=tool message answering this call."""
        with self.atomic():
            self._db.execute(
                "UPDATE calls SET status = ?, result = ?, replayed = ? WHERE seq = ?",
                (CallStatus.DONE, result, int(replayed), call.seq),
            )
            self.append_message(call.run_id, {"role": "tool", "tool_call_id": call.call_id, "content": result})

    def attribute_changes(self, call: Call, paths: Sequence[str]) -> None:
        with self.atomic():
            known = self.call(call.seq).changes
            self._db.execute(
                "UPDATE calls SET changes = ? WHERE seq = ?", (json.dumps(sorted({*known, *paths})), call.seq)
            )

    def start_attempt(self, call: Call, pre_snapshot: int | None) -> int:
        with self.atomic():
            number = self._db.execute(
                "SELECT COALESCE(MAX(number), 0) + 1 FROM attempts WHERE call_seq = ?", (call.seq,)
            ).fetchone()[0]
            self._db.execute(
                "INSERT INTO attempts (call_seq, number, status, pre_snapshot, started_at) VALUES (?, ?, ?, ?, ?)",
                (call.seq, number, AttemptStatus.STARTED, pre_snapshot, _now()),
            )
            self.set_call_status(call, CallStatus.STARTED)
            return number

    def close_attempt(self, call: Call, number: int, status: AttemptStatus, diff: str = "") -> None:
        with self.atomic():
            self._db.execute(
                "UPDATE attempts SET status = ?, diff = ?, finished_at = ? WHERE call_seq = ? AND number = ?",
                (status, diff, _now(), call.seq, number),
            )

    def last_attempt(self, call: Call) -> Attempt:
        return self.attempts(call)[-1]

    def attempts(self, call: Call) -> list[Attempt]:
        rows = self._db.execute(
            "SELECT number, status, diff FROM attempts WHERE call_seq = ? ORDER BY number", (call.seq,)
        )
        return [Attempt(row["number"], AttemptStatus(row["status"]), row["diff"]) for row in rows]

    def log_decision(self, run_id: str, kind: str, subject: str, verdict: str, reason: str = "") -> None:
        with self.atomic():
            self._db.execute(
                "INSERT INTO decisions (run_id, at, kind, subject, verdict, reason) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, _now(), kind, subject, verdict, reason),
            )

    def decisions(self, run_id: str) -> list[dict[str, str]]:
        rows = self._db.execute(
            "SELECT at, kind, subject, verdict, reason FROM decisions WHERE run_id = ? ORDER BY seq", (run_id,)
        )
        return [dict(row) for row in rows]

    def save_plan(self, run_id: str, plan: Plan) -> int:
        with self.atomic():
            version = self._db.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM plans WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            self._db.execute(
                "INSERT INTO plans (run_id, version, body, status) VALUES (?, ?, ?, ?)",
                (run_id, version, json.dumps(plan.to_dict(), ensure_ascii=False), PlanStatus.PROPOSED),
            )
            return version

    def latest_plan(self, run_id: str) -> PlanVersion | None:
        row = self._db.execute(
            "SELECT * FROM plans WHERE run_id = ? ORDER BY version DESC LIMIT 1", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return PlanVersion(
            row["version"], Plan.from_dict(json.loads(row["body"])), PlanStatus(row["status"]), row["feedback"]
        )

    def set_plan_status(self, run_id: str, version: int, status: PlanStatus, feedback: str = "") -> None:
        with self.atomic():
            self._db.execute(
                "UPDATE plans SET status = ?, feedback = ? WHERE run_id = ? AND version = ?",
                (status, feedback, run_id, version),
            )

    def create_tasks(self, run_id: str, steps: Sequence[Step]) -> None:
        with self.atomic():
            self._db.executemany(
                "INSERT INTO tasks (run_id, id, position, description, depends_on, check_kind, check_target, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        step.id,
                        position,
                        step.description,
                        json.dumps(list(step.depends_on)),
                        step.check.kind,
                        step.check.target,
                        TaskStatus.PENDING,
                    )
                    for position, step in enumerate(steps)
                ],
            )

    def tasks(self, run_id: str) -> list[Task]:
        rows = self._db.execute("SELECT * FROM tasks WHERE run_id = ? ORDER BY position", (run_id,))
        return [
            Task(
                id=row["id"],
                description=row["description"],
                depends_on=tuple(json.loads(row["depends_on"])),
                check=Check(CheckKind(row["check_kind"]), row["check_target"]),
                status=TaskStatus(row["status"]),
                evidence=row["evidence"],
            )
            for row in rows
        ]

    def set_task_status(self, run_id: str, task_id: str, status: TaskStatus, evidence: str | None = None) -> None:
        with self.atomic():
            self._db.execute(
                "UPDATE tasks SET status = ?, evidence = ? WHERE run_id = ? AND id = ?",
                (status, evidence, run_id, task_id),
            )

    def put_blob(self, digest: str, content: bytes) -> None:
        with self.atomic():
            self._db.execute("INSERT OR IGNORE INTO blobs (hash, content) VALUES (?, ?)", (digest, content))

    def blob(self, digest: str) -> bytes:
        return self._db.execute("SELECT content FROM blobs WHERE hash = ?", (digest,)).fetchone()["content"]

    def save_snapshot(self, run_id: str, manifest: Mapping[str, str]) -> int:
        with self.atomic():
            cursor = self._db.execute(
                "INSERT INTO snapshots (run_id, manifest) VALUES (?, ?)",
                (run_id, json.dumps(dict(sorted(manifest.items())))),
            )
            return cursor.lastrowid

    def snapshot(self, snapshot_id: int) -> dict[str, str]:
        row = self._db.execute("SELECT manifest FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        return json.loads(row["manifest"])

    def record_change(self, run_id: str, path: str, before: str | None, after: str | None) -> None:
        """Track a file's net change: the first recorded `before` is kept, `after` always moves forward."""
        with self.atomic():
            self._db.execute(
                "INSERT INTO file_changes (run_id, path, original_hash, final_hash) VALUES (?, ?, ?, ?)"
                " ON CONFLICT (run_id, path) DO UPDATE SET final_hash = excluded.final_hash",
                (run_id, path, before, after),
            )

    def file_changes(self, run_id: str) -> list[FileChange]:
        rows = self._db.execute(
            "SELECT path, original_hash, final_hash FROM file_changes WHERE run_id = ? ORDER BY path", (run_id,)
        )
        return [FileChange(row["path"], row["original_hash"], row["final_hash"]) for row in rows]

    def _calls(self, where: str, parameters: Sequence[Any]) -> list[Call]:
        rows = self._db.execute(f"SELECT * FROM calls WHERE {where} ORDER BY seq", parameters)
        return [
            Call(
                seq=row["seq"],
                run_id=row["run_id"],
                call_id=row["call_id"],
                batch=row["batch"],
                phase=Phase(row["phase"]),
                task_id=row["task_id"],
                name=row["name"],
                arguments=row["arguments"],
                status=CallStatus(row["status"]),
                approved=bool(row["approved"]),
                result=row["result"],
                changes=tuple(json.loads(row["changes"])),
            )
            for row in rows
        ]


def _run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        created_at=row["created_at"],
        task=row["task"],
        mode=Mode(row["mode"]),
        phase=Phase(row["phase"]),
        status=RunStatus(row["status"]),
        workspace=Path(row["workspace"]),
        policy_path=Path(row["policy_path"]),
        model=row["model"],
        policy_text=row["policy_text"],
        limits=Budget(**{name: row[f"limit_{name}"] for name in Budget.counters()}),
        used=Budget(**{name: row[f"used_{name}"] for name in Budget.counters()}),
        baseline=row["baseline"],
        active_task=row["active_task"],
        note=row["note"],
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
