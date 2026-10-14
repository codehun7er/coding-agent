import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .budget import Budget
from .schema import decode_arguments


class Mode(StrEnum):
    SAFE = "safe"
    AUTO = "auto"


class Phase(StrEnum):
    PLANNING = "planning"
    EXECUTING = "executing"


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_PLAN = "awaiting_plan"
    UNCERTAIN = "uncertain"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    EXHAUSTED = "exhausted"
    ROLLED_BACK = "rolled_back"


class CallStatus(StrEnum):
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    STARTED = "started"
    UNCERTAIN = "uncertain"
    DONE = "done"


class AttemptStatus(StrEnum):
    STARTED = "started"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class Run:
    id: str
    created_at: str
    task: str
    mode: Mode
    phase: Phase
    status: RunStatus
    workspace: Path
    policy_path: Path
    model: str
    policy_text: str
    limits: Budget
    used: Budget
    baseline: int | None
    active_task: str | None
    note: str

    @property
    def remaining(self) -> Budget:
        return self.limits.minus(self.used)


@dataclass(frozen=True)
class Call:
    """A tool call proposed by the model and persisted before anything acts on it."""

    seq: int
    run_id: str
    call_id: str
    batch: int
    phase: Phase
    task_id: str | None
    name: str
    arguments: str
    status: CallStatus
    approved: bool
    result: str | None
    changes: tuple[str, ...]

    @property
    def output(self) -> dict[str, Any]:
        return json.loads(self.result) if self.result else {}

    def parsed_arguments(self) -> Any:
        try:
            return decode_arguments(self.arguments)
        except ValueError:
            return None


@dataclass(frozen=True)
class Attempt:
    number: int
    status: AttemptStatus
    diff: str


@dataclass(frozen=True)
class FileChange:
    """Net effect of a run on one file: hashes before its first change and after its last one; None means absent."""

    path: str
    original: str | None
    final: str | None
