import posixpath
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .policy import Violation, command_words
from .records import Call
from .schema import object_schema
from .tools import BASH, READ_FILE


class CheckKind(StrEnum):
    READ = "read"
    CHANGED = "changed"
    COMMAND = "command"


_STRING_LIST = {"type": "array", "items": {"type": "string", "minLength": 1}}

PLAN_PARAMETERS = object_schema(
    {
        "goal": {"type": "string", "minLength": 1},
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": object_schema(
                {
                    "id": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 1},
                    "tools": _STRING_LIST,
                    "depends_on": _STRING_LIST,
                    "check": object_schema(
                        {
                            "kind": {"type": "string", "enum": [kind.value for kind in CheckKind]},
                            "target": {"type": "string", "minLength": 1},
                        },
                        ("kind", "target"),
                    ),
                },
                ("id", "description", "tools", "depends_on", "check"),
            ),
        },
        "risks": _STRING_LIST,
        "assumptions": _STRING_LIST,
        "done_criteria": {"type": "string", "minLength": 1},
    },
    ("goal", "steps", "risks", "assumptions", "done_criteria"),
)


class PlanError(ValueError):
    pass


class Unverified(Exception):
    pass


class TaskStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    FAILED = "failed"


class PlanStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Check:
    """A fact the runtime can confirm from recorded tool calls, so a task is never done on the model's word alone."""

    kind: CheckKind
    target: str

    def describe(self) -> str:
        match self.kind:
            case CheckKind.READ:
                return f"{READ_FILE} of {self.target} succeeds"
            case CheckKind.CHANGED:
                return f"{self.target} is changed"
            case CheckKind.COMMAND:
                return f"`{self.target}` exits with code 0 after the last file change"

    def verify(self, calls: Sequence[Call]) -> Call:
        """Return the recorded call that proves this check, or raise Unverified with the reason it does not hold."""
        match self.kind:
            case CheckKind.READ:
                return _first(
                    calls, lambda call: _reads(call, self.target), f"no successful {READ_FILE} of {self.target}"
                )
            case CheckKind.CHANGED:
                target = posixpath.normpath(self.target)
                return _first(calls, lambda call: target in call.changes, f"no change to {self.target}")
            case CheckKind.COMMAND:
                return self._verify_command(calls)

    def _verify_command(self, calls: Sequence[Call]) -> Call:
        runs = [index for index, call in enumerate(calls) if _runs(call, self.target)]
        if not runs:
            raise Unverified(f"`{self.target}` was not run in this task")
        last = runs[-1]
        exit_code = calls[last].output.get("exit_code")
        if exit_code != 0:
            raise Unverified(f"the last run of `{self.target}` exited with code {exit_code}")
        if any(call.changes for call in calls[last + 1 :]):
            raise Unverified(f"files changed after the last run of `{self.target}`")
        return calls[last]


@dataclass(frozen=True)
class Step:
    id: str
    description: str
    tools: tuple[str, ...]
    depends_on: tuple[str, ...]
    check: Check


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: tuple[Step, ...]
    risks: tuple[str, ...]
    assumptions: tuple[str, ...]
    done_criteria: str

    @classmethod
    def parse(cls, data: Mapping[str, Any], *, tools: Collection[str]) -> "Plan":
        """Build a plan from arguments that already match PLAN_PARAMETERS, enforcing what a schema cannot."""
        plan = cls.from_dict(data)
        _validate(plan.steps, tools)
        return plan

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Plan":
        return cls(
            goal=data["goal"],
            steps=tuple(
                Step(
                    id=step["id"],
                    description=step["description"],
                    tools=tuple(step["tools"]),
                    depends_on=tuple(step["depends_on"]),
                    check=Check(CheckKind(step["check"]["kind"]), step["check"]["target"]),
                )
                for step in data["steps"]
            ),
            risks=tuple(data["risks"]),
            assumptions=tuple(data["assumptions"]),
            done_criteria=data["done_criteria"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [f"Goal: {self.goal}", "Steps:"]
        for step in self.steps:
            after = f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
            lines.append(f"  {step.id}. {step.description}{after}")
            lines.append(f"      tools: {', '.join(step.tools) or '-'}; done when {step.check.describe()}")
        lines.append("Risks: " + ("; ".join(self.risks) or "-"))
        lines.append("Assumptions: " + ("; ".join(self.assumptions) or "-"))
        lines.append(f"Done when: {self.done_criteria}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PlanVersion:
    version: int
    plan: Plan
    status: PlanStatus
    feedback: str


@dataclass(frozen=True)
class Task:
    id: str
    description: str
    depends_on: tuple[str, ...]
    check: Check
    status: TaskStatus
    evidence: str | None


def next_task(tasks: Sequence[Task]) -> Task | None:
    done = {task.id for task in tasks if task.status is TaskStatus.DONE}
    return next(
        (task for task in tasks if task.status is TaskStatus.PENDING and set(task.depends_on) <= done),
        None,
    )


def checklist(tasks: Sequence[Task]) -> str:
    suffixes = {TaskStatus.ACTIVE: "  <- in progress", TaskStatus.FAILED: "  <- failed"}
    return "\n".join(
        f"- {'[x]' if task.status is TaskStatus.DONE else '[ ]'} {task.id} {task.description}"
        + suffixes.get(task.status, "")
        for task in tasks
    )


def _validate(steps: Sequence[Step], tools: Collection[str]) -> None:
    ids = [step.id for step in steps]
    duplicates = sorted({step_id for step_id in ids if ids.count(step_id) > 1})
    if duplicates:
        raise PlanError(f"step ids must be unique: {', '.join(duplicates)}")
    for step in steps:
        unknown_tools = sorted(set(step.tools) - set(tools))
        if unknown_tools:
            raise PlanError(
                f"step {step.id} names unknown tools: {', '.join(unknown_tools)}; "
                f"valid tools are {', '.join(sorted(tools))}"
            )
        unknown_steps = sorted(set(step.depends_on) - set(ids))
        if unknown_steps:
            raise PlanError(f"step {step.id} depends on unknown steps: {', '.join(unknown_steps)}")
        if step.check.kind is CheckKind.COMMAND:
            try:
                command_words(step.check.target)
            except Violation as violation:
                raise PlanError(f"step {step.id} has an invalid check: {violation}") from None
    _reject_cycles(steps)


def _reject_cycles(steps: Sequence[Step]) -> None:
    remaining = {step.id: set(step.depends_on) for step in steps}
    while remaining:
        ready = [step_id for step_id, needs in remaining.items() if not needs & remaining.keys()]
        if not ready:
            raise PlanError(f"steps depend on each other in a cycle: {', '.join(sorted(remaining))}")
        for step_id in ready:
            del remaining[step_id]


def _first(calls: Sequence[Call], proves: Callable[[Call], bool], missing: str) -> Call:
    proof = next((call for call in calls if proves(call)), None)
    if proof is None:
        raise Unverified(f"{missing} is recorded for this task")
    return proof


def _reads(call: Call, path: str) -> bool:
    arguments = call.parsed_arguments()
    return (
        call.name == READ_FILE
        and call.output.get("ok") is True
        and isinstance(arguments, dict)
        and posixpath.normpath(str(arguments.get("path", ""))) == posixpath.normpath(path)
    )


def _runs(call: Call, command: str) -> bool:
    arguments = call.parsed_arguments()
    return (
        call.name == BASH
        and isinstance(arguments, dict)
        and str(arguments.get("command", "")).split() == command.split()
    )
