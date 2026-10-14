import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from codeagent.model import ScriptedModel
from codeagent.plan import Task
from codeagent.records import Call, Mode, Run
from codeagent.runtime import Reporter, Runtime
from codeagent.settings import load_settings, parse_settings
from codeagent.store import Store

ROOT = Path(__file__).resolve().parents[1]

FIXED_ORDER = """def calculate_total(prices: list[float], discount_percent: float) -> float:
    if not 0 <= discount_percent <= 100:
        raise ValueError("discount_percent must be between 0 and 100")
    return sum(prices) * (100 - discount_percent) / 100
"""

TEST_COMMAND = "python3 -m unittest -v"

BUDGET = {"model_requests": 40, "tool_calls": 80, "side_effects": 20, "output_bytes": 200000}


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": arguments}


def reply(*calls: dict[str, Any], content: str = "") -> dict[str, Any]:
    return {"content": content, "tool_calls": list(calls)}


def final(text: str = "Done.") -> dict[str, Any]:
    return {"content": text}


def policy_text(
    *,
    allow_commands: Sequence[str] = (TEST_COMMAND, "ls"),
    deny_commands: Sequence[str] = ("rm", "bash", "python3 -c"),
    allow_dirs: Sequence[str] = (".",),
    deny_dirs: Sequence[str] = ("private",),
    budget: dict[str, int] | None = None,
    tool_output_bytes: int = 20000,
) -> str:
    def strings(values: Sequence[str]) -> str:
        return "[" + ", ".join(f'"{value}"' for value in values) + "]"

    limits = {**BUDGET, **(budget or {})}
    return "\n".join(
        [
            "[commands]",
            f"allow = {strings(allow_commands)}",
            f"deny = {strings(deny_commands)}",
            "[directories]",
            f"allow = {strings(allow_dirs)}",
            f"deny = {strings(deny_dirs)}",
            "[limits]",
            "bash_timeout_seconds = 2",
            f"tool_output_bytes = {tool_output_bytes}",
            "[budget]",
            *(f"{name} = {value}" for name, value in limits.items()),
            "[journal]",
            'ignore = ["__pycache__", "*.pyc"]',
        ]
    )


def make_project(root: Path, **policy: Any) -> tuple[Path, Path]:
    workspace = root / "workspace"
    shutil.copytree(ROOT / "workspace", workspace, ignore=shutil.ignore_patterns("__pycache__"))
    policy_path = root / "policy.toml"
    policy_path.write_text(policy_text(**policy), encoding="utf-8")
    return workspace, policy_path


class SimulatedCrash(BaseException):
    """Stops the runtime the way a kill would: no except Exception handler sees it."""


class RecordingReporter(Reporter):
    def __init__(self) -> None:
        self.executed: list[Call] = []
        self.results: list[tuple[str, dict[str, Any]]] = []
        self.checklists: list[list[Task]] = []

    def tool_call(self, call: Call) -> None:
        self.executed.append(call)

    def tool_result(self, call: Call, result: dict[str, Any]) -> None:
        self.results.append((call.name, result))

    def tasks(self, tasks: Sequence[Task]) -> None:
        self.checklists.append(list(tasks))


class RecordingModel(ScriptedModel):
    def __init__(self, steps: list[dict[str, Any]]) -> None:
        super().__init__(steps)
        self.offered: list[list[str]] = []

    def complete(self, messages, tools):
        self.offered.append([tool["function"]["name"] for tool in tools])
        return super().complete(messages, tools)


class Agent:
    """A runtime over a temporary project; `restart` drops every in-memory object, as a new process would."""

    def __init__(self, root: Path, steps: list[dict[str, Any]], **policy: Any) -> None:
        self.workspace, self.policy_path = make_project(root, **policy)
        self.database = root / "state" / "runs.db"
        self.steps = steps
        self.crash: Callable[[Call], None] | None = None
        self.store: Store | None = None
        self.restart()

    def restart(self) -> None:
        if self.store is not None:
            self.store.close()
        self.store = Store(self.database)
        if hasattr(self, "run_id"):
            run = self.store.run(self.run_id)
            self.settings = parse_settings(run.policy_text, run.policy_path, run.workspace)
        else:
            self.settings = load_settings(self.policy_path, self.workspace)
        self.reporter = RecordingReporter()
        self.model = RecordingModel(self.steps)
        self.runtime = Runtime(self.store, self.settings, self.model, reporter=self.reporter, after_execute=self.crash)

    def crash_after(self, tool_name: str) -> None:
        def crash(call: Call) -> None:
            if call.name == tool_name:
                raise SimulatedCrash(call.name)

        self.crash = crash
        self.restart()

    def heal(self) -> None:
        self.crash = None
        self.restart()

    def start(self, *, mode: Mode = Mode.AUTO, plan_first: bool = False, task: str = "Fix the order total") -> Run:
        self.run_id = self.runtime.start(task, mode=mode, plan_first=plan_first)
        return self.runtime.advance(self.run_id)

    @property
    def run(self) -> Run:
        return self.store.run(self.run_id)

    def file(self, name: str) -> str:
        return (self.workspace / name).read_text(encoding="utf-8")

    def tool_messages(self) -> list[dict[str, Any]]:
        return [message for message in self.store.messages(self.run_id) if message["role"] == "tool"]
