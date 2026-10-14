import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .budget import Budget
from .journal import Conflict
from .model import Model, ModelError, OpenAIModel, ScriptedModel, load_model
from .plan import Task, checklist
from .policy import ConfigError, Violation
from .records import Call, Mode, Run, RunStatus
from .runtime import InvalidRequest, Reporter, Resolution, Runtime
from .settings import Settings, load_settings, parse_settings
from .store import Store, UnknownRun

_FAILURE_STATUSES = {RunStatus.FAILED, RunStatus.EXHAUSTED}


class ConsoleReporter(Reporter):
    def assistant(self, text: str) -> None:
        print(f"model: {text}")

    def tool_call(self, call: Call) -> None:
        task = f"[{call.task_id}] " if call.task_id else ""
        print(f"{task}-> {call.name} {call.arguments}")

    def tool_result(self, call: Call, result: dict[str, Any]) -> None:
        print(f"   <- {call.name}: {_summary(result)}")

    def tasks(self, tasks: Sequence[Task]) -> None:
        print("Tasks:")
        print(checklist(tasks))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    store = Store(arguments.state / "runs.db")
    try:
        return arguments.handler(arguments, store)
    except (InvalidRequest, Conflict, Violation, ConfigError, UnknownRun, ModelError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        store.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeagent", description="A coding agent with a guarded, resumable runtime.")
    parser.add_argument("--state", type=Path, default=Path(".codeagent"), help="directory with the run database")
    commands = parser.add_subparsers(required=True, metavar="command")

    run = commands.add_parser("run", help="start a run")
    run.add_argument("task")
    run.add_argument("--workspace", type=Path, default=Path("workspace"))
    run.add_argument("--policy", type=Path, default=Path("policy.toml"))
    run.add_argument("--mode", choices=[mode.value for mode in Mode], default=Mode.SAFE.value)
    run.add_argument("--plan", action="store_true", help="start in plan mode")
    run.add_argument("--script", type=Path, help="replay model replies from a JSON file instead of calling a model")
    run.set_defaults(handler=_run)

    for name, handler, help_text in (
        ("resume", _resume, "continue a run after a crash or a resolved conflict"),
        ("approve", _approve, "allow the action the run is waiting on"),
        ("status", _status, "show a run's state, budget and pending decision"),
        ("tasks", _tasks, "show the task checklist"),
        ("diff", _diff, "show what the run changed in the workspace"),
        ("rollback", _rollback, "restore the files the run changed"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("run_id")
        command.set_defaults(handler=handler)

    deny = commands.add_parser("deny", help="refuse the action the run is waiting on")
    deny.add_argument("run_id")
    deny.add_argument("--reason", default="")
    deny.set_defaults(handler=_deny)

    plan = commands.add_parser("plan", help="show, approve or reject the proposed plan")
    plan.add_argument("decision", choices=["show", "approve", "reject"])
    plan.add_argument("run_id")
    plan.add_argument("--reason", default="")
    plan.set_defaults(handler=_plan)

    resolve = commands.add_parser("resolve", help="settle a call interrupted by a crash")
    resolve.add_argument("run_id")
    resolve.add_argument("resolution", choices=[resolution.value for resolution in Resolution])
    ownership = resolve.add_mutually_exclusive_group()
    ownership.add_argument("--adopt", dest="adopt", action="store_true", default=None)
    ownership.add_argument("--disown", dest="adopt", action="store_false")
    resolve.set_defaults(handler=_resolve)

    return parser


def _run(arguments: argparse.Namespace, store: Store) -> int:
    settings = load_settings(arguments.policy, arguments.workspace)
    if arguments.state.resolve().is_relative_to(settings.workspace):
        raise ConfigError("the state directory must live outside the workspace")
    model = ScriptedModel(path=arguments.script.resolve()) if arguments.script else OpenAIModel.from_env()
    runtime = _build(store, settings, model)
    run_id = runtime.start(arguments.task, mode=Mode(arguments.mode), plan_first=arguments.plan)
    print(f"run {run_id} started in {arguments.mode} mode" + (" with plan mode" if arguments.plan else ""))
    return _report(store, runtime, runtime.advance(run_id))


def _resume(arguments: argparse.Namespace, store: Store) -> int:
    runtime = _runtime(store, arguments.run_id)
    return _report(store, runtime, runtime.resume(arguments.run_id))


def _approve(arguments: argparse.Namespace, store: Store) -> int:
    runtime = _runtime(store, arguments.run_id)
    return _report(store, runtime, runtime.decide_action(arguments.run_id, approve=True))


def _deny(arguments: argparse.Namespace, store: Store) -> int:
    runtime = _runtime(store, arguments.run_id)
    return _report(store, runtime, runtime.decide_action(arguments.run_id, approve=False, reason=arguments.reason))


def _plan(arguments: argparse.Namespace, store: Store) -> int:
    if arguments.decision == "show":
        proposal = store.latest_plan(arguments.run_id)
        if proposal is None:
            raise InvalidRequest(f"run {arguments.run_id} has no plan")
        print(f"plan v{proposal.version} ({proposal.status})")
        print(proposal.plan.render())
        return 0
    runtime = _runtime(store, arguments.run_id)
    approve = arguments.decision == "approve"
    return _report(store, runtime, runtime.decide_plan(arguments.run_id, approve=approve, feedback=arguments.reason))


def _resolve(arguments: argparse.Namespace, store: Store) -> int:
    runtime = _runtime(store, arguments.run_id)
    run = runtime.resolve(arguments.run_id, Resolution(arguments.resolution), adopt=arguments.adopt)
    return _report(store, runtime, run)


def _status(arguments: argparse.Namespace, store: Store) -> int:
    run = store.run(arguments.run_id)
    print(f"task: {run.task}")
    print(f"mode: {run.mode}, phase: {run.phase}, created: {run.created_at}")
    tasks = store.tasks(run.id)
    if tasks:
        ConsoleReporter().tasks(tasks)
    return _report(store, _runtime(store, run.id), run)


def _tasks(arguments: argparse.Namespace, store: Store) -> int:
    tasks = store.tasks(arguments.run_id)
    if not tasks:
        print("this run has no tasks: it was started without plan mode or the plan is not approved yet")
        return 0
    print(checklist(tasks))
    for task in tasks:
        if task.evidence:
            print(f"  {task.id}: {task.evidence}")
    return 0


def _diff(arguments: argparse.Namespace, store: Store) -> int:
    diff = _runtime(store, arguments.run_id).diff(arguments.run_id)
    print(diff or "the run has not changed any file", end="" if diff else "\n")
    return 0


def _rollback(arguments: argparse.Namespace, store: Store) -> int:
    restored = _runtime(store, arguments.run_id).rollback(arguments.run_id)
    print("restored: " + (", ".join(restored) or "nothing to restore"))
    return 0


def _runtime(store: Store, run_id: str) -> Runtime:
    run = store.run(run_id)
    return _build(store, parse_settings(run.policy_text, run.policy_path, run.workspace), load_model(run.model))


def _build(store: Store, settings: Settings, model: Model) -> Runtime:
    return Runtime(store, settings, model, reporter=ConsoleReporter(), after_execute=_crash_hook())


def _crash_hook() -> Callable[[Call], None] | None:
    """Simulate a kill between a tool's execution and the storing of its result when CODEAGENT_CRASH_AFTER names it."""
    target = os.environ.get("CODEAGENT_CRASH_AFTER")
    if not target:
        return None

    def crash(call: Call) -> None:
        if call.name == target:
            print(f"simulated crash: {call.name} ran but its result was not saved", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(70)

    return crash


def _report(store: Store, runtime: Runtime, run: Run) -> int:
    print(f"\nrun {run.id}: {run.status}")
    if run.note:
        print(run.note)
    match run.status:
        case RunStatus.AWAITING_APPROVAL:
            print(f"  codeagent approve {run.id}")
            print(f"  codeagent deny {run.id} --reason '...'")
        case RunStatus.AWAITING_PLAN:
            print(store.latest_plan(run.id).plan.render())
            print(f"  codeagent plan approve {run.id}")
            print(f"  codeagent plan reject {run.id} --reason '...'")
        case RunStatus.UNCERTAIN:
            _explain_interruption(runtime, run)
        case RunStatus.BLOCKED:
            print(f"  codeagent resume {run.id}")
    print("budget: " + _budget(run.used, run.limits))
    return 1 if run.status in _FAILURE_STATUSES else 0


def _explain_interruption(runtime: Runtime, run: Run) -> None:
    interruption = runtime.interruption(run.id)
    if interruption.drift.paths:
        print("files changed since the call started:")
        print(interruption.drift.text, end="")
        ownership = " --adopt|--disown"
    else:
        print("no file changed since the call started")
        ownership = ""
    options = "|".join(interruption.options)
    print(f"  codeagent resolve {run.id} {{{options}}}{ownership}")


def _budget(used: Budget, limits: Budget) -> str:
    return ", ".join(f"{name} {getattr(used, name)}/{getattr(limits, name)}" for name in Budget.counters())


def _summary(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"error: {result.get('error', 'unknown')}"
    parts = ["ok"]
    if "exit_code" in result:
        parts.append(f"exit code {result['exit_code']}")
    if "truncated" in result:
        parts.append("output truncated")
    return ", ".join(parts)
