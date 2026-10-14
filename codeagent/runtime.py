import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from . import prompts
from .budget import Budget, BudgetExhausted
from .journal import Conflict, Diff, Journal
from .model import Model, Reply, ToolCall
from .plan import Task
from .planner import ControlTool, Planner
from .policy import Decision, Verdict, Violation
from .records import AttemptStatus, Call, CallStatus, Mode, Phase, Run, RunStatus
from .schema import decode_arguments, validate
from .settings import Settings
from .store import Store
from .tools import Outcome, Tool, Toolbox, ToolFailure


class Resolution(StrEnum):
    RETRY = "retry"
    FAIL = "fail"
    ABORT = "abort"


class InvalidRequest(Exception):
    """A user command that does not fit the run's current state."""


class Reporter:
    """Receives progress as it happens; the default ignores everything."""

    def assistant(self, text: str) -> None:
        pass

    def tool_call(self, call: Call) -> None:
        pass

    def tool_result(self, call: Call, result: dict[str, Any]) -> None:
        pass

    def tasks(self, tasks: Sequence[Task]) -> None:
        pass


@dataclass(frozen=True)
class Interruption:
    """A call a crash cut short: what it was, what changed since it started, and how the user may settle it."""

    call: Call
    drift: Diff
    options: tuple[Resolution, ...]


class _ArgumentError(Exception):
    pass


class Runtime:
    """Runs the agent loop for stored runs and applies the user's decisions to them.

    The model only proposes tool calls. Each call is recorded before anything acts on it, validated against its
    schema, judged by the policy for the run's mode and phase, charged to the budget and checked against the
    journal before it executes, and its result is stored in the same transaction that records its file changes.
    Whenever the run needs the user, it stops with a status saying why; `advance` resumes it from storage alone.
    `after_execute` is called between a tool's execution and the storing of its result, the window a crash test
    aims at.
    """

    def __init__(
        self,
        store: Store,
        settings: Settings,
        model: Model,
        *,
        reporter: Reporter | None = None,
        after_execute: Callable[[Call], None] | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._model = model
        self._reporter = reporter or Reporter()
        self._after_execute = after_execute or (lambda call: None)
        self._tools = Toolbox(settings)
        self._journal = Journal(store, settings)
        self._planner = Planner(store, self._tools.names())

    def start(self, task: str, *, mode: Mode, plan_first: bool) -> str:
        with self._store.atomic():
            run_id = self._store.create_run(
                task=task,
                mode=mode,
                phase=Phase.PLANNING if plan_first else Phase.EXECUTING,
                workspace=self._settings.workspace,
                policy_path=self._settings.policy_path,
                model=self._model.ref,
                policy_text=self._settings.policy_text,
                limits=self._settings.budget,
            )
            commands = self._settings.policy.allowed_commands
            self._store.append_message(run_id, {"role": "system", "content": prompts.system_prompt(commands)})
            self._store.append_message(run_id, {"role": "user", "content": task})
            if plan_first:
                self._store.append_message(run_id, {"role": "user", "content": prompts.planning()})
            self._journal.begin(run_id)
        return run_id

    def advance(self, run_id: str) -> Run:
        self._recover(run_id)
        while (run := self._store.run(run_id)).status is RunStatus.RUNNING:
            self._step(run)
        return run

    def resume(self, run_id: str) -> Run:
        """Continue a run after a crash, or after the user put back the files whose outside edit blocked it."""
        status = self._store.run(run_id).status
        if status is RunStatus.BLOCKED:
            self._reopen(run_id)
        elif status is not RunStatus.RUNNING:
            raise InvalidRequest(f"run {run_id} is {status}; only running and blocked runs resume")
        return self.advance(run_id)

    def decide_action(self, run_id: str, *, approve: bool, reason: str = "") -> Run:
        self._expect(run_id, RunStatus.AWAITING_APPROVAL)
        call = self._store.calls_with_status(run_id, CallStatus.AWAITING_APPROVAL)[0]
        with self._store.atomic():
            self._store.log_decision(run_id, "action", _describe(call), "approved" if approve else "denied", reason)
            if approve:
                self._store.set_call_status(call, CallStatus.PENDING, approved=True)
            else:
                self._finish(call, Outcome.failure(f"the user denied this action: {reason or 'no reason given'}"))
            self._reopen(run_id)
        return self.advance(run_id)

    def decide_plan(self, run_id: str, *, approve: bool, feedback: str = "") -> Run:
        self._expect(run_id, RunStatus.AWAITING_PLAN)
        with self._store.atomic():
            version = self._planner.decide(run_id, approve=approve, feedback=feedback)
            verdict = "approved" if approve else "rejected"
            self._store.log_decision(run_id, "plan", f"plan v{version}", verdict, feedback)
            self._reopen(run_id)
        return self.advance(run_id)

    def interruption(self, run_id: str) -> Interruption:
        self._expect(run_id, RunStatus.UNCERTAIN)
        call = self._store.calls_with_status(run_id, CallStatus.UNCERTAIN)[0]
        tool = self._tools.get(call.name)
        retryable = tool is not None and tool.retryable
        options = tuple(option for option in Resolution if option is not Resolution.RETRY or retryable)
        return Interruption(call, self._journal.drift(run_id), options)

    def resolve(self, run_id: str, resolution: Resolution, *, adopt: bool | None = None) -> Run:
        """Settle a call interrupted mid-execution: the user decides its fate and who owns the files it touched."""
        interruption = self.interruption(run_id)
        call, drift = interruption.call, interruption.drift
        if resolution not in interruption.options:
            raise InvalidRequest(f"{call.name} is not safe to repeat; choose fail or abort")
        if drift.paths and adopt is None:
            raise InvalidRequest(
                f"files changed since the interrupted call started: {', '.join(drift.paths)}; "
                "choose --adopt to attribute them to this run or --disown to treat them as external"
            )
        attempt = self._store.last_attempt(call)
        with self._store.atomic():
            settled = self._journal.settle(run_id, adopt=bool(adopt)) if drift.paths else Diff((), "")
            self._store.close_attempt(call, attempt.number, AttemptStatus.INTERRUPTED, settled.text)
            if adopt:
                self._store.attribute_changes(call, settled.paths)
            ownership = f"{'adopted' if adopt else 'disowned'}: {', '.join(settled.paths)}" if settled.paths else ""
            self._store.log_decision(run_id, "recovery", _describe(call), resolution, ownership)
            if resolution is Resolution.RETRY:
                self._store.set_call_status(call, CallStatus.PENDING)
            else:
                outcome = "the call was interrupted, its outcome is unknown and the user marked it failed"
                self._finish(call, Outcome.failure(outcome))
            if resolution is Resolution.ABORT:
                self._halt(self._store.run(run_id), RunStatus.FAILED, "aborted after an interrupted call")
            else:
                self._reopen(run_id)
        return self.advance(run_id)

    def diff(self, run_id: str) -> str:
        self._store.run(run_id)
        return self._journal.diff(run_id)

    def rollback(self, run_id: str) -> tuple[str, ...]:
        self._recover(run_id)
        run = self._store.run(run_id)
        if run.status is RunStatus.UNCERTAIN:
            raise InvalidRequest("resolve the interrupted call before rolling back")
        if run.status is RunStatus.ROLLED_BACK:
            raise InvalidRequest(f"run {run_id} is already rolled back")
        restored = self._journal.rollback(run_id)
        with self._store.atomic():
            self._store.log_decision(run_id, "rollback", ", ".join(restored) or "-", "restored")
            self._halt(run, RunStatus.ROLLED_BACK, f"rolled back, restored: {', '.join(restored) or '-'}")
        return restored

    def _step(self, run: Run) -> None:
        open_calls = self._store.open_calls(run.id)
        if open_calls:
            self._process(run, open_calls[0])
        elif self._planner.awaits_decision(run):
            self._store.update_run(run.id, status=RunStatus.AWAITING_PLAN, note="the plan waits for approval")
        elif (tasks := self._planner.assign(run)) is not None:
            self._reporter.tasks(tasks)
        else:
            self._consult_model(run)

    def _consult_model(self, run: Run) -> None:
        try:
            self._store.charge(run.id, Budget(model_requests=1))
        except BudgetExhausted as exhausted:
            self._exhaust(run, exhausted.counter)
            return
        reply = self._model.complete(self._store.messages(run.id), self._offered_tools(run)).sanitized()
        if reply.content:
            self._reporter.assistant(reply.content)
        if reply.tool_calls:
            self._admit(run, reply)
        else:
            self._conclude(run, reply)

    def _offered_tools(self, run: Run) -> list[dict[str, Any]]:
        effects = self._settings.policy.allowed_effects(planning=run.phase is Phase.PLANNING)
        return [*self._tools.specs(effects), *self._planner.specs(run)]

    def _admit(self, run: Run, reply: Reply) -> None:
        """Record a batch of calls and charge it whole, or answer every call unexecuted if the budget falls short."""
        side_effects = sum(1 for tool_call in reply.tool_calls if self._may_take_effect(run, tool_call))
        need = Budget(tool_calls=len(reply.tool_calls), side_effects=side_effects)
        with self._store.atomic():
            self._store.record_reply(
                run.id, reply.message(), reply.tool_calls, phase=run.phase, task_id=run.active_task
            )
            current = self._store.run(run.id)
            missing = current.limits.shortfall(current.used, need)
            if missing is None:
                self._store.charge(run.id, Budget(tool_calls=need.tool_calls))
            else:
                self._exhaust(current, missing)

    def _may_take_effect(self, run: Run, tool_call: ToolCall) -> bool:
        tool = self._tools.get(tool_call.name)
        if tool is None or not tool.effect.has_side_effect:
            return False
        try:
            arguments = _arguments(tool_call.arguments, tool.parameters)
        except _ArgumentError:
            return False
        return self._verdict(run, tool, arguments).decision is not Decision.DENY

    def _conclude(self, run: Run, reply: Reply) -> None:
        """Finish on a text reply, or push back once when plan or task work is still open and fail on a second try."""
        blocker = self._planner.blocker(run)
        nudge = {"role": "user", "content": blocker}
        with self._store.atomic():
            nudged = blocker is not None and self._store.messages(run.id)[-1] == nudge
            self._store.append_message(run.id, reply.message())
            if blocker is None:
                self._store.update_run(run.id, status=RunStatus.COMPLETED, note=reply.content)
            elif nudged:
                self._halt(run, RunStatus.FAILED, f"the model gave up: {blocker}")
            else:
                self._store.append_message(run.id, nudge)

    def _process(self, run: Run, call: Call) -> None:
        if call.status is CallStatus.UNCERTAIN:
            self._store.update_run(run.id, status=RunStatus.UNCERTAIN, note=_interrupted(call))
            return
        prior = self._store.prior_call(call)
        if prior is not None:
            self._replay(call, prior)
            return
        control = self._planner.get(call.name)
        if control is not None:
            self._control(run, call, control)
            return
        tool = self._tools.get(call.name)
        if tool is None:
            self._finish(call, Outcome.failure(f"unknown tool: {call.name}"))
            return
        refusal = self._planner.refusal_without_task(run)
        if refusal is not None:
            self._finish(call, Outcome.failure(refusal))
            return
        arguments = self._validated(call, tool.parameters)
        if arguments is None:
            return
        verdict = self._verdict(run, tool, arguments)
        if verdict.decision is Decision.DENY:
            self._finish(call, Outcome.failure(f"denied by policy: {verdict.reason}"))
        elif verdict.decision is Decision.CONFIRM and not call.approved:
            with self._store.atomic():
                self._store.set_call_status(call, CallStatus.AWAITING_APPROVAL)
                note = f"{_describe(call)} needs approval"
                self._store.update_run(run.id, status=RunStatus.AWAITING_APPROVAL, note=note)
        else:
            self._execute(run, call, tool, arguments)

    def _verdict(self, run: Run, tool: Tool, arguments: dict[str, Any]) -> Verdict:
        return self._settings.policy.check(tool.action(arguments), run.mode, planning=run.phase is Phase.PLANNING)

    def _replay(self, call: Call, prior: Call) -> None:
        """Answer a reused tool_call_id from storage: the same call is never executed twice."""
        if prior.name != call.name or _canonical(prior.arguments) != _canonical(call.arguments):
            reason = f"tool_call_id {call.call_id} was already used for {prior.name} with other arguments"
        elif prior.status is not CallStatus.DONE:
            reason = f"tool_call_id {call.call_id} is already in progress"
        else:
            self._store.finish_call(call, prior.result or "", replayed=True)
            self._reporter.tool_result(call, prior.output)
            return
        self._finish(call, Outcome.failure(f"protocol error: {reason}"))

    def _control(self, run: Run, call: Call, control: ControlTool) -> None:
        arguments = self._validated(call, control.parameters)
        if arguments is None:
            return
        self._reporter.tool_call(call)
        with self._store.atomic():
            outcome = control.execute(run, arguments)
            self._finish(call, outcome)
        tasks = self._store.tasks(run.id)
        if outcome.ok and tasks:
            self._reporter.tasks(tasks)

    def _execute(self, run: Run, call: Call, tool: Tool, arguments: dict[str, Any]) -> None:
        side_effect = tool.effect.has_side_effect
        limit = min(self._settings.limits.tool_output_bytes, run.remaining.output_bytes)
        if tool.metered and limit <= 0:
            self._exhaust(run, "output_bytes")
            return
        try:
            with self._store.atomic():
                self._store.charge(run.id, Budget(side_effects=int(side_effect)))
                pre = self._journal.guard(run.id) if side_effect else None
                attempt = self._store.start_attempt(call, pre)
        except BudgetExhausted as exhausted:
            self._exhaust(run, exhausted.counter)
            return
        except Conflict as conflict:
            note = f"{conflict}; put them back and resume, the pending call has not run"
            self._store.update_run(run.id, status=RunStatus.BLOCKED, note=note)
            return
        self._reporter.tool_call(call)
        policy_file = self._settings.read_policy_file()
        try:
            outcome = tool.execute(arguments, limit)
        except (ToolFailure, Violation, OSError) as error:
            outcome = Outcome.failure(str(error))
        self._after_execute(call)
        breach = self._settings.restore_policy_file(policy_file)
        outcome = outcome.clip(limit)
        with self._store.atomic():
            diff = self._journal.record(run.id, pre) if pre is not None else Diff((), "")
            self._store.close_attempt(call, attempt, AttemptStatus.FINISHED, diff.text)
            self._store.attribute_changes(call, diff.paths)
            self._store.charge(run.id, Budget(output_bytes=outcome.size))
            self._finish(call, outcome)
            breach = breach or self._settings.policy.audit(diff.paths)
            if breach is not None:
                self._halt(run, RunStatus.FAILED, breach)

    def _recover(self, run_id: str) -> None:
        """Requeue reads a crash cut short and mark interrupted side effects uncertain for the user to settle."""
        for call in self._store.calls_with_status(run_id, CallStatus.STARTED):
            tool = self._tools.get(call.name)
            attempt = self._store.last_attempt(call)
            with self._store.atomic():
                if tool is not None and not tool.effect.has_side_effect:
                    self._store.close_attempt(call, attempt.number, AttemptStatus.INTERRUPTED)
                    self._store.set_call_status(call, CallStatus.PENDING)
                else:
                    self._store.set_call_status(call, CallStatus.UNCERTAIN)
                    self._store.update_run(run_id, status=RunStatus.UNCERTAIN, note=_interrupted(call))

    def _exhaust(self, run: Run, counter: str) -> None:
        self._halt(run, RunStatus.EXHAUSTED, f"budget exhausted: {counter} (limit {getattr(run.limits, counter)})")

    def _halt(self, run: Run, status: RunStatus, note: str) -> None:
        """End the run: answer every open call so the conversation stays valid and fail the task left active."""
        with self._store.atomic():
            for call in self._store.open_calls(run.id):
                self._finish(call, Outcome.failure(f"not executed: {note}"))
            tasks = self._planner.abandon(self._store.run(run.id))
            self._store.update_run(run.id, status=status, note=note)
        if tasks is not None:
            self._reporter.tasks(tasks)

    def _validated(self, call: Call, schema: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return _arguments(call.arguments, schema)
        except _ArgumentError as error:
            self._finish(call, Outcome.failure(str(error)))
            return None

    def _finish(self, call: Call, outcome: Outcome) -> None:
        result = outcome.render()
        self._store.finish_call(call, result)
        self._reporter.tool_result(call, json.loads(result))

    def _reopen(self, run_id: str) -> None:
        self._store.update_run(run_id, status=RunStatus.RUNNING, note="")

    def _expect(self, run_id: str, status: RunStatus) -> None:
        current = self._store.run(run_id).status
        if current is not status:
            raise InvalidRequest(f"run {run_id} is {current}, not {status}")


def _arguments(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        arguments = decode_arguments(raw)
    except ValueError:
        raise _ArgumentError("arguments are not valid JSON") from None
    try:
        json.dumps(arguments, ensure_ascii=False).encode()
    except UnicodeEncodeError:
        raise _ArgumentError("arguments contain text that is not valid Unicode") from None
    errors = validate(arguments, schema)
    if errors:
        raise _ArgumentError("invalid arguments: " + "; ".join(errors))
    return arguments


def _canonical(arguments: str) -> str:
    try:
        return json.dumps(decode_arguments(arguments), sort_keys=True)
    except ValueError:
        return arguments


def _describe(call: Call) -> str:
    arguments = call.arguments if len(call.arguments) <= 160 else call.arguments[:160] + "..."
    return f"{call.name} {arguments}"


def _interrupted(call: Call) -> str:
    return f"{_describe(call)} was interrupted and its outcome is unknown"
