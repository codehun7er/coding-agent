from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from . import prompts
from .plan import PLAN_PARAMETERS, Plan, PlanError, PlanStatus, Task, TaskStatus, Unverified, next_task
from .records import Phase, Run
from .schema import function_spec, object_schema
from .store import Store
from .tools import Outcome


@dataclass(frozen=True)
class ControlTool:
    """A tool that changes plan or task state instead of the workspace."""

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[Run, dict[str, Any]], Outcome]

    def spec(self) -> dict[str, Any]:
        return function_spec(self.name, self.description, self.parameters)


class Planner:
    """Plan mode and task execution: the control tools, the plan decision, task assignment and task verification.

    Every method joins the caller's transaction, so plan and task state move together with the run around them.
    Methods that change tasks return the new task list for the caller to report.
    """

    def __init__(self, store: Store, tool_names: Collection[str]) -> None:
        self._store = store
        self._submit_plan = ControlTool(
            name="submit_plan",
            description="Submit the structured plan for the user to approve or reject.",
            parameters=PLAN_PARAMETERS,
            execute=self._submit,
        )
        self._complete_task = ControlTool(
            name="complete_task",
            description="Ask the runtime to verify the active task and mark it done.",
            parameters=object_schema(
                {"task_id": {"type": "string", "minLength": 1}, "summary": {"type": "string", "minLength": 1}},
                ("task_id", "summary"),
            ),
            execute=self._complete,
        )
        self._tools = {tool.name: tool for tool in (self._submit_plan, self._complete_task)}
        self._plan_tools = (*tool_names, self._complete_task.name)

    def get(self, name: str) -> ControlTool | None:
        return self._tools.get(name)

    def specs(self, run: Run) -> list[dict[str, Any]]:
        if run.phase is Phase.PLANNING:
            return [self._submit_plan.spec()]
        if self._store.tasks(run.id):
            return [self._complete_task.spec()]
        return []

    def awaits_decision(self, run: Run) -> bool:
        if run.phase is not Phase.PLANNING:
            return False
        proposal = self._store.latest_plan(run.id)
        return proposal is not None and proposal.status is PlanStatus.PROPOSED

    def decide(self, run_id: str, *, approve: bool, feedback: str) -> int:
        """Apply the user's verdict on the proposed plan and return its version."""
        proposal = self._store.latest_plan(run_id)
        if approve:
            self._store.set_plan_status(run_id, proposal.version, PlanStatus.APPROVED)
            self._store.create_tasks(run_id, proposal.plan.steps)
            self._store.update_run(run_id, phase=Phase.EXECUTING)
            self._store.append_message(run_id, {"role": "user", "content": prompts.PLAN_APPROVED})
        else:
            self._store.set_plan_status(run_id, proposal.version, PlanStatus.REJECTED, feedback)
            self._store.append_message(run_id, {"role": "user", "content": prompts.plan_rejected(feedback)})
        return proposal.version

    def assign(self, run: Run) -> list[Task] | None:
        """Activate the next task whose dependencies are done and brief the model on it."""
        if run.phase is not Phase.EXECUTING or run.active_task is not None:
            return None
        task = next_task(self._store.tasks(run.id))
        if task is None:
            return None
        self._store.set_task_status(run.id, task.id, TaskStatus.ACTIVE)
        self._store.update_run(run.id, active_task=task.id)
        self._store.append_message(run.id, {"role": "user", "content": prompts.task_brief(task)})
        return self._store.tasks(run.id)

    def refusal_without_task(self, run: Run) -> str | None:
        """The refusal for a workspace tool call made after every task is done, or None while calls have a task."""
        if run.phase is Phase.EXECUTING and run.active_task is None and self._store.tasks(run.id):
            return prompts.TOOLS_AFTER_LAST_TASK
        return None

    def blocker(self, run: Run) -> str | None:
        """What still stands between the run and a final answer, phrased for the model; None when it may finish."""
        if run.phase is Phase.PLANNING:
            return prompts.SUBMIT_PLAN_FIRST
        if run.active_task is not None:
            return prompts.unfinished(run.active_task)
        return None

    def abandon(self, run: Run) -> list[Task] | None:
        """Mark the active task failed when the run stops before finishing it."""
        if run.active_task is None:
            return None
        self._store.set_task_status(run.id, run.active_task, TaskStatus.FAILED)
        return self._store.tasks(run.id)

    def _submit(self, run: Run, arguments: dict[str, Any]) -> Outcome:
        if run.phase is not Phase.PLANNING:
            return Outcome.failure("the plan is already approved; submit_plan works only in plan mode")
        try:
            plan = Plan.parse(arguments, tools=self._plan_tools)
        except PlanError as error:
            return Outcome.failure(f"invalid plan: {error}")
        version = self._store.save_plan(run.id, plan)
        return Outcome(True, {"version": version, "status": "waiting for the user to approve or reject the plan"})

    def _complete(self, run: Run, arguments: dict[str, Any]) -> Outcome:
        task_id = arguments["task_id"]
        if run.active_task is None:
            return Outcome.failure("there is no active task")
        if task_id != run.active_task:
            return Outcome.failure(f"task {task_id} is not active; the active task is {run.active_task}")
        tasks = self._store.tasks(run.id)
        task = next(task for task in tasks if task.id == task_id)
        try:
            proof = task.check.verify(self._store.task_evidence(run.id, task_id))
        except Unverified as unverified:
            return Outcome.failure(f"task {task_id} is not verified: {unverified}")
        evidence = f"{proof.name} call {proof.call_id} shows that {task.check.describe()}"
        self._store.set_task_status(run.id, task_id, TaskStatus.DONE, evidence)
        self._store.update_run(run.id, active_task=None)
        last = all(other.status is TaskStatus.DONE for other in tasks if other.id != task_id)
        follow_up = prompts.ALL_TASKS_DONE if last else "the runtime assigns the next task"
        return Outcome(True, {"task_id": task_id, "evidence": evidence, "next": follow_up})
