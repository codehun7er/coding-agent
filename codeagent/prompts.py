from collections.abc import Sequence

from .plan import Check, CheckKind, Task

_PLACEHOLDERS = {CheckKind.READ: "<path>", CheckKind.CHANGED: "<path>", CheckKind.COMMAND: "<allowed command>"}


def system_prompt(commands: Sequence[str]) -> str:
    allowed = "\n".join(f"  - {command}" for command in commands)
    return (
        "You are a coding agent. You change a project only through the tools you are given.\n"
        "\n"
        "Rules:\n"
        "- Paths are relative to the workspace root. Absolute paths, '..' and symlinks are rejected.\n"
        "- The bash tool runs only these exact commands; use cwd '.' for the workspace root:\n"
        f"{allowed}\n"
        "- Shell syntax such as pipes, redirects, substitutions, '&&' and ';' is rejected.\n"
        "- The runtime checks every call. A denied or failed call returns an error with the reason; "
        "read it and choose another step.\n"
        "- Leave tests unchanged unless the user asks to change them.\n"
        "- When the task is finished and verified, reply with a short summary and no tool calls."
    )


def planning() -> str:
    checks = "\n".join(
        f'- {{"kind": "{kind}", "target": "{target}"}}: {Check(kind, target).describe()};'
        for kind, target in _PLACEHOLDERS.items()
    )
    return (
        "Plan mode is on. Study the project with list_files and read_file, the only tools that work now, "
        "then call submit_plan once. Every step needs an id, a description, the tools it will use, "
        "the ids of the steps it depends on, and a check the runtime confirms from recorded tool calls:\n"
        f"{checks}\n"
        "Nothing runs until the user approves the plan."
    )


SUBMIT_PLAN_FIRST = "Plan mode is still on and a text reply does not end it. Call submit_plan."

PLAN_APPROVED = "The user approved the plan. Work on the tasks one at a time as the runtime assigns them."

ALL_TASKS_DONE = "All tasks are verified. Reply with a short summary of the work and no tool calls."

TOOLS_AFTER_LAST_TASK = "all tasks are verified; reply with a final summary instead of calling tools"


def plan_rejected(feedback: str) -> str:
    return f"The user rejected the plan: {feedback or 'no reason given'}. Revise it and call submit_plan again."


def task_brief(task: Task) -> str:
    return (
        f"Current task {task.id}: {task.description}\n"
        f"The runtime marks it done when {task.check.describe()}. "
        f'Call complete_task with task_id "{task.id}" once that holds.'
    )


def unfinished(task_id: str) -> str:
    return f"Task {task_id} is not verified yet. Keep working on it and call complete_task when its check holds."
