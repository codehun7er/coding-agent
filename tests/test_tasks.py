import json

from codeagent.plan import TaskStatus, checklist
from codeagent.records import CallStatus, Mode, Phase, RunStatus

from .support import FIXED_ORDER, ROOT, TEST_COMMAND, Agent, call, final, reply

PLAN_SCRIPT = json.loads((ROOT / "examples" / "fix_order_plan.json").read_text())
PLANNING = PLAN_SCRIPT[:2]


def approved(agent: Agent, mode: Mode = Mode.AUTO):
    agent.start(plan_first=True, mode=mode)
    return agent.runtime.decide_plan(agent.run_id, approve=True)


def statuses(agent: Agent) -> dict[str, TaskStatus]:
    return {task.id: task.status for task in agent.store.tasks(agent.run_id)}


def test_tasks_run_in_dependency_order_and_finish_with_evidence(tmp_path):
    agent = Agent(tmp_path, PLAN_SCRIPT)
    run = approved(agent)

    assert run.status is RunStatus.COMPLETED
    assert agent.file("order.py") == FIXED_ORDER
    tasks = agent.store.tasks(agent.run_id)
    assert [task.status for task in tasks] == [TaskStatus.DONE] * 3
    assert "bash call t5" in tasks[2].evidence
    assert tasks[1].depends_on == ("T1",) and tasks[2].depends_on == ("T2",)


def test_execution_calls_are_bound_to_the_active_task(tmp_path):
    agent = Agent(tmp_path, PLAN_SCRIPT)
    approved(agent)
    binding = {c.call_id: (c.phase, c.task_id) for c in agent.store.calls(agent.run_id)}
    assert binding["p2"] == (Phase.PLANNING, None)
    assert binding["t1"] == (Phase.EXECUTING, "T1")
    assert binding["t3"] == (Phase.EXECUTING, "T2")
    assert binding["t5"] == (Phase.EXECUTING, "T3")


def test_checklist_is_reported_as_tasks_progress(tmp_path):
    agent = Agent(tmp_path, PLAN_SCRIPT)
    approved(agent)
    rendered = [checklist(tasks) for tasks in agent.reporter.checklists]
    assert rendered[0].splitlines()[0] == "- [ ] T1 Read order.py to find how the total is computed  <- in progress"
    assert rendered[-1].splitlines() == [
        "- [x] T1 Read order.py to find how the total is computed",
        "- [x] T2 Rewrite calculate_total to apply a percentage discount and raise ValueError outside 0..100",
        "- [x] T3 Run the test suite",
    ]


def test_the_models_word_does_not_complete_a_task(tmp_path):
    agent = Agent(
        tmp_path,
        [
            *PLANNING,
            reply(call("x1", "complete_task", task_id="T1", summary="I read it")),
            reply(call("x2", "complete_task", task_id="T2", summary="skip ahead")),
            reply(call("x3", "read_file", path="order.py")),
            reply(call("x4", "complete_task", task_id="T1", summary="read for real")),
            final("stopping early"),
            final("still stopping"),
        ],
    )
    run = approved(agent)
    results = {m["tool_call_id"]: m["content"] for m in agent.tool_messages()}
    assert "not verified: no successful read_file of order.py" in results["x1"]
    assert "task T2 is not active" in results["x2"]
    assert '"ok": true' in results["x4"]
    assert statuses(agent) == {"T1": TaskStatus.DONE, "T2": TaskStatus.FAILED, "T3": TaskStatus.PENDING}
    assert run.status is RunStatus.FAILED
    nudges = [m["content"] for m in agent.store.messages(agent.run_id) if "is not verified yet" in m["content"]]
    assert len(nudges) == 1
    assert "<- failed" in checklist(agent.reporter.checklists[-1])


def test_command_check_needs_a_passing_run_after_the_last_change(tmp_path):
    agent = Agent(
        tmp_path,
        [
            *PLAN_SCRIPT[:6],
            reply(call("y1", "bash", command=TEST_COMMAND, cwd=".")),
            reply(call("y2", "write_file", path="order.py", content=FIXED_ORDER + "\n")),
            reply(call("y3", "complete_task", task_id="T3", summary="tests passed")),
            reply(call("y4", "bash", command=TEST_COMMAND, cwd=".")),
            reply(call("y5", "complete_task", task_id="T3", summary="tests passed again")),
            final(),
        ],
    )
    run = approved(agent)
    results = {m["tool_call_id"]: m["content"] for m in agent.tool_messages()}
    assert "files changed after the last run" in results["y3"]
    assert run.status is RunStatus.COMPLETED


def test_finished_tasks_are_not_started_again_after_a_restart(tmp_path):
    agent = Agent(tmp_path, PLAN_SCRIPT)
    approved(agent, mode=Mode.SAFE)
    agent.runtime.decide_action(agent.run_id, approve=True)
    assert statuses(agent) == {"T1": TaskStatus.DONE, "T2": TaskStatus.DONE, "T3": TaskStatus.ACTIVE}
    assert agent.store.calls_with_status(agent.run_id, CallStatus.AWAITING_APPROVAL)[0].name == "bash"

    agent.restart()
    run = agent.runtime.decide_action(agent.run_id, approve=True)

    assert run.status is RunStatus.COMPLETED
    briefs = [m["content"] for m in agent.store.messages(agent.run_id) if m["content"].startswith("Current task")]
    assert [brief.split(":")[0] for brief in briefs] == ["Current task T1", "Current task T2", "Current task T3"]
    assert [executed.call_id for executed in agent.reporter.executed] == ["t5", "t6"]


def test_a_replayed_result_does_not_prove_a_new_task(tmp_path):
    agent = Agent(
        tmp_path,
        [
            *PLAN_SCRIPT[:6],
            reply(call("t1", "read_file", path="order.py")),
            reply(call("y1", "complete_task", task_id="T3", summary="the old read counts?")),
            final(),
            final(),
        ],
    )
    plan = json.loads(json.dumps(PLAN_SCRIPT[1]))
    plan["tool_calls"][0]["arguments"]["steps"][2]["check"] = {"kind": "read", "target": "order.py"}
    agent.steps[1] = plan
    agent.restart()
    approved(agent)
    results = {m["tool_call_id"]: m["content"] for m in agent.tool_messages()}
    assert "not verified" in results["y1"]


def test_tool_calls_after_every_task_is_done_are_refused(tmp_path):
    agent = Agent(
        tmp_path,
        [*PLAN_SCRIPT[:8], reply(call("z1", "write_file", path="extra.py", content="")), final()],
    )
    run = approved(agent)
    assert run.status is RunStatus.COMPLETED
    assert "all tasks are verified" in agent.tool_messages()[-1]["content"]
    assert not (agent.workspace / "extra.py").exists()
