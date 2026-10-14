import json

import pytest

from codeagent.plan import PlanStatus
from codeagent.records import Mode, Phase, RunStatus

from .support import FIXED_ORDER, ROOT, TEST_COMMAND, Agent, call, final, reply

PLAN_SCRIPT = json.loads((ROOT / "examples" / "fix_order_plan.json").read_text())
PLAN = PLAN_SCRIPT[1]["tool_calls"][0]["arguments"]


def sneaky_planning() -> list[dict]:
    return [
        reply(
            call("p1", "read_file", path="order.py"),
            call("p2", "write_file", path="order.py", content=FIXED_ORDER),
            call("p3", "bash", command=TEST_COMMAND, cwd="."),
            call("p4", "delete_file", path="test_order.py"),
        ),
        reply(call("p5", "submit_plan", **PLAN)),
    ]


def test_nothing_with_a_side_effect_happens_before_the_plan_is_approved(tmp_path):
    agent = Agent(tmp_path, sneaky_planning())
    before = {path.name: path.read_bytes() for path in agent.workspace.iterdir()}

    run = agent.start(plan_first=True)

    assert run.status is RunStatus.AWAITING_PLAN
    assert {path.name: path.read_bytes() for path in agent.workspace.iterdir()} == before
    assert [executed.name for executed in agent.reporter.executed] == ["read_file", "submit_plan"]
    assert run.used.side_effects == 0
    assert agent.store.file_changes(agent.run_id) == []
    denials = [message["content"] for message in agent.tool_messages()[1:4]]
    assert all("plan mode is read-only" in denial for denial in denials)


def test_planning_offers_only_read_tools_and_submit_plan(tmp_path):
    agent = Agent(tmp_path, sneaky_planning())
    agent.start(plan_first=True)
    assert agent.model.offered[0] == ["list_files", "read_file", "submit_plan"]


def test_plan_is_stored_as_structured_data_with_the_run(tmp_path):
    agent = Agent(tmp_path, PLAN_SCRIPT)
    agent.start(plan_first=True)
    agent.restart()
    proposal = agent.store.latest_plan(agent.run_id)
    assert proposal.status is PlanStatus.PROPOSED
    assert proposal.plan.goal == PLAN["goal"]
    assert [step.id for step in proposal.plan.steps] == ["T1", "T2", "T3"]
    assert proposal.plan.steps[2].tools == ("bash",)
    assert proposal.plan.steps[2].depends_on == ("T2",)
    assert proposal.plan.risks and proposal.plan.assumptions and proposal.plan.done_criteria


def test_text_reply_does_not_end_planning(tmp_path):
    agent = Agent(tmp_path, [final("I would edit order.py"), reply(call("p1", "submit_plan", **PLAN))])
    run = agent.start(plan_first=True)
    assert run.status is RunStatus.AWAITING_PLAN
    nudges = [m["content"] for m in agent.store.messages(agent.run_id) if m["role"] == "user"]
    assert any("Call submit_plan" in nudge for nudge in nudges)


def test_rejected_plan_goes_back_to_the_model_with_feedback(tmp_path):
    revised = {**PLAN, "goal": "Revised goal"}
    agent = Agent(
        tmp_path,
        [reply(call("p1", "submit_plan", **PLAN)), reply(call("p2", "submit_plan", **revised))],
    )
    agent.start(plan_first=True)
    run = agent.runtime.decide_plan(agent.run_id, approve=False, feedback="explain the rounding")
    assert run.status is RunStatus.AWAITING_PLAN
    assert run.phase is Phase.PLANNING
    assert agent.store.latest_plan(agent.run_id).plan.goal == "Revised goal"
    assert any("explain the rounding" in message["content"] for message in agent.store.messages(agent.run_id))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda plan: plan["steps"][0].update(depends_on=["T3"]), "cycle"),
        (lambda plan: plan["steps"][1].update(id="T1"), "unique"),
        (
            lambda plan: plan["steps"][0].update(tools=["teleport"]),
            "teleport; valid tools are bash, complete_task, delete_file, list_files, read_file, write_file",
        ),
        (lambda plan: plan["steps"][2].update(depends_on=["T9"]), "unknown steps"),
        (lambda plan: plan.pop("risks"), "$.risks is required"),
    ],
)
def test_invalid_plans_are_returned_to_the_model(tmp_path, change, error):
    broken = json.loads(json.dumps(PLAN))
    change(broken)
    agent = Agent(tmp_path, [reply(call("p1", "submit_plan", **broken)), reply(call("p2", "submit_plan", **PLAN))])
    agent.start(plan_first=True)
    assert error in agent.tool_messages()[0]["content"]
    assert agent.store.latest_plan(agent.run_id).plan.goal == PLAN["goal"]


def test_approval_starts_execution_but_safe_mode_still_confirms_each_action(tmp_path):
    agent = Agent(tmp_path, PLAN_SCRIPT)
    agent.start(plan_first=True, mode=Mode.SAFE)
    run = agent.runtime.decide_plan(agent.run_id, approve=True)
    assert run.phase is Phase.EXECUTING
    assert run.status is RunStatus.AWAITING_APPROVAL
    assert "write_file" in run.note
