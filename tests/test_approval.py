import pytest

from codeagent.records import CallStatus, Mode, RunStatus
from codeagent.runtime import InvalidRequest

from .support import FIXED_ORDER, TEST_COMMAND, Agent, call, final, reply


def write_then_test() -> list[dict]:
    return [
        reply(call("c1", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("c2", "bash", command=TEST_COMMAND, cwd=".")),
        final(),
    ]


def test_safe_mode_stops_before_a_write_and_continues_after_approval(tmp_path):
    agent = Agent(tmp_path, write_then_test())
    original = agent.file("order.py")

    run = agent.start(mode=Mode.SAFE)
    assert run.status is RunStatus.AWAITING_APPROVAL
    assert agent.file("order.py") == original

    agent.restart()
    run = agent.runtime.decide_action(agent.run_id, approve=True)
    assert run.status is RunStatus.AWAITING_APPROVAL
    assert agent.file("order.py") == FIXED_ORDER
    assert agent.store.calls_with_status(agent.run_id, CallStatus.AWAITING_APPROVAL)[0].name == "bash"

    run = agent.runtime.decide_action(agent.run_id, approve=True)
    assert run.status is RunStatus.COMPLETED
    assert [decision["verdict"] for decision in agent.store.decisions(agent.run_id)] == ["approved", "approved"]


def test_denial_reaches_the_model_with_the_reason(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "delete_file", path="test_order.py")),
            reply(call("c2", "read_file", path="order.py")),
            final(),
        ],
    )
    agent.start(mode=Mode.SAFE)
    run = agent.runtime.decide_action(agent.run_id, approve=False, reason="tests must stay")
    assert run.status is RunStatus.COMPLETED
    assert (agent.workspace / "test_order.py").exists()
    denial = agent.tool_messages()[0]
    assert denial["tool_call_id"] == "c1"
    assert "the user denied this action: tests must stay" in denial["content"]


def test_auto_mode_runs_writes_and_bash_but_still_confirms_deletes(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "write_file", path="order.py", content=FIXED_ORDER)),
            reply(call("c2", "bash", command=TEST_COMMAND, cwd=".")),
            reply(call("c3", "delete_file", path="order.py")),
            final(),
        ],
    )
    run = agent.start(mode=Mode.AUTO)
    assert run.status is RunStatus.AWAITING_APPROVAL
    assert agent.store.calls_with_status(agent.run_id, CallStatus.AWAITING_APPROVAL)[0].name == "delete_file"
    assert agent.file("order.py") == FIXED_ORDER


def test_approval_does_not_bypass_the_policy_recheck(tmp_path):
    agent = Agent(tmp_path, write_then_test())
    agent.start(mode=Mode.SAFE)
    outside = tmp_path / "outside.py"
    outside.write_text("")
    (agent.workspace / "order.py").unlink()
    (agent.workspace / "order.py").symlink_to(outside)
    agent.runtime.decide_action(agent.run_id, approve=True)
    assert "symlinks are not allowed" in agent.tool_messages()[0]["content"]
    assert outside.read_text() == ""


def test_a_run_keeps_the_policy_it_started_with(tmp_path):
    agent = Agent(tmp_path, write_then_test())
    agent.start(mode=Mode.SAFE)
    agent.policy_path.write_text(agent.policy_path.read_text().replace('deny = ["private"]', 'deny = ["."]'))
    agent.restart()
    run = agent.runtime.decide_action(agent.run_id, approve=True)
    assert agent.file("order.py") == FIXED_ORDER
    assert run.status is RunStatus.AWAITING_APPROVAL


def test_decisions_require_a_matching_state(tmp_path):
    agent = Agent(tmp_path, [final()])
    agent.start()
    with pytest.raises(InvalidRequest):
        agent.runtime.decide_action(agent.run_id, approve=True)
