import pytest

from codeagent.records import Mode, RunStatus
from codeagent.runtime import Resolution

from .support import FIXED_ORDER, TEST_COMMAND, Agent, SimulatedCrash, call, final, reply


def reads(count: int) -> list[dict]:
    return [reply(call(f"r{index}", "read_file", path="order.py")) for index in range(count)]


def test_model_requests_limit_stops_the_run(tmp_path):
    agent = Agent(tmp_path, [*reads(3), final()], budget={"model_requests": 2})
    run = agent.start()
    assert run.status is RunStatus.EXHAUSTED
    assert "model_requests" in run.note
    assert run.used.model_requests == 2


def test_batch_larger_than_the_tool_call_budget_runs_no_call(tmp_path):
    batch = reply(*(call(f"c{index}", "read_file", path="order.py") for index in range(4)))
    agent = Agent(tmp_path, [batch, final()], budget={"tool_calls": 3})
    run = agent.start()
    assert run.status is RunStatus.EXHAUSTED
    assert agent.reporter.executed == []
    assert run.used.tool_calls == 0
    assert all("not executed" in message["content"] for message in agent.tool_messages())
    assert len(agent.tool_messages()) == 4


def test_batch_with_more_side_effects_than_left_runs_no_call(tmp_path):
    batch = reply(
        call("c1", "read_file", path="order.py"),
        call("c2", "write_file", path="order.py", content=FIXED_ORDER),
        call("c3", "bash", command=TEST_COMMAND, cwd="."),
    )
    agent = Agent(tmp_path, [batch, final()], budget={"side_effects": 1})
    original = agent.file("order.py")
    run = agent.start()
    assert run.status is RunStatus.EXHAUSTED
    assert "side_effects" in run.note
    assert agent.reporter.executed == []
    assert agent.file("order.py") == original


def test_side_effect_budget_is_spent_per_execution(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "write_file", path="order.py", content=FIXED_ORDER)),
            reply(call("c2", "bash", command=TEST_COMMAND, cwd=".")),
            final(),
        ],
        budget={"side_effects": 1},
    )
    run = agent.start()
    assert run.status is RunStatus.EXHAUSTED
    assert [executed.name for executed in agent.reporter.executed] == ["write_file"]
    assert run.used.side_effects == 1


def test_output_is_cut_to_the_remaining_budget_and_marked_incomplete(tmp_path):
    agent = Agent(tmp_path, [*reads(2), final()], budget={"output_bytes": 10})
    run = agent.start()
    first = agent.reporter.results[0][1]
    assert first["content"] == agent.file("order.py")[:10]
    assert "incomplete" in first["truncated"]
    assert run.status is RunStatus.EXHAUSTED
    assert "output_bytes" in run.note
    assert [executed.call_id for executed in agent.reporter.executed] == ["r0"]
    assert run.used.output_bytes == 10


def test_counters_survive_restarts_and_decisions(tmp_path):
    agent = Agent(
        tmp_path,
        [
            reply(call("c1", "read_file", path="order.py")),
            reply(call("c2", "write_file", path="order.py", content=FIXED_ORDER)),
            final(),
        ],
    )
    paused = agent.start(mode=Mode.SAFE)
    agent.restart()
    assert agent.run.used == paused.used
    assert paused.used.model_requests == 2
    assert paused.used.tool_calls == 2
    run = agent.runtime.decide_action(agent.run_id, approve=True)
    assert run.used.model_requests == 3
    assert run.used.side_effects == 1
    assert run.used.output_bytes == paused.used.output_bytes


def test_budget_spent_before_a_crash_is_not_returned(tmp_path):
    agent = Agent(tmp_path, [reply(call("c1", "write_file", path="order.py", content=FIXED_ORDER)), final()])
    agent.crash_after("write_file")
    with pytest.raises(SimulatedCrash):
        agent.start()
    agent.heal()
    assert agent.run.used.side_effects == 1
    assert agent.run.used.tool_calls == 1
    agent.runtime.advance(agent.run_id)
    run = agent.runtime.resolve(agent.run_id, Resolution.RETRY, adopt=True)
    assert run.status is RunStatus.COMPLETED
    assert run.used.side_effects == 2
    assert run.used.tool_calls == 1


def test_exhausted_side_effect_budget_stops_a_retry(tmp_path):
    agent = Agent(
        tmp_path,
        [reply(call("c1", "write_file", path="order.py", content=FIXED_ORDER)), final()],
        budget={"side_effects": 1},
    )
    agent.crash_after("write_file")
    with pytest.raises(SimulatedCrash):
        agent.start()
    agent.heal()
    agent.runtime.advance(agent.run_id)
    run = agent.runtime.resolve(agent.run_id, Resolution.RETRY, adopt=True)
    assert run.status is RunStatus.EXHAUSTED
    assert run.used.side_effects == 1


def test_output_running_out_mid_batch_answers_every_call(tmp_path):
    batch = reply(*(call(f"c{index}", "read_file", path="order.py") for index in range(3)))
    agent = Agent(tmp_path, [batch, final()], budget={"output_bytes": 50})
    run = agent.start()
    assert run.status is RunStatus.EXHAUSTED
    assert [executed.call_id for executed in agent.reporter.executed] == ["c0"]
    assert [message["tool_call_id"] for message in agent.tool_messages()] == ["c0", "c1", "c2"]


def test_calls_the_policy_denies_need_no_side_effect_budget(tmp_path):
    batch = reply(
        call("p1", "read_file", path="order.py"),
        call("p2", "write_file", path="order.py", content=FIXED_ORDER),
        call("p3", "write_file", path="order.py", content=FIXED_ORDER),
    )
    agent = Agent(tmp_path, [batch, final(), final()], budget={"side_effects": 1})
    agent.start(plan_first=True)
    assert [executed.call_id for executed in agent.reporter.executed] == ["p1"]
